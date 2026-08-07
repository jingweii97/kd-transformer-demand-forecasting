"""Read-only validation diagnostics for teacher selection and KD planning.

This script deliberately performs no training and reads only the configured
validation horizon. It uses the same ``get_predictions`` and hierarchical
WRMSSE implementation as ``scripts/evaluate_models.py``.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cache import load_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset
from models.student import M5TransformerStudent
from scripts.evaluate_models import (
    FULL_M5_SERIES_COUNT,
    compute_hierarchical_wrmsse,
    compute_mase,
    compute_mase_scales,
    compute_wrmsse_weights_and_scales,
    get_predictions,
)
from utils.config import load_config
from utils.paths import get_dataset_dir, resolve_path


DEFAULT_STUDENT = "outputs/student/no_kd/exp_full_phase1/best_student.ckpt"
DEFAULT_HUBER = "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=05-val_loss=val_loss=0.606112.ckpt"
DEFAULT_QUANTILE = "outputs/teacher/tft64_optimized/tft64-opt-epoch=epoch=05-val_loss=val_loss=0.474301.ckpt"
DEFAULT_HUBER_ENSEMBLE = [
    DEFAULT_HUBER,
    "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=08-val_loss=val_loss=0.612628.ckpt",
    "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=13-val_loss=val_loss=0.606753.ckpt",
]

CATEGORICAL_COLUMNS = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "weekday", "month", "year", "event_name_1", "event_type_1",
]


def checkpoint_info(path):
    """Resolve a checkpoint and return immutable provenance information."""
    resolved = resolve_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    digest = hashlib.sha256()
    with open(resolved, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": os.path.abspath(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": os.path.getsize(resolved),
    }


def to_device(batch_x, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch_x.items()
    }


def as_point_prediction(value):
    """Normalize TFT/student point-output tensors to [batch, horizon]."""
    if value.ndim == 3 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def tensor_summary(value):
    value = value.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
    }


def sample_rows(value, count=3):
    value = as_point_prediction(value).detach().float().cpu().numpy()
    return value[: min(count, len(value))].tolist()


def prepare_partition(frame):
    frame = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).astype("category")
    return frame


def raw_targets_for_batch(frame, decoded_index, batch_x, batch_size, horizon):
    """Recover raw sales for the exact ids and decoder times of one batch.

    ``groups`` is an encoded tensor and may use the label encoder's NaN token.
    ``decoded_index`` is the authoritative id mapping used by the evaluator.
    """
    if len(decoded_index) < batch_size:
        raise AssertionError("Decoded index has fewer rows than the matched batch")
    matched_rows = decoded_index.iloc[:batch_size].reset_index(drop=True)
    series_ids = matched_rows["id"].astype(str).tolist()
    starts = batch_x["decoder_time_idx"][:, 0].detach().cpu().numpy().astype(int)
    decoded_starts = matched_rows["time_idx_first_prediction"].to_numpy(dtype=int)
    if not np.array_equal(starts, decoded_starts):
        raise AssertionError("First validation batch is not aligned with decoded_index")

    raw = frame[["id", "time_idx", "sales"]].copy()
    raw["id"] = raw["id"].astype(str)
    lookup = raw.set_index(["id", "time_idx"])["sales"]
    rows = []
    for series_id, start in zip(series_ids, starts):
        index = pd.MultiIndex.from_product(
            [[series_id], range(start, start + horizon)], names=["id", "time_idx"]
        )
        values = lookup.reindex(index).to_numpy()
        if np.isnan(values).any():
            raise AssertionError(
                f"Raw target lookup failed for id={series_id}, start={start}"
            )
        rows.append(values)
    return np.asarray(rows, dtype=np.float32), series_ids, starts.tolist()


def effective_huber_audit(
    teacher, student, loader, raw_frame, decoded_index, training_dataset, device, horizon
):
    """Audit the exact tensor domain and reduction for one matched batch."""
    batch_x, batch_y = next(iter(loader))
    target = batch_y[0] if isinstance(batch_y, (tuple, list)) else batch_y
    raw_targets, series_ids, starts = raw_targets_for_batch(
        raw_frame, decoded_index, batch_x, len(target), horizon
    )
    target_cpu = target.detach().float().cpu()
    if raw_targets.shape != tuple(target_cpu.shape):
        raise AssertionError(
            f"Raw/loader target shape mismatch: raw={raw_targets.shape}, loader={tuple(target_cpu.shape)}"
        )
    raw_target_gap = float(np.max(np.abs(raw_targets - target_cpu.numpy())))

    batch_x_device = to_device(batch_x, device)
    target_device = target.to(device)
    teacher.eval()
    student.eval()
    with torch.no_grad():
        teacher_output = teacher(batch_x_device)
        teacher_raw = teacher_output.prediction
        teacher_point = teacher.to_prediction(teacher_output)
        student_point = student(batch_x_device)

        teacher_elementwise = teacher.loss.loss(teacher_raw, target_device)
        teacher_elementwise = as_point_prediction(teacher_elementwise)
        student_point = as_point_prediction(student_point)
        student_elementwise = F.huber_loss(
            student_point,
            target_device,
            reduction="none",
            delta=float(student.loss_fn.delta),
        )

    lengths = batch_x.get("decoder_lengths")
    if lengths is None:
        valid_mask = torch.ones_like(target_cpu, dtype=torch.bool)
    else:
        positions = torch.arange(horizon).unsqueeze(0)
        valid_mask = positions < lengths.detach().cpu().unsqueeze(1)
    if valid_mask.shape != target_cpu.shape:
        raise AssertionError("Decoder validity mask does not match target shape")

    valid_count = int(valid_mask.sum())
    teacher_reduced = float(teacher_elementwise.detach().cpu()[valid_mask].mean())
    student_reduced = float(student_elementwise.detach().cpu()[valid_mask].mean())
    target_scale = batch_x.get("target_scale")
    report = {
        "purpose": "One matched validation batch only; this is not a retraining decision.",
        "batch_series_ids": series_ids[:3],
        "decoder_start_times": starts[:3],
        "horizon": horizon,
        "valid_elements": valid_count,
        "total_elements": int(valid_mask.numel()),
        "target_normalizer": repr(training_dataset.target_normalizer),
        "teacher_huber_delta": float(getattr(teacher.loss, "delta", float("nan"))),
        "student_huber_delta": float(student.loss_fn.delta),
        "raw_sales_vs_dataset_target_max_abs_gap": raw_target_gap,
        "raw_sales_vs_dataset_target_equal": bool(np.isclose(raw_target_gap, 0.0, atol=1e-5)),
        "summaries": {
            "raw_sales_target": {
                "shape": list(raw_targets.shape),
                "min": float(raw_targets.min()),
                "max": float(raw_targets.max()),
                "mean": float(raw_targets.mean()),
            },
            "dataset_loss_target": tensor_summary(target_cpu),
            "teacher_forward_prediction": tensor_summary(teacher_raw),
            "teacher_point_prediction": tensor_summary(teacher_point),
            "student_point_prediction": tensor_summary(student_point),
            "teacher_elementwise_huber": tensor_summary(teacher_elementwise),
            "student_elementwise_huber": tensor_summary(student_elementwise),
            "target_scale": tensor_summary(target_scale) if isinstance(target_scale, torch.Tensor) else None,
        },
        "reduced_losses_on_valid_mask": {
            "teacher_manual_mean": teacher_reduced,
            "student_manual_mean": student_reduced,
        },
        "samples_first_three_series": {
            "raw_sales_target": raw_targets[:3].tolist(),
            "dataset_loss_target": sample_rows(target_cpu),
            "teacher_forward_prediction": sample_rows(teacher_raw),
            "teacher_point_prediction": sample_rows(teacher_point),
            "student_point_prediction": sample_rows(student_point),
            "teacher_elementwise_huber": sample_rows(teacher_elementwise),
            "student_elementwise_huber": sample_rows(student_elementwise),
            "valid_mask": valid_mask[:3].int().tolist(),
        },
    }
    return report


def forecast_metrics(name, forecasts, actuals, gt_frame, weights_dict, scales_dict, mase_scales, series_ids):
    """Evaluate one predefined forecast with the common authoritative evaluator."""
    if forecasts.shape != actuals.shape:
        raise AssertionError(f"{name}: forecast shape {forecasts.shape} != actual shape {actuals.shape}")
    if not np.isfinite(forecasts).all():
        raise AssertionError(f"{name}: non-finite forecast")

    pred_frame = gt_frame.copy()
    pred_frame["sales"] = forecasts.reshape(-1)
    wrmsse, level_scores = compute_hierarchical_wrmsse(
        gt_frame, pred_frame, weights_dict, scales_dict
    )
    scales = np.asarray([mase_scales[series_id] for series_id in series_ids])
    actual_flat = actuals.reshape(-1)
    prediction_flat = forecasts.reshape(-1)
    total_actual = float(actual_flat.sum())
    return {
        "model": name,
        "WRMSSE": float(wrmsse),
        "MAE": float(np.mean(np.abs(prediction_flat - actual_flat))),
        "RMSE": float(np.sqrt(np.mean((prediction_flat - actual_flat) ** 2))),
        "WAPE": float(np.abs(prediction_flat - actual_flat).sum() / (total_actual + 1e-9)),
        "MASE": float(compute_mase(actuals, forecasts, scales)),
        "aggregate_percentage_bias": float(prediction_flat.sum() / (total_actual + 1e-9) - 1.0),
        "actual_total": total_actual,
        "predicted_total": float(prediction_flat.sum()),
        "level_scores": level_scores,
    }


def economic_weight_quantiles(actuals, forecasts_by_name, weights_dict, scales_dict, series_ids):
    """Compare Student and Huber across equal-count bottom-level value-weight quintiles."""
    level_weights = weights_dict["Level_12"]
    level_scales = scales_dict["Level_12"]
    weights = np.asarray([level_weights[series_id] for series_id in series_ids], dtype=float)
    scales = np.asarray([level_scales[series_id] for series_id in series_ids], dtype=float)
    weight_frame = pd.DataFrame({"id": series_ids, "economic_weight": weights})
    weight_frame["economic_weight_quintile"] = pd.qcut(
        weight_frame["economic_weight"].rank(method="first"),
        q=5,
        labels=["Q1_lowest", "Q2", "Q3", "Q4", "Q5_highest"],
    )

    rows = []
    for model_name, forecasts in forecasts_by_name.items():
        errors = forecasts - actuals
        per_series_rmsse = np.sqrt(np.mean(errors ** 2, axis=1) / scales)
        for quantile, group in weight_frame.groupby("economic_weight_quintile", observed=True):
            indices = group.index.to_numpy()
            local_weights = weights[indices]
            local_errors = errors[indices]
            weighted_rmsse = (
                float(np.average(per_series_rmsse[indices], weights=local_weights))
                if local_weights.sum() > 0
                else float("nan")
            )
            rows.append({
                "model": model_name,
                "economic_weight_quintile": str(quantile),
                "series_count": int(len(indices)),
                "economic_weight_sum": float(local_weights.sum()),
                "MAE": float(np.mean(np.abs(local_errors))),
                "RMSE": float(np.sqrt(np.mean(local_errors ** 2))),
                "mean_signed_error": float(np.mean(local_errors)),
                "bottom_level_weighted_RMSSE": weighted_rmsse,
            })
    return pd.DataFrame(rows)


def markdown_table(frame, include_index=False):
    """Render a compact Markdown table without a tabulate dependency."""
    printable = frame.reset_index() if include_index else frame.copy()
    headers = [str(column) for column in printable.columns]
    rows = printable.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(path, manifest, metrics, hierarchy, correlations, quantiles):
    with open(path, "w", encoding="utf-8") as report:
        report.write("# Teacher-strengthening diagnostic report\n\n")
        report.write("This is a read-only analysis of the validation split. No model training, "
                     "blend-weight search, or held-out evaluation was performed.\n\n")
        report.write("## Validation scope\n\n")
        report.write(f"- Validation days: `{manifest['validation_start']}`–`{manifest['validation_end']}`\n")
        report.write(f"- Series: `{manifest['series_count']}`\n")
        report.write(f"- Forecast horizon: `{manifest['horizon']}`\n\n")
        report.write("## Predefined teacher and ensemble metrics\n\n")
        report.write(markdown_table(metrics.drop(columns=["level_scores"], errors="ignore")))
        report.write("\n\n## Student versus Huber hierarchy decomposition\n\n")
        report.write(markdown_table(hierarchy))
        report.write("\n\n## Residual correlations\n\n")
        report.write(markdown_table(correlations, include_index=True))
        report.write("\n\n## Bottom-level economic-weight quintiles\n\n")
        report.write(markdown_table(quantiles))
        report.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Read-only validation audit for teacher strength")
    parser.add_argument("--env", default="dicc")
    parser.add_argument("--experiment", default="full")
    parser.add_argument("--student-checkpoint", default=DEFAULT_STUDENT)
    parser.add_argument("--huber-checkpoint", default=DEFAULT_HUBER)
    parser.add_argument("--quantile-checkpoint", default=DEFAULT_QUANTILE)
    parser.add_argument(
        "--huber-ensemble-checkpoints",
        nargs="+",
        default=DEFAULT_HUBER_ENSEMBLE,
        help="Predefined Huber snapshots only; no checkpoint search is performed.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(args.output_dir or f"artifacts/teacher_strength_audit_{timestamp}")
    if os.path.exists(output_dir):
        raise FileExistsError(f"Refusing to overwrite diagnostic output directory: {output_dir}")
    os.makedirs(output_dir)

    student_info = checkpoint_info(args.student_checkpoint)
    huber_info = checkpoint_info(args.huber_checkpoint)
    quantile_info = checkpoint_info(args.quantile_checkpoint)
    ensemble_infos = []
    seen_hashes = set()
    for checkpoint in args.huber_ensemble_checkpoints:
        info = checkpoint_info(checkpoint)
        if info["sha256"] not in seen_hashes:
            ensemble_infos.append(info)
            seen_hashes.add(info["sha256"])
    if huber_info["sha256"] not in seen_hashes:
        ensemble_infos.insert(0, huber_info)
        seen_hashes.add(huber_info["sha256"])
    if len(ensemble_infos) < 2:
        raise ValueError("At least two distinct Huber checkpoint hashes are required for the snapshot ensemble")

    train_end = cfg.dataset.splits.train.end
    validation_end = cfg.dataset.splits.validation.end
    horizon = cfg.dataset.prediction_window
    validation_start = validation_end - horizon + 1
    lookback = cfg.dataset.lookback_window
    if train_end >= validation_start:
        raise AssertionError("Validation window must begin after the training split")

    manifest = {
        "script": os.path.abspath(__file__),
        "device": str(device),
        "validation_start": int(validation_start),
        "validation_end": int(validation_end),
        "horizon": int(horizon),
        "held_out_set_used": False,
        "training_performed": False,
        "blend_weights": ["Huber only", "0.75 Huber + 0.25 Quantile", "0.50 Huber + 0.50 Quantile"],
        "checkpoints": {
            "student": student_info,
            "huber_epoch_5": huber_info,
            "quantile_epoch_5": quantile_info,
            "unique_huber_snapshot_ensemble": ensemble_infos,
        },
    }
    print(json.dumps(manifest, indent=2))

    dataset_dir = get_dataset_dir(cfg)
    stores = resolve_stores(cfg.environment.store_filter)
    if len(stores) != 10:
        raise AssertionError(f"Expected all 10 M5 store partitions, found {len(stores)}")
    training_dataset = build_timeseries_dataset(None, cfg, is_train=True)

    student = M5TransformerStudent.load_from_checkpoint(
        student_info["path"], training_dataset=training_dataset, map_location="cpu"
    ).to(device).eval()
    huber = TemporalFusionTransformer.load_from_checkpoint(huber_info["path"], map_location="cpu").to(device).eval()
    quantile = TemporalFusionTransformer.load_from_checkpoint(quantile_info["path"], map_location="cpu").to(device).eval()
    snapshot_models = [("Huber epoch 5", huber)]
    snapshot_index = 2
    for info in ensemble_infos:
        if info["sha256"] == huber_info["sha256"]:
            continue
        snapshot_models.append(
            (f"Huber snapshot {snapshot_index}", TemporalFusionTransformer.load_from_checkpoint(
                info["path"], map_location="cpu"
            ).to(device).eval())
        )
        snapshot_index += 1

    models = {
        "Student supervised": student,
        "Huber epoch 5": huber,
        "Quantile epoch 5": quantile,
        **dict(snapshot_models),
    }

    all_actuals = []
    all_ids = []
    all_predictions = {name: [] for name in models}
    loss_audit = None
    for store in stores:
        print(f"Running validation inference for {store}")
        raw_partition = load_from_cache(artifacts_dir=dataset_dir, store_filter=store)
        if raw_partition is None:
            raise FileNotFoundError(f"Missing cached partition for store {store}")
        frame = raw_partition[
            (raw_partition["time_idx"] >= validation_start - lookback)
            & (raw_partition["time_idx"] <= validation_end)
        ].copy()
        del raw_partition
        frame = prepare_partition(frame)
        dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, frame, predict=True, stop_randomization=True
        )
        decoded = dataset.decoded_index.copy()
        if decoded["time_idx_first_prediction"].nunique() != 1:
            raise AssertionError(f"{store}: multiple validation origins")
        if int(decoded["time_idx_first_prediction"].iloc[0]) != validation_start:
            raise AssertionError(f"{store}: validation origin mismatch")
        if not decoded["id"].is_unique:
            raise AssertionError(f"{store}: duplicate decoded ids")
        loader = dataset.to_dataloader(
            train=False,
            batch_size=cfg.teacher.batch_size,
            shuffle=False,
            num_workers=getattr(cfg.environment, "num_workers", 0),
        )
        if loss_audit is None:
            loss_audit = effective_huber_audit(
                huber, student, loader, frame, decoded, training_dataset, device, horizon
            )

        store_actuals = []
        for _, batch_y in loader:
            target = batch_y[0] if isinstance(batch_y, (tuple, list)) else batch_y
            store_actuals.append(target.cpu().numpy())
        actuals = np.concatenate(store_actuals, axis=0)
        if actuals.shape != (len(decoded), horizon):
            raise AssertionError(f"{store}: actual shape mismatch {actuals.shape}")
        all_actuals.append(actuals)
        all_ids.extend(decoded["id"].astype(str).tolist())
        for name, model in models.items():
            forecasts = get_predictions(model, loader)
            if forecasts.shape != actuals.shape:
                raise AssertionError(f"{store}/{name}: forecast shape mismatch")
            if not np.isfinite(forecasts).all():
                raise AssertionError(f"{store}/{name}: non-finite forecast")
            all_predictions[name].append(forecasts)
        del frame, dataset, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    order = np.argsort(np.asarray(all_ids, dtype=str))
    series_ids = np.asarray(all_ids, dtype=str)[order]
    actuals = np.concatenate(all_actuals, axis=0)[order]
    predictions = {
        name: np.concatenate(parts, axis=0)[order] for name, parts in all_predictions.items()
    }
    if actuals.shape != (FULL_M5_SERIES_COUNT, horizon):
        raise AssertionError(f"Expected {FULL_M5_SERIES_COUNT} series, found {actuals.shape}")
    if len(np.unique(series_ids)) != FULL_M5_SERIES_COUNT:
        raise AssertionError("Decoded validation ids are not unique")

    train_frames = []
    validation_frames = []
    for store in stores:
        frame = load_from_cache(artifacts_dir=dataset_dir, store_filter=store)
        train_frames.append(frame[frame["time_idx"] <= train_end].copy())
        validation_frames.append(
            frame[(frame["time_idx"] >= validation_start) & (frame["time_idx"] <= validation_end)].copy()
        )
        del frame
    train_frame = pd.concat(train_frames, ignore_index=True)
    gt_frame = pd.concat(validation_frames, ignore_index=True)
    del train_frames, validation_frames
    gt_frame["id"] = gt_frame["id"].astype(str)
    gt_frame = gt_frame.sort_values(["id", "time_idx"]).reset_index(drop=True)
    gt_ids = gt_frame["id"].drop_duplicates().to_numpy()
    if not np.array_equal(series_ids, gt_ids):
        raise AssertionError("Decoded id order differs from validation ground truth")
    raw_actuals = gt_frame.pivot(index="id", columns="time_idx", values="sales").reindex(series_ids).to_numpy()
    if not np.allclose(actuals, raw_actuals):
        raise AssertionError("Validation loader targets do not match raw sales")

    weights_dict, scales_dict, scale_diagnostics = compute_wrmsse_weights_and_scales(train_frame, train_end)
    mase_scales = compute_mase_scales(train_frame, train_end)
    del train_frame
    gc.collect()

    ensemble_predictions = np.mean(
        np.stack([predictions[name] for name, _ in snapshot_models], axis=0), axis=0
    )
    candidates = {
        "Huber epoch 5": predictions["Huber epoch 5"],
        "Quantile epoch 5": predictions["Quantile epoch 5"],
        "75% Huber + 25% Quantile": 0.75 * predictions["Huber epoch 5"] + 0.25 * predictions["Quantile epoch 5"],
        "50% Huber + 50% Quantile": 0.50 * predictions["Huber epoch 5"] + 0.50 * predictions["Quantile epoch 5"],
        "Equal-weight unique Huber snapshots": ensemble_predictions,
    }
    metric_details = [
        forecast_metrics(name, forecast, actuals, gt_frame, weights_dict, scales_dict, mase_scales, series_ids)
        for name, forecast in candidates.items()
    ]
    metrics = pd.DataFrame(metric_details)
    metrics.to_csv(os.path.join(output_dir, "predefined_ensemble_metrics.csv"), index=False)

    student_metrics = forecast_metrics(
        "Student supervised", predictions["Student supervised"], actuals, gt_frame,
        weights_dict, scales_dict, mase_scales, series_ids,
    )
    huber_metrics = next(item for item in metric_details if item["model"] == "Huber epoch 5")
    hierarchy_rows = []
    for index, (student_level, huber_level) in enumerate(
        zip(student_metrics["level_scores"], huber_metrics["level_scores"]), 1
    ):
        hierarchy_rows.append({
            "level": f"Level_{index}",
            "student_WRMSSE": float(student_level),
            "huber_WRMSSE": float(huber_level),
            "huber_minus_student": float(huber_level - student_level),
            "huber_relative_change_pct": float(100.0 * (huber_level / student_level - 1.0)),
            "student_contribution_to_overall": float(student_level / 12.0),
            "huber_contribution_to_overall": float(huber_level / 12.0),
        })
    hierarchy = pd.DataFrame(hierarchy_rows)
    hierarchy.to_csv(os.path.join(output_dir, "student_vs_huber_hierarchy_wrmsse.csv"), index=False)

    residual_names = list(dict.fromkeys(
        ["Huber epoch 5", "Quantile epoch 5"] + [name for name, _ in snapshot_models]
    ))
    residuals = {name: (predictions[name] - actuals).reshape(-1) for name in residual_names}
    correlations = pd.DataFrame(residuals).corr(method="pearson")
    correlations.to_csv(os.path.join(output_dir, "residual_correlations.csv"))

    quantiles = economic_weight_quantiles(
        actuals,
        {
            "Student supervised": predictions["Student supervised"],
            "Huber epoch 5": predictions["Huber epoch 5"],
        },
        weights_dict,
        scales_dict,
        series_ids,
    )
    quantiles.to_csv(os.path.join(output_dir, "student_vs_huber_economic_weight_quintiles.csv"), index=False)

    manifest["series_count"] = int(len(series_ids))
    manifest["scale_diagnostics"] = scale_diagnostics
    with open(os.path.join(output_dir, "effective_huber_loss_domain_audit.json"), "w", encoding="utf-8") as file:
        json.dump(loss_audit, file, indent=2)
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    write_report(
        os.path.join(output_dir, "report.md"), manifest, metrics, hierarchy, correlations, quantiles
    )
    print(f"Teacher-strengthening diagnostic complete: {output_dir}")


if __name__ == "__main__":
    main()

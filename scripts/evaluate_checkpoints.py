"""
evaluate_checkpoints.py — Teacher Checkpoint Validation Script

Evaluates existing TFT-64 teacher checkpoints (Quantile and Huber) against the
full validation split. Uses the authoritative compute_hierarchical_wrmsse and
compute_mase from evaluate_models.py, with identical DataFrame construction and
series-ordering logic.
"""
import os
import sys
import argparse
import hashlib
import torch
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.paths import get_dataset_dir
from data.cache import load_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from models.losses import MSELossMetric, WRMSSEInformedLossMetric
from scripts.evaluate_models import (
    compute_wrmsse_weights_and_scales,
    compute_hierarchical_wrmsse,
    compute_mase_scales,
    compute_mase,
    FULL_M5_SERIES_COUNT,
)


def get_predictions_tft(model, loader):
    """
    Generate point forecasts from a TFT using model.predict().
    Matches get_predictions() in evaluate_models.py exactly.
    Returns a numpy array [N, H].
    """
    preds = model.predict(
        loader,
        mode="prediction",
        trainer_kwargs={
            "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
            "devices": 1,
        },
    )
    return preds.cpu().numpy()


def evaluate_checkpoint(ckpt_path, ds_dir, cfg, df_train, train_end,
                        weights_dict, scales_dict, mase_scales_dict, series_ids):
    """
    Loads one checkpoint, runs inference per-store on the validation window,
    and returns all metrics. Mirrors the per-store loop in evaluate_models.py.
    """
    print(f"\n{'='*60}")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")

    if not os.path.exists(ckpt_path):
        print("  MISSING — skipping.")
        return None

    # ── Checkpoint metadata ──────────────────────────────────────────────────
    with open(ckpt_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    ckpt_size = os.path.getsize(ckpt_path)

    try:
        model = TemporalFusionTransformer.load_from_checkpoint(
            ckpt_path, map_location="cpu"
        )
    except Exception as e:
        print(f"  Failed to load checkpoint: {e}")
        return None

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    is_quantile = hasattr(model.loss, "quantiles")
    if is_quantile:
        objective = "Quantile"
    elif isinstance(model.loss, MSELossMetric):
        objective = "MSE"
    elif isinstance(model.loss, WRMSSEInformedLossMetric):
        objective = "WRMSSE-informed"
    else:
        objective = "Huber"
    internal_epoch = model.current_epoch
    global_step = model.global_step
    hidden_size = getattr(model.hparams, "hidden_size", None)

    # ── Build training dataset for TimeSeriesDataSet ─────────────────────────
    # We load one small slice per store to build a combined training dataset.
    # This follows the same logic as evaluate_models.py which loads df and
    # passes it to build_timeseries_dataset once.
    store_filter = getattr(cfg.environment, "store_filter", None)
    stores = resolve_stores(store_filter)

    val_end = cfg.dataset.splits.validation.end
    L = cfg.dataset.lookback_window
    H = cfg.dataset.prediction_window
    
    # In predict=True mode, TimeSeriesDataSet only emits the LAST H-day window
    # in the dataframe. Thus, the actual validation window being evaluated is:
    val_start = val_end - H + 1

    # We need training_dataset for TimeSeriesDataSet.from_dataset().
    # Load train data from cache to build it.
    print("  Building training dataset schema...")
    train_dfs = []
    for store in stores:
        df_s = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_s is not None:
            train_dfs.append(df_s[df_s["time_idx"] <= train_end])
    df_train_full = pd.concat(train_dfs, ignore_index=True)
    del train_dfs
    for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                "weekday", "month", "year", "event_name_1", "event_type_1"]:
        if col in df_train_full.columns:
            df_train_full[col] = df_train_full[col].astype(str).astype("category")

    training_dataset = build_timeseries_dataset(df_train_full, cfg, is_train=True)
    del df_train_full

    # ── Per-store validation inference ───────────────────────────────────────
    # Exactly mirrors evaluate_models.py: load each store's slice separately,
    # build a per-store dataset, run inference, collect predictions + decoded_index.
    print("  Running per-store validation inference...")
    min_idx = val_start - L
    all_actuals = []
    all_preds = []
    all_decoded = []

    for store in stores:
        df_part = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_part is None:
            continue

        df_part_sliced = df_part[
            (df_part["time_idx"] >= min_idx) & (df_part["time_idx"] <= val_end)
        ].copy()
        del df_part

        for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                    "weekday", "month", "year", "event_name_1", "event_type_1"]:
            if col in df_part_sliced.columns:
                df_part_sliced[col] = df_part_sliced[col].astype(str).astype("category")

        if len(df_part_sliced) == 0:
            continue

        part_ds = TimeSeriesDataSet.from_dataset(
            training_dataset, df_part_sliced, predict=True, stop_randomization=True
        )

        # Alignment guard — mirrors evaluate_models.py lines 1094-1096
        _decoded = part_ds.decoded_index
        assert _decoded["time_idx_first_prediction"].nunique() == 1, (
            "Multiple prediction starts in one store partition"
        )
        assert int(_decoded["time_idx_first_prediction"].iloc[0]) == val_start, (
            f"Prediction start mismatch: expected {val_start}, "
            f"got {_decoded['time_idx_first_prediction'].iloc[0]}"
        )
        assert _decoded["id"].is_unique, "Duplicate series IDs in store partition"

        part_loader = part_ds.to_dataloader(
            train=False,
            batch_size=cfg.teacher.batch_size,
            shuffle=False,
            num_workers=getattr(cfg.environment, "num_workers", 0),
        )

        # Collect actuals from batches
        store_actuals = []
        for batch_x, batch_y in part_loader:
            target = batch_y[0] if isinstance(batch_y, (tuple, list)) else batch_y
            store_actuals.append(target.cpu().numpy())
        actuals_np = np.concatenate(store_actuals, axis=0)  # [n_series, H]

        # Run inference
        store_preds = get_predictions_tft(model, part_loader)  # [n_series, H]

        assert store_preds.shape == actuals_np.shape, (
            f"Shape mismatch: preds={store_preds.shape}, actuals={actuals_np.shape}"
        )

        all_actuals.append(actuals_np)
        all_preds.append(store_preds)
        all_decoded.append(_decoded)
        del df_part_sliced, part_ds, part_loader

    if not all_actuals:
        print("  No data collected — skipping.")
        return None

    # ── Sort by id — identical to evaluate_models.py lines 1136-1147 ─────────
    concatenated_decoded = pd.concat(all_decoded, ignore_index=True)
    concatenated_decoded["id"] = concatenated_decoded["id"].astype(str)
    order = (
        concatenated_decoded
        .assign(row_idx=np.arange(len(concatenated_decoded)))
        .sort_values("id")["row_idx"]
        .to_numpy()
    )
    actuals = np.concatenate(all_actuals, axis=0)[order]        # [N, H]
    forecasts = np.concatenate(all_preds, axis=0)[order]        # [N, H]
    sorted_ids = concatenated_decoded.sort_values("id")["id"].to_numpy()

    print(f"  Series count: {actuals.shape[0]} (expected {FULL_M5_SERIES_COUNT})")
    assert actuals.shape[0] == FULL_M5_SERIES_COUNT, (
        f"Expected {FULL_M5_SERIES_COUNT} series, got {actuals.shape[0]}"
    )
    assert np.isfinite(actuals).all(), "Non-finite actuals"
    assert np.isfinite(forecasts).all(), "Non-finite forecasts"

    # ── Build gt/pred DataFrames — mirrors evaluate_models.py lines 1180-1182 ─
    # Load the actual validation ground-truth for hierarchical grouping columns.
    print("  Loading validation ground-truth for hierarchy columns...")
    val_gt_dfs = []
    for store in stores:
        df_s = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_s is not None:
            val_gt_dfs.append(
                df_s[(df_s["time_idx"] >= val_start) & (df_s["time_idx"] <= val_end)]
            )
    df_val_gt = pd.concat(val_gt_dfs, ignore_index=True)
    del val_gt_dfs
    df_val_gt["id"] = df_val_gt["id"].astype(str)
    df_val_gt = df_val_gt.sort_values(["id", "time_idx"]).reset_index(drop=True)

    # Build predictions DataFrame: copy gt (for hierarchy columns), replace sales
    df_preds_df = df_val_gt.copy()
    df_preds_df["sales"] = forecasts.flatten()

    # ── Compute authoritative WRMSSE ─────────────────────────────────────────
    wrmsse, level_wrmsses = compute_hierarchical_wrmsse(
        df_val_gt, df_preds_df, weights_dict, scales_dict
    )

    # ── Point metrics ────────────────────────────────────────────────────────
    actuals_flat = actuals.flatten()
    preds_flat = forecasts.flatten()
    mae  = float(np.mean(np.abs(actuals_flat - preds_flat)))
    rmse = float(np.sqrt(np.mean((actuals_flat - preds_flat) ** 2)))
    total_actual = float(np.sum(actuals_flat))
    total_pred   = float(np.sum(preds_flat))
    wape = float(np.sum(np.abs(actuals_flat - preds_flat)) / (total_actual + 1e-9))
    agg_bias = float((total_pred / (total_actual + 1e-9)) - 1.0)

    # ── Seasonal MASE — uses compute_mase() exactly as in evaluate_models.py ─
    scales_array = np.array([
        mase_scales_dict.get(sid, 1.0) for sid in sorted_ids
    ])
    mase = float(compute_mase(actuals, forecasts, scales_array))

    # ── Cleanup ──────────────────────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()

    result = {
        "checkpoint": os.path.basename(ckpt_path),
        "internal_epoch": internal_epoch,
        "global_step": global_step,
        "hidden_size": hidden_size,
        "objective": objective,
        "parameter_count": total_params,
        "checkpoint_size_bytes": ckpt_size,
        "checkpoint_SHA256": sha256,
        "validation_WRMSSE": wrmsse,
        "validation_MAE": mae,
        "validation_RMSE": rmse,
        "validation_WAPE": wape,
        "validation_seasonal_MASE": mase,
        "aggregate_percentage_bias": agg_bias,
        "actual_total": total_actual,
        "predicted_total": total_pred,
    }
    result.update({
        f"validation_WRMSSE_Level_{level}": float(score)
        for level, score in enumerate(level_wrmsses, start=1)
    })
    return result


def prepare_validation_context(cfg):
    """Build the shared context used by the authoritative checkpoint evaluator."""
    ds_dir = get_dataset_dir(cfg)
    stores = resolve_stores(getattr(cfg.environment, "store_filter", None))
    train_end = cfg.dataset.splits.train.end

    print("\nLoading training data for scale computation...")
    train_dfs = []
    for store in stores:
        df_s = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_s is not None:
            train_dfs.append(df_s[df_s["time_idx"] <= train_end])
    df_train = pd.concat(train_dfs, ignore_index=True)
    del train_dfs
    print(f"  Train rows: {len(df_train):,}")

    weights_dict, scales_dict, _ = compute_wrmsse_weights_and_scales(df_train, train_end)
    mase_scales_dict = compute_mase_scales(df_train, train_end)
    series_ids = df_train["id"].astype(str).drop_duplicates().sort_values().to_numpy()
    del df_train
    return ds_dir, train_end, weights_dict, scales_dict, mase_scales_dict, series_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="local")
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default=None,
        help="Evaluate every checkpoint matching this glob with the common validation evaluator.",
    )
    args = parser.parse_args()

    cfg = load_config(args.env)

    checkpoints = [
        ("Teacher-v1: Original Quantile",
         "outputs/teacher/exp_full_phase1/best_tft_teacher.ckpt"),
        ("Teacher-v2: Optimized Quantile (Epoch 3)",
         "outputs/teacher/tft64_optimized/tft64-opt-epoch=epoch=03-val_loss=val_loss=0.473921.ckpt"),
        ("Teacher-v2: Optimized Quantile (Epoch 5)",
         "outputs/teacher/tft64_optimized/tft64-opt-epoch=epoch=05-val_loss=val_loss=0.474301.ckpt"),
        ("Teacher-v2: Optimized Quantile (Epoch 8)",
         "outputs/teacher/tft64_optimized/tft64-opt-epoch=epoch=08-val_loss=val_loss=0.474549.ckpt"),
        ("Teacher-v3: Huber (Epoch 5 - best val_loss)",
         "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=05-val_loss=val_loss=0.606112.ckpt"),
        ("Teacher-v3: Huber (Epoch 13 - best MAE)",
         "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=13-val_loss=val_loss=0.606753.ckpt"),
        ("Teacher-v3: Huber (Epoch 8)",
         "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=08-val_loss=val_loss=0.612628.ckpt"),
    ]
    if args.checkpoint_glob:
        import glob
        matched = sorted(glob.glob(args.checkpoint_glob))
        if not matched:
            raise FileNotFoundError(f"No checkpoints matched: {args.checkpoint_glob}")
        checkpoints = [(os.path.basename(path), path) for path in matched]

    print("\nCheckpoint existence check:")
    for label, path in checkpoints:
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {label}")

    valid_checkpoints = [(label, path) for label, path in checkpoints if os.path.exists(path)]
    if not valid_checkpoints:
        print("No valid checkpoints found — aborting.")
        return

    # ── Pre-load shared data: train split for WRMSSE/MASE scales ─────────────
    ds_dir, train_end, weights_dict, scales_dict, mase_scales_dict, series_ids = (
        prepare_validation_context(cfg)
    )

    # ── Evaluate each checkpoint ──────────────────────────────────────────────
    results = []
    for label, ckpt_path in valid_checkpoints:
        result = evaluate_checkpoint(
            ckpt_path, ds_dir, cfg, None, train_end,
            weights_dict, scales_dict, mase_scales_dict, series_ids,
        )
        if result is not None:
            result["teacher_version"] = label
            results.append(result)

    if not results:
        print("No results produced — exiting.")
        return

    # ── Output ────────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    columns = [
        "teacher_version", "checkpoint", "internal_epoch", "global_step",
        "hidden_size", "objective", "parameter_count", "checkpoint_size_bytes",
        "validation_WRMSSE", "validation_MAE", "validation_RMSE",
        "validation_WAPE", "validation_seasonal_MASE", "aggregate_percentage_bias",
        "actual_total", "predicted_total", "checkpoint_SHA256",
    ]
    columns.extend(f"validation_WRMSSE_Level_{level}" for level in range(1, 13))
    df_results = df_results[[c for c in columns if c in df_results.columns]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = f"artifacts/teacher_checkpoint_validation_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    df_results.to_csv(os.path.join(out_dir, "checkpoint_metrics.csv"), index=False)

    with open(os.path.join(out_dir, "checkpoint_validation_report.md"), "w") as f:
        f.write("# Teacher Checkpoint Validation Report\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        # Manually create markdown table to avoid 'tabulate' dependency
        cols = df_results.columns.tolist()
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, row in df_results.iterrows():
            f.write("| " + " | ".join(str(x) for x in row.values) + " |\n")

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_dir}")
    print(df_results[["teacher_version", "validation_WRMSSE", "validation_MAE",
                       "validation_RMSE", "validation_seasonal_MASE"]].to_string(index=False))


if __name__ == "__main__":
    main()

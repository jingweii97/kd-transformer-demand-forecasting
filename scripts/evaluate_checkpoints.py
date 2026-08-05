import os
import sys
import argparse
import hashlib
import torch
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.paths import get_dataset_dir
from data.cache import load_dataset_from_cache
from data.dataset import build_timeseries_dataset
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from scripts.evaluate_models import (
    compute_wrmsse_weights_and_scales,
    compute_hierarchical_wrmsse,
    compute_mase_scales,
    compute_mase,
    HIERARCHY_LEVELS,
    FULL_M5_SERIES_COUNT,
)


def get_checkpoint_meta(ckpt_path):
    """Load checkpoint and extract identity metadata without running inference."""
    print(f"\nLoading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"  MISSING — skipping.")
        return None

    with open(ckpt_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    ckpt_size = os.path.getsize(ckpt_path)

    try:
        model = TemporalFusionTransformer.load_from_checkpoint(
            ckpt_path, map_location="cpu"
        )
    except Exception as e:
        print(f"  Failed to load: {e}")
        return None

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    is_quantile = hasattr(model.loss, "quantiles")
    objective = "Quantile" if is_quantile else "Huber"

    if is_quantile:
        quantiles = list(model.loss.quantiles)
        assert 0.5 in quantiles, f"0.5 not in quantiles: {quantiles}"
        q50_idx = quantiles.index(0.5)
    else:
        quantiles = None
        q50_idx = 0

    return {
        "checkpoint": os.path.basename(ckpt_path),
        "path": ckpt_path,
        "internal_epoch": model.current_epoch,
        "global_step": model.global_step,
        "objective": objective,
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "checkpoint_size": ckpt_size,
        "checkpoint_SHA256": sha256,
        "quantiles": quantiles,
        "q50_idx": q50_idx,
        "is_quantile": is_quantile,
        "model": model,
    }


def run_inference(model, val_dl, meta, device):
    """
    Runs authoritative inference using model.predict() with trainer_kwargs,
    exactly matching the get_predictions() pattern in evaluate_models.py.
    Returns (preds_q50_numpy [N, H], quantile_loss_or_None).
    """
    trainer_kwargs = {
        "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
        "devices": 1,
    }

    if meta["is_quantile"]:
        # Return full quantile tensor for loss calculation
        preds_full = model.predict(
            val_dl,
            mode="quantiles",
            return_y=True,
            trainer_kwargs=trainer_kwargs,
        )
        # preds_full is a tensor [N, H, Q]
        preds_q50 = preds_full[:, :, meta["q50_idx"]].cpu().numpy()

        # Quantile loss: use model.predict with mode="raw" to get targets, or
        # skip and set None — the key metrics are WRMSSE/MAE/RMSE, not raw Q-loss.
        # We leave quantile_loss as None here to avoid triggering predictions.y bugs;
        # the correct loss is already logged per-epoch in metrics.csv.
        quantile_loss = None
    else:
        preds_full = model.predict(
            val_dl,
            mode="prediction",
            trainer_kwargs=trainer_kwargs,
        )
        # preds_full is [N, H] or [N, H, 1]
        preds_q50 = preds_full.cpu().numpy()
        if preds_q50.ndim == 3 and preds_q50.shape[-1] == 1:
            preds_q50 = preds_q50.squeeze(-1)
        quantile_loss = None

    return preds_q50, quantile_loss


def build_prediction_df(preds_q50, val_ds):
    """
    Reconstructs a long-format DataFrame of (id, time_idx, pred_q50)
    from the stacked prediction array, using val_ds.decoded_index for alignment.
    Uses the same alignment approach as the authoritative evaluator.
    """
    decoded = val_ds.decoded_index
    N, H = preds_q50.shape
    assert len(decoded) == N, (
        f"decoded_index length {len(decoded)} != prediction rows {N}"
    )

    rows = []
    for b in range(N):
        sid = str(decoded.iloc[b]["id"])
        # encoder_length is the number of encoder steps; the first decoder step
        # starts at encoder_length time steps after the sample's start.
        # time_idx at decoder step h is the time_idx of the first decoder position + h.
        # val_ds.decoded_index has the 'time_idx' of the LAST encoder step.
        last_encoder_tidx = int(decoded.iloc[b]["time_idx"])
        for h in range(H):
            t_idx = last_encoder_tidx + 1 + h
            rows.append({
                "id": sid,
                "time_idx": t_idx,
                "pred_q50": float(preds_q50[b, h]),
            })

    return pd.DataFrame(rows)


def evaluate_predictions(df_preds, df_val, df_train, weights_dict, scales_dict,
                         mase_scales_dict, series_ids):
    """
    Computes all required validation metrics using the authoritative
    compute_hierarchical_wrmsse from evaluate_models.py.
    """
    print("  Running authoritative metric computation...")

    # Merge predictions with validation actuals
    df_val_str = df_val.copy()
    df_val_str["id"] = df_val_str["id"].astype(str)

    df_merged = df_val_str.merge(df_preds, on=["id", "time_idx"], how="inner")
    print(f"  Merged rows: {len(df_merged)} (predictions: {len(df_preds)})")

    if len(df_merged) == 0:
        print("  ERROR: No rows matched after merge — check id/time_idx alignment.")
        return None

    # Build gt and preds DataFrames in the format expected by compute_hierarchical_wrmsse
    # (which expects 'sales' column for both)
    df_gt = df_merged[["id", "time_idx", "state_id", "store_id", "cat_id",
                         "dept_id", "item_id", "sales"]].copy()
    df_pred = df_merged[["id", "time_idx", "state_id", "store_id", "cat_id",
                           "dept_id", "item_id"]].copy()
    df_pred["sales"] = df_merged["pred_q50"].values

    # WRMSSE (authoritative)
    wrmsse, level_wrmsses = compute_hierarchical_wrmsse(df_gt, df_pred, weights_dict, scales_dict)

    # Point metrics
    actuals = df_merged["sales"].values
    preds   = df_merged["pred_q50"].values

    mae  = float(np.mean(np.abs(actuals - preds)))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))

    total_actual = float(np.sum(actuals))
    total_pred   = float(np.sum(preds))
    wape = float(np.sum(np.abs(actuals - preds)) / (total_actual + 1e-9))
    agg_bias = float((total_pred / (total_actual + 1e-9)) - 1.0)

    # Seasonal MASE — series-level, aligned to series_ids order
    # Reshape merged into [num_series, horizon] arrays
    df_grouped_act  = df_merged.groupby("id")["sales"].apply(np.array)
    df_grouped_pred = df_merged.groupby("id")["pred_q50"].apply(np.array)

    # Only compute MASE for series that have matching scale
    mase_vals = []
    for sid in df_grouped_act.index:
        sid_str = str(sid)
        if sid_str not in mase_scales_dict:
            continue
        scale = mase_scales_dict[sid_str]
        if scale <= 0:
            continue
        a = df_grouped_act[sid]
        p = df_grouped_pred[sid]
        if len(a) != len(p) or len(a) == 0:
            continue
        mase_vals.append(float(np.mean(np.abs(a - p)) / scale))

    mase = float(np.mean(mase_vals)) if mase_vals else float("nan")

    return {
        "validation_WRMSSE": wrmsse,
        "validation_MAE": mae,
        "validation_RMSE": rmse,
        "validation_WAPE": wape,
        "validation_seasonal_MASE": mase,
        "aggregate_percentage_bias": agg_bias,
        "actual_total": total_actual,
        "predicted_total": total_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="local")
    args = parser.parse_args()

    cfg = load_config(args.env)

    checkpoints = [
        "outputs/teacher/exp_full_phase1/best_tft_teacher.ckpt",
        "outputs/teacher/tft64_optimized/tft64-opt-epoch=epoch=03-val_loss=val_loss=0.473921.ckpt",
        "outputs/teacher/tft64_optimized/tft64-opt-epoch=epoch=05-val_loss=val_loss=0.474301.ckpt",
        "outputs/teacher/tft64_optimized/tft64-opt-epoch=epoch=08-val_loss=val_loss=0.474549.ckpt",
        "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=05-val_loss=val_loss=0.606112.ckpt",
        "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=13-val_loss=val_loss=0.606753.ckpt",
        "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=08-val_loss=val_loss=0.612628.ckpt",
    ]

    valid_checkpoints = [c for c in checkpoints if os.path.exists(c)]
    missing = [c for c in checkpoints if not os.path.exists(c)]
    if missing:
        for m in missing:
            print(f"MISSING checkpoint (will skip): {m}")
    if not valid_checkpoints:
        print("No valid checkpoints found — aborting.")
        return

    # ── Data loading — identical pattern to evaluate_models.py ──
    print("\nLoading dataset...")
    ds_dir       = get_dataset_dir(cfg)
    store_filter = getattr(cfg.environment, "store_filter", None)
    df           = load_dataset_from_cache(artifacts_dir=ds_dir, store_filter=store_filter)

    train_end = cfg.dataset.splits.train.end
    val_end   = cfg.dataset.splits.validation.end
    df_train  = df[df["time_idx"] <= train_end].copy()
    df_val    = df[(df["time_idx"] > train_end) & (df["time_idx"] <= val_end)].copy()

    print(f"  Train rows: {len(df_train):,} | Val rows: {len(df_val):,}")

    # ── Pre-compute WRMSSE scales/weights and MASE scales ──
    weights_dict, scales_dict, scale_diag = compute_wrmsse_weights_and_scales(df_train, train_end)
    mase_scales_dict = compute_mase_scales(df_train, train_end)
    series_ids = df["id"].astype(str).drop_duplicates().sort_values().to_numpy()

    # ── Build TimeSeriesDataSet for validation ──
    print("\nBuilding validation dataset...")
    training_dataset = build_timeseries_dataset(df, cfg, is_train=True)
    val_ds = TimeSeriesDataSet.from_dataset(
        training_dataset, df_val, predict=True, stop_randomization=True
    )
    val_dl = val_ds.to_dataloader(
        train=False,
        batch_size=cfg.teacher.batch_size,
        num_workers=getattr(cfg.environment, "num_workers", 0),
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for ckpt_path in valid_checkpoints:
        meta = get_checkpoint_meta(ckpt_path)
        if meta is None:
            continue

        model = meta.pop("model").to(device)

        print(f"  Running inference for: {meta['checkpoint']}")
        preds_q50, quantile_loss = run_inference(model, val_dl, meta, device)

        print(f"  Prediction shape: {preds_q50.shape}")

        df_preds = build_prediction_df(preds_q50, val_ds)

        metrics = evaluate_predictions(
            df_preds, df_val, df_train,
            weights_dict, scales_dict,
            mase_scales_dict, series_ids,
        )
        if metrics is None:
            print(f"  Evaluation failed for {meta['checkpoint']} — skipping.")
            continue

        meta["validation_quantile_loss"] = quantile_loss
        meta.update(metrics)

        if "exp_full" in ckpt_path:
            meta["teacher_version"] = "Teacher-v1: Standard Quantile TFT"
        elif "tft64_optimized" in ckpt_path:
            meta["teacher_version"] = "Teacher-v2: Optimized Quantile TFT"
        elif "tft64_huber" in ckpt_path:
            meta["teacher_version"] = "Teacher-v3: Huber TFT"
        else:
            meta["teacher_version"] = "Teacher"

        results.append(meta)
        print(f"  Done. WRMSSE={metrics['validation_WRMSSE']:.5f}  MAE={metrics['validation_MAE']:.5f}")

        # Free model memory before loading the next checkpoint
        del model
        torch.cuda.empty_cache()

    if not results:
        print("No results produced — exiting.")
        return

    df_results = pd.DataFrame(results)

    columns = [
        "teacher_version", "checkpoint", "internal_epoch", "global_step",
        "objective", "parameter_count", "trainable_parameter_count",
        "checkpoint_size", "validation_quantile_loss",
        "validation_MAE", "validation_RMSE", "validation_seasonal_MASE",
        "validation_WAPE", "validation_WRMSSE", "aggregate_percentage_bias",
        "actual_total", "predicted_total", "checkpoint_SHA256",
    ]
    df_results = df_results[[c for c in columns if c in df_results.columns]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = f"artifacts/teacher_checkpoint_validation_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "checkpoint_metrics.csv")
    md_path  = os.path.join(out_dir, "checkpoint_validation_report.md")

    df_results.to_csv(csv_path, index=False)

    with open(md_path, "w") as f:
        f.write("# Checkpoint Validation Report\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(df_results.to_markdown(index=False))

    print(f"\nResults saved to: {out_dir}")
    print(df_results[["teacher_version", "checkpoint", "validation_WRMSSE",
                       "validation_MAE", "validation_RMSE"]].to_string(index=False))


if __name__ == "__main__":
    main()

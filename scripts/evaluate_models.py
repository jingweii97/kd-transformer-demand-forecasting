"""
evaluate_models.py — M5 Model Evaluation Script

Methodology (P2):
  - Fixed Windows: ID Reference, Event-Intensive OOD, Extended-Gap OOD
  - Overall Test Stream: 52 seven-day-aligned origins
  - Strict checkpointing with SHA-256
  - Robust WRMSSE (zero-sales stripping)
  - 90-day benchmark inference latency
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import hashlib
import glob
import time
import json
import gc
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, save_config, save_metadata
from utils.paths import resolve_path
from utils.seed import set_seed
from data.cache import load_from_cache, load_dataset_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset
from models.student import M5TransformerStudent
from utils.wrmsse import (
    compute_rmsse_scale,
    economic_weight_numerators,
    normalize_economic_weight,
)

HIERARCHY_LEVELS = [
    [], ['state_id'], ['store_id'], ['cat_id'], ['dept_id'],
    ['state_id', 'cat_id'], ['state_id', 'dept_id'], ['store_id', 'cat_id'],
    ['store_id', 'dept_id'], ['item_id'], ['item_id', 'state_id'], ['id']
]

FULL_M5_SERIES_COUNT = 30490

def resolve_model_checkpoint(cli_path, config_path, outputs_dir, rel_subpath):

    if cli_path:

        resolved = resolve_path(cli_path)

        if os.path.exists(resolved):

            return resolved

        raise FileNotFoundError(f"CLI-specified checkpoint not found: '{resolved}'")

    if config_path:

        resolved = resolve_path(config_path)

        if os.path.exists(resolved):

            return resolved

        raise FileNotFoundError(f"Config-specified checkpoint not found: '{resolved}'")



    # Fallback to deterministic expected path in outputs

    fallback = os.path.abspath(os.path.join(outputs_dir, rel_subpath))

    if os.path.exists(fallback):

        return fallback



    raise FileNotFoundError(

        f"Final evaluation requires explicit checkpoints. '{fallback}' not found "

        "and no valid CLI or config path provided."

    )

def _compute_scale(series: np.ndarray):

    """

    M5-aligned in-sample naive (first-difference) scale for one hierarchy aggregate.



    The caller must reindex the series to a contiguous time_idx range (gap days

    filled with 0) before calling, so that missing calendar days are treated as

    zero-sale observations consistent with M5 convention.



    Steps:

      1. Strip leading observations before the first non-zero sale.

      2. Compute mean squared first difference on remaining observations.

      3. Return FALLBACK (1e-4) if fewer than 2 observations remain or result is 0.



    Returns

    -------

    scale  : float

    reason : str ΓÇö one of 'valid', 'insufficient_length', 'all_zero', 'zero_variance'

    """

    return compute_rmsse_scale(series)

def compute_wrmsse_weights_and_scales(df_train, train_end):

    """

    Pre-computes M5 hierarchy scales (naive std) and dollar-value weights.



    Applies M5-aligned corrections:

      - Reindexes each aggregate series to a full contiguous time_idx range

        (zero-fills calendar gaps) before computing the scale.

      - Strips leading zero observations per series (before first non-zero sale).



    Returns

    -------

    weights_dict     : dict  Level_k ΓåÆ {group_key: weight}

    scales_dict      : dict  Level_k ΓåÆ {group_key: scale}

    scale_diagnostics: dict  audit counts and affected WRMSSE weight

    """

    print("Pre-computing WRMSSE scale factors and value weights...")

    df_train = df_train.copy()

    df_train['dollar_value'] = df_train['sales'] * df_train['sell_price']



    # Value weights: last 28 days of training

    df_weight_window = df_train[df_train['time_idx'] > (train_end - 28)].copy()

    total_dollar_sum = df_weight_window['dollar_value'].sum()



    weights_dict = {}

    scales_dict  = {}



    scale_diagnostics = {

        "valid": 0,

        "insufficient_length": 0,

        "all_zero": 0,

        "zero_variance": 0,

        "total_wrmsse_weight_on_fallback": 0.0,

        "fallback_series": [],

    }



    # Full contiguous time_idx range for reindexing (calendar gap fill)

    full_time_range = np.arange(int(df_train['time_idx'].min()), train_end + 1)



    for level_idx, group_cols in enumerate(HIERARCHY_LEVELS, 1):

        level_name = f"Level_{level_idx}"

        weights_dict[level_name] = {}

        scales_dict[level_name]  = {}

        scale_reasons = {}  # key_str ΓåÆ reason string



        if len(group_cols) == 0:

            # Level 1: aggregate all series

            agg = (df_train.groupby('time_idx')['sales']

                           .sum()

                           .reindex(full_time_range, fill_value=0)

                           .values)

            scale, reason = _compute_scale(agg)

            scales_dict[level_name]['Total']  = scale

            weights_dict[level_name]['Total'] = 1.0

            scale_reasons['Total'] = reason

            scale_diagnostics[reason] += 1

            if reason != "valid":

                scale_diagnostics["total_wrmsse_weight_on_fallback"] += 1.0

                scale_diagnostics["fallback_series"].append(f"{level_name}/Total")

        else:

            # Grouped levels

            df_grouped_train  = (df_train.groupby(group_cols + ['time_idx'])['sales']

                                         .sum().reset_index())

            df_grouped_weight = economic_weight_numerators(df_weight_window, group_cols)



            # Compute scales

            for keys, group in df_grouped_train.groupby(group_cols):

                if isinstance(keys, tuple):

                    key_str = "_".join(str(k) for k in keys)

                else:

                    key_str = str(keys)

                agg = (group.set_index('time_idx')['sales']

                            .reindex(full_time_range, fill_value=0)

                            .values)

                scale, reason = _compute_scale(agg)

                scales_dict[level_name][key_str] = scale

                scale_reasons[key_str] = reason

                scale_diagnostics[reason] += 1



            # Compute weights; audit fallback impact

            for _, row in df_grouped_weight.iterrows():

                keys_vals = row[group_cols].values

                if len(group_cols) > 1:

                    key_str = "_".join(str(k) for k in keys_vals)

                else:

                    key_str = str(keys_vals[0])

                w = normalize_economic_weight(row['dollar_value'], total_dollar_sum)

                weights_dict[level_name][key_str] = w

                if scale_reasons.get(key_str, "valid") != "valid":

                    scale_diagnostics["total_wrmsse_weight_on_fallback"] += w

                    scale_diagnostics["fallback_series"].append(f"{level_name}/{key_str}")



        # Validate weights sum to 1.0 (with a small tolerance)

        level_weight_sum = sum(weights_dict[level_name].values())

        assert np.isclose(level_weight_sum, 1.0, atol=1e-6), \
            f"{level_name} weights sum to {level_weight_sum}"



    n_fallback = (scale_diagnostics["insufficient_length"]

                  + scale_diagnostics["all_zero"]

                  + scale_diagnostics["zero_variance"])

    print(f"  Scale diagnostics: valid={scale_diagnostics['valid']}, "

          f"fallback={n_fallback}, "

          f"total_weight_on_fallback={scale_diagnostics['total_wrmsse_weight_on_fallback']:.4f}")

    if scale_diagnostics["total_wrmsse_weight_on_fallback"] > 0.05:

        print("  WARNING: >5% of WRMSSE weight is on fallback scales ΓÇö "

              "inspect wrmsse_scale_diagnostics.json before reporting results.")



    return weights_dict, scales_dict, scale_diagnostics

def compute_hierarchical_wrmsse(df_test_gt, df_test_preds, weights_dict, scales_dict):

    """

    Computes M5 WRMSSE across all hierarchy levels.

    """

    level_wrmsses = []

    

    for level_idx, group_cols in enumerate(HIERARCHY_LEVELS, 1):

        level_name = f"Level_{level_idx}"

        level_weights = weights_dict[level_name]

        level_scales = scales_dict[level_name]

        

        rmsses = []

        weights = []

        

        if len(group_cols) == 0:

            # Level 1

            gt_agg = df_test_gt.groupby('time_idx')['sales'].sum().sort_index().values

            pred_agg = df_test_preds.groupby('time_idx')['sales'].sum().sort_index().values

            

            mse = np.mean((gt_agg - pred_agg) ** 2)

            scale = level_scales['Total']

            rmsses.append(np.sqrt(mse / scale))

            weights.append(1.0)

        else:

            # Group actuals and predictions

            df_gt_grouped = df_test_gt.groupby(group_cols + ['time_idx'])['sales'].sum().reset_index()

            df_pred_grouped = df_test_preds.groupby(group_cols + ['time_idx'])['sales'].sum().reset_index()

            

            # Merge to align keys

            df_merged = df_gt_grouped.merge(df_pred_grouped, on=group_cols + ['time_idx'], suffixes=('_gt', '_pred'))

            

            for keys, group in df_merged.groupby(group_cols):

                key_str = "_".join(keys) if isinstance(keys, tuple) else str(keys)

                

                gt_vals = group.sort_values(by='time_idx')['sales_gt'].values

                pred_vals = group.sort_values(by='time_idx')['sales_pred'].values

                

                mse = np.mean((gt_vals - pred_vals) ** 2)

                assert key_str in level_scales, f"Missing WRMSSE scale for {level_name}/{key_str}"
                assert key_str in level_weights, f"Missing WRMSSE weight for {level_name}/{key_str}"
                scale = level_scales[key_str]
                w = level_weights[key_str]

                

                rmsses.append(np.sqrt(mse / scale))

                weights.append(w)

                

        level_wrmsse = np.sum(np.array(rmsses) * np.array(weights))

        level_wrmsses.append(level_wrmsse)

        

    overall_wrmsse = np.mean(level_wrmsses)

    return overall_wrmsse, level_wrmsses

def compute_point_metrics(actuals, forecasts):

    """

    Computes standard point forecast accuracy metrics.

    """

    mae = np.mean(np.abs(actuals - forecasts))

    rmse = np.sqrt(np.mean((actuals - forecasts) ** 2))

    

    total_abs_error = np.sum(np.abs(actuals - forecasts))

    total_sales = np.sum(actuals)

    wape = total_abs_error / total_sales if total_sales > 0 else 0.0

    

    return mae, rmse, wape

def compute_mase_scales(df_train, train_end):

    """

    Precomputes the seasonal naive MAE denominator (in-sample absolute difference scale)

    for each series.

    """

    print("Pre-computing scale factors for the MASE calculation...")

    # Group by id and time_idx to get sales per series per day, ensuring correct order

    df_sorted = df_train.sort_values(by=['id', 'time_idx']).reset_index(drop=True)

    

    # Calculate absolute differences lagged by 28 days per series

    # Using pandas groupby shift to avoid boundary leakage between different ids

    sales = df_sorted['sales'].values

    prev_sales = df_sorted.groupby('id')['sales'].shift(28).values

    

    df_sorted['abs_diff'] = np.abs(sales - prev_sales)

    

    # Mean absolute difference for each series (ignoring NaNs from first 28 days)

    scales = df_sorted.groupby('id', observed=True)['abs_diff'].mean()

    

    # Fill zero or NaN scales to avoid division by zero

    scales = scales.fillna(1.0).replace(0.0, 1.0)

    return scales.to_dict()

def compute_mase(actuals_slice, forecasts_slice, scales_array):

    """

    Computes MASE for each series and returns the average MASE.

    actuals_slice shape: (num_series, slice_len)

    forecasts_slice shape: (num_series, slice_len)

    scales_array shape: (num_series,)

    """

    mae_per_series = np.mean(np.abs(actuals_slice - forecasts_slice), axis=1)

    mase_per_series = mae_per_series / scales_array

    return np.mean(mase_per_series)



def get_predictions(model, loader):

    """

    Generates point forecasts from PyTorch Forecasting (TFT) or custom Lightning Student Module.

    """

    # For TFT Teacher, use PyTorch Forecasting's built-in predict method

    if isinstance(model, TemporalFusionTransformer):

        preds = model.predict(

            loader,

            mode="prediction",

            trainer_kwargs={

                "accelerator": "cuda" if torch.cuda.is_available() else "cpu",

                "devices": 1

            }

        )

        return preds.cpu().numpy()

        

    # For custom Student models, run standard batch evaluation

    model.eval()

    all_preds = []

    with torch.no_grad():

        for batch in loader:

            x, _ = batch

            if hasattr(model, "device"):

                for k in x.keys():

                    if isinstance(x[k], torch.Tensor):

                        x[k] = x[k].to(model.device)

            preds = model(x)

            all_preds.append(preds.cpu())

    return torch.cat(all_preds, dim=0).numpy()

def forward_tft(model, x):
    output = model(x)
    return model.to_prediction(output)

def forward_student(model, x):
    return model(x)

def benchmark_inference(models_info, sample_batch, device,

                        num_warmup=3, num_runs=10):

    """

    Controlled inference benchmark with identical timing boundaries for all models.



    Uses forward adapters (forward_tft / forward_student) so that TFT is measured

    through a simple forward pass (no Lightning Trainer), enabling fair latency

    comparison across models.



    Parameters

    ----------

    models_info  : list of (model_name, model_obj, is_tft)

    sample_batch : dict (batch_x) ΓÇö fixed batch used for all timed runs

    device       : torch.device



    Returns

    -------

    list of benchmark result dicts

    """

    results    = []

    batch_size = next(

        (v.shape[0] for v in sample_batch.values() if isinstance(v, torch.Tensor)),

        0

    )



    # Move batch tensors to target device once

    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v

                 for k, v in sample_batch.items()}



    for model_name, model_obj, is_tft in models_info:

        # ΓöÇΓöÇ Seasonal Naive: array baseline, no neural forward pass ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

        if model_name == "Seasonal Naive":

            naive_input = batch_dev.get('encoder_target', None)

            if naive_input is None:

                results.append({

                    "model": model_name,

                    "notes": "encoder_target not found in sample batch",

                })

                continue

            # Warm-up

            for _ in range(num_warmup):

                _ = naive_input[:, -28:].clone()

            times = []

            for _ in range(num_runs):

                t0 = time.perf_counter()

                _ = naive_input[:, -28:].clone()

                times.append(time.perf_counter() - t0)

            results.append({

                "model":       model_name,

                "device":      str(device),

                "batch_size":  batch_size,

                "num_warmup":  num_warmup,

                "num_runs":    num_runs,

                "mean_ms":     float(np.mean(times) * 1000),

                "sd_ms":       float(np.std(times)  * 1000),

                "notes":       "array-based baseline (cloned), no model forward pass",

            })

            continue



        # ΓöÇΓöÇ Neural models ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

        forward_fn = forward_tft if is_tft else forward_student



        model_obj.eval()

        model_obj.to(device)



        # Warm-up (not timed)

        for _ in range(num_warmup):

            with torch.inference_mode():

                out = forward_fn(model_obj, batch_dev)

                _ = out.sum()

        if torch.cuda.is_available():

            torch.cuda.synchronize()



        # Timed runs

        times = []

        for _ in range(num_runs):

            if torch.cuda.is_available():

                torch.cuda.synchronize()

            t0 = time.perf_counter()

            with torch.inference_mode():

                out = forward_fn(model_obj, batch_dev)

                _ = out.sum()

            if torch.cuda.is_available():

                torch.cuda.synchronize()

            times.append(time.perf_counter() - t0)



        results.append({

            "model":       model_name,

            "device":      str(device),

            "batch_size":  batch_size,

            "num_warmup":  num_warmup,

            "num_runs":    num_runs,

            "mean_ms":     float(np.mean(times) * 1000),

            "sd_ms":       float(np.std(times)  * 1000),

            "notes":       "forward only, no trainer or data loading",

        })

        print(f"  Benchmark {model_name:26s}: "

              f"{float(np.mean(times)*1000):.3f} ┬▒ {float(np.std(times)*1000):.3f} ms")



    return results





# ΓöÇΓöÇΓöÇ Main ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def _run_alignment_audit(df_part, decoded, origin, H, actuals):
    # Independent alignment audit (executed exactly once)
    if not hasattr(_run_alignment_audit, "audited"):
        audit_positions = [0, len(decoded) // 2, len(decoded) - 1]
        for pos in audit_positions:
            sid = str(decoded.iloc[pos]["id"])
            raw_target = (
                df_part[
                    (df_part["id"].astype(str) == sid) &
                    (df_part["time_idx"].between(origin, origin + H - 1))
                ]
                .sort_values("time_idx")["sales"]
                .values
            )
            batch_target = actuals[pos]
            assert np.allclose(raw_target, batch_target), f"Alignment audit failed for {sid} at origin {origin}!"
        _run_alignment_audit.audited = True

def main():
    parser = argparse.ArgumentParser(description="Evaluate M5 Models")
    parser.add_argument("--env", type=str, default="local")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--student-nokd-checkpoint", type=str, default=None)
    parser.add_argument("--student-kd-checkpoint", type=str, default=None)
    parser.add_argument(
        "--selected-student-label",
        type=str,
        default="Student With KD",
        help="Display label for the third student checkpoint; does not alter evaluation logic.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    set_seed(cfg.environment.seed)
    
    # Assert partial-data is disabled for final evaluation
    assert getattr(cfg.environment, "max_stores", None) is None, "max_stores must be None for final evaluation"
    assert getattr(cfg.environment, "max_batches_per_store", None) is None, "max_batches_per_store must be None for final evaluation"

    assert cfg.dataset.splits.train.end == 1359, f"Expected train.end=1359, got {cfg.dataset.splits.train.end}"
    assert cfg.dataset.splits.test_stream.start == 1554, f"Expected test_stream.start=1554, got {cfg.dataset.splits.test_stream.start}"
    assert cfg.dataset.splits.test_stream.end == 1941, f"Expected test_stream.end=1941, got {cfg.dataset.splits.test_stream.end}"
    assert cfg.dataset.lookback_window == 90, f"Expected lookback=90, got {cfg.dataset.lookback_window}"
    assert cfg.dataset.prediction_window == 28, f"Expected horizon=28, got {cfg.dataset.prediction_window}"
    assert cfg.dataset.window_stride == 7, f"Expected stride=7, got {cfg.dataset.window_stride}"
    assert hasattr(cfg.dataset.splits, "event_ood_test"), "Config missing event_ood_test split — check YAML key name"
    assert cfg.dataset.splits.id_test.start == 1554 and cfg.dataset.splits.id_test.end == 1581
    assert cfg.dataset.splits.event_ood_test.start == 1819 and cfg.dataset.splits.event_ood_test.end == 1846
    assert cfg.dataset.splits.extended_ood_test.start == 1914 and cfg.dataset.splits.extended_ood_test.end == 1941

    outputs_dir = resolve_path(cfg.environment.outputs_dir)
    from utils.paths import get_dataset_dir
    ds_dir = get_dataset_dir(cfg)
    
    teacher_chk = resolve_model_checkpoint(
        args.teacher_checkpoint, cfg.evaluation.teacher_checkpoint, outputs_dir,
        os.path.join("teacher", args.exp_name, "best_tft_teacher.ckpt")
    )
    student_nokd_chk = resolve_model_checkpoint(
        args.student_nokd_checkpoint, cfg.evaluation.student_nokd_checkpoint, outputs_dir,
        os.path.join("student", "no_kd", args.exp_name, "best_student.ckpt")
    )
    student_kd_chk = resolve_model_checkpoint(
        args.student_kd_checkpoint, cfg.evaluation.student_kd_checkpoint, outputs_dir,
        os.path.join("student", "kd", args.exp_name, "best_student.ckpt")
    )

    print("\nResolved checkpoints:")
    print(f"Teacher: {teacher_chk}")
    print(f"Student (No KD): {student_nokd_chk}")
    print(f"Selected student ({args.selected_student_label}): {student_kd_chk}")

    df = load_dataset_from_cache(artifacts_dir=ds_dir, store_filter=cfg.environment.store_filter)
    
    training_data = build_timeseries_dataset(df, cfg, is_train=True)
    
    teacher = TemporalFusionTransformer.load_from_checkpoint(teacher_chk)
    student_nokd = M5TransformerStudent.load_from_checkpoint(student_nokd_chk, training_dataset=training_data, strict=True)
    student_kd = M5TransformerStudent.load_from_checkpoint(student_kd_chk, training_dataset=training_data, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device).eval()
    student_nokd = student_nokd.to(device).eval()
    student_kd = student_kd.to(device).eval()

    train_end = cfg.dataset.splits.train.end
    df_train = df[df['time_idx'] <= train_end].copy()
    weights_dict, scales_dict, scale_diag = compute_wrmsse_weights_and_scales(df_train, train_end)
    mase_scales_dict = compute_mase_scales(df_train, train_end)
    series_ids = df['id'].astype(str).drop_duplicates().sort_values().to_numpy()
    assert len(series_ids) == FULL_M5_SERIES_COUNT
    missing_mase = [sid for sid in series_ids if sid not in mase_scales_dict]
    assert not missing_mase, f"Missing MASE scales for {len(missing_mase)} series"
    scales_array = np.array([mase_scales_dict[sid] for sid in series_ids])

    # --- Controlled latency benchmark (separate from operational timing) ---
    print("\nRunning controlled inference benchmark...")
    _bench_store = resolve_stores(cfg.environment.store_filter)[0]
    _bench_df = load_from_cache(artifacts_dir=ds_dir, store_filter=_bench_store)
    _bench_slice = _bench_df[
        (_bench_df['time_idx'] >= train_end - cfg.dataset.lookback_window + 1) &
        (_bench_df['time_idx'] <= train_end + cfg.dataset.prediction_window)
    ].copy()
    del _bench_df
    for col in ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                'weekday', 'month', 'year', 'event_name_1', 'event_type_1']:
        if col in _bench_slice.columns:
            _bench_slice[col] = _bench_slice[col].astype(str).astype('category')
    _bench_ds = TimeSeriesDataSet.from_dataset(training_data, _bench_slice, predict=True, stop_randomization=True)
    _bench_loader = _bench_ds.to_dataloader(train=False, batch_size=args.batch_size, shuffle=False, num_workers=0)
    _bench_batch_x, _ = next(iter(_bench_loader))
    del _bench_slice, _bench_ds, _bench_loader
    benchmark_models_info = [
        ("Seasonal Naive", None, False),
        ("TFT Teacher", teacher, True),
        ("Student Without KD", student_nokd, False),
        (args.selected_student_label, student_kd, False),
    ]
    bench_results = benchmark_inference(benchmark_models_info, _bench_batch_x, device)
    bench_csv = os.path.join(
        os.path.join(resolve_path(cfg.environment.outputs_dir), "evaluation", args.exp_name),
        "benchmark_results.csv"
    )
    os.makedirs(os.path.dirname(bench_csv), exist_ok=True)
    pd.DataFrame(bench_results).to_csv(bench_csv, index=False)
    del _bench_batch_x
    gc.collect()

    results = []

    diagnostics_dict = {}
    
    windows = [
        ("ID Reference", cfg.dataset.splits.id_test.start, cfg.dataset.splits.id_test.end),
        ("Event-Intensive OOD", cfg.dataset.splits.event_ood_test.start, cfg.dataset.splits.event_ood_test.end),
        ("Extended-Gap OOD", cfg.dataset.splits.extended_ood_test.start, cfg.dataset.splits.extended_ood_test.end)
    ]
    
    test_stream_start = cfg.dataset.splits.test_stream.start
    test_stream_end = cfg.dataset.splits.test_stream.end
    H = cfg.dataset.prediction_window
    L = cfg.dataset.lookback_window

    overall_origins = list(range(test_stream_start, test_stream_end - H + 2, 7))
    assert len(overall_origins) == 52, f"Expected 52 origins, got {len(overall_origins)}"
    assert overall_origins[0] == 1554
    assert overall_origins[-1] == 1911
    assert overall_origins[-1] + H - 1 == 1938
    for origin in overall_origins:
        windows.append((f"Overall Test Stream (Origin {origin})", origin, origin + H - 1))

    eval_exp_dir = os.path.join(outputs_dir, "evaluation", args.exp_name)
    os.makedirs(eval_exp_dir, exist_ok=True)
    inc_csv = os.path.join(eval_exp_dir, "evaluation_results_incremental.csv")
    run_state_path = os.path.join(eval_exp_dir, "run_state.json")

    def get_hash(path):
        with open(path, "rb") as _f:
            return hashlib.sha256(_f.read()).hexdigest()

    with open(__file__, "rb") as _f:
        script_hash = hashlib.sha256(_f.read()).hexdigest()

    teacher_hash = get_hash(teacher_chk)
    student_nokd_hash = get_hash(student_nokd_chk)
    student_kd_hash = get_hash(student_kd_chk)

    run_state = {
        "script_hash": script_hash,
        "teacher_hash": teacher_hash,
        "student_nokd_hash": student_nokd_hash,
        "student_kd_hash": student_kd_hash,
        "train_end": int(cfg.dataset.splits.train.end),
        "test_stream_start": int(cfg.dataset.splits.test_stream.start),
        "test_stream_end": int(cfg.dataset.splits.test_stream.end),
        "lookback": int(cfg.dataset.lookback_window),
        "horizon": int(cfg.dataset.prediction_window),
        "stride": int(cfg.dataset.window_stride),
        "seed": int(cfg.environment.seed),
        "series_count": FULL_M5_SERIES_COUNT,
    }

    expected_models = {"Seasonal Naive", "TFT Teacher", "Student Without KD", args.selected_student_label}
    expected_horizons = {"Overall (1-28)", "Short (1-7)", "Medium (8-14)", "Long (15-28)"}
    expected_pairs = {(m, h) for m in expected_models for h in expected_horizons}

    completed_origins = set()
    if os.path.exists(inc_csv) and os.path.exists(run_state_path):
        with open(run_state_path, "r") as _f:
            saved_state = json.load(_f)
        if saved_state != run_state:
            raise RuntimeError(
                "Incremental CSV exists but run state does not match current run. "
                "Use a different --exp-name or delete the existing output directory."
            )
        df_inc = pd.read_csv(inc_csv)
        if not df_inc.empty:
            for w, group in df_inc.groupby('Window'):
                actual_pairs = set(zip(group["Model"], group["Horizon"]))
                if (
                    len(group) == 16
                    and actual_pairs == expected_pairs
                    and np.isfinite(group[['WRMSSE', 'MAE', 'RMSE', 'MASE', 'WAPE']].values).all()
                ):
                    completed_origins.add(w)
            if completed_origins:
                df_inc = df_inc[df_inc["Window"].isin(completed_origins)].copy()
                results = df_inc.to_dict("records")
                df_inc.to_csv(inc_csv, index=False)
                print(
                    f"Resuming from {len(completed_origins)} "
                    "fully completed windows."
                )
            else:
                results = []
    elif os.path.exists(inc_csv) and not os.path.exists(run_state_path):
        raise RuntimeError(
            "Incremental CSV exists but no run_state.json found. "
            "Cannot safely resume. Use a different --exp-name or delete the existing output directory."
        )

    with open(run_state_path, "w") as _f:
        json.dump(run_state, _f, indent=2)

    models_eval = [
        ("TFT Teacher", teacher),
        ("Student Without KD", student_nokd),
        (args.selected_student_label, student_kd)
    ]
    


    for test_name, start_day, end_day in windows:
        if test_name in completed_origins:
            continue
            
        print(f"\n--- Evaluating Models on {test_name} (Days {start_day} to {end_day}) ---")
        
        df_test_gt = df[(df['time_idx'] >= start_day) & (df['time_idx'] <= end_day)].copy()
        df_test_gt['id'] = df_test_gt['id'].astype(str)
        df_test_gt = df_test_gt.sort_values(by=['id', 'time_idx']).reset_index(drop=True)
        
        start_t = time.perf_counter()
        df_naive_source = df[(df['time_idx'] >= (start_day - 28)) & (df['time_idx'] < start_day)].copy()
        df_naive_source['id'] = df_naive_source['id'].astype(str)
        df_naive_source = df_naive_source.sort_values(by=['id', 'time_idx']).reset_index(drop=True)
        naive_time = time.perf_counter() - start_t
        
        stores = resolve_stores(cfg.environment.store_filter)
        min_idx = start_day - L
        
        all_actuals = []
        all_naive = []
        all_decoded = []
        model_preds = {n: [] for n, _ in models_eval}
        model_times = {n: 0.0 for n, _ in models_eval}
        
        for store in stores:
            df_part = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
            if df_part is None: continue
            
            df_part_sliced = df_part[(df_part['time_idx'] >= min_idx) & (df_part['time_idx'] <= end_day)].copy()
            del df_part
            
            for col in ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'weekday', 'month', 'year', 'event_name_1', 'event_type_1']:
                if col in df_part_sliced.columns:
                    df_part_sliced[col] = df_part_sliced[col].astype(str).astype('category')
                    
            if len(df_part_sliced) == 0: continue
            
            part_ds = TimeSeriesDataSet.from_dataset(training_data, df_part_sliced, predict=True, stop_randomization=True)
            _decoded = part_ds.decoded_index
            assert _decoded["time_idx_first_prediction"].nunique() == 1, "Multiple prediction starts in one store partition"
            assert int(_decoded["time_idx_first_prediction"].iloc[0]) == start_day, f"Prediction start mismatch: expected {start_day}, got {_decoded['time_idx_first_prediction'].iloc[0]}"
            assert _decoded["id"].is_unique, "Duplicate series IDs in store partition"
            part_loader = part_ds.to_dataloader(train=False, batch_size=args.batch_size, shuffle=False, num_workers=cfg.environment.num_workers)
            
            store_actuals = []
            store_naive = []
            for batch_x, batch_y in part_loader:
                target = batch_y[0] if isinstance(batch_y, (tuple, list)) else batch_y
                store_actuals.append(target.cpu().numpy())
                store_naive.append(batch_x['encoder_target'][:, -28:].cpu().numpy())
            
            actuals_np = np.concatenate(store_actuals, axis=0)
            naive_np = np.concatenate(store_naive, axis=0)

            assert actuals_np.shape[1] == H, f"Expected H={H} targets, got {actuals_np.shape[1]}"
            assert naive_np.shape == actuals_np.shape, "Naive shape mismatch"
            assert np.isfinite(actuals_np).all(), f"Non-finite actuals for store={store}, window={test_name}"
            assert np.isfinite(naive_np).all(), f"Non-finite naive for store={store}, window={test_name}"

            all_actuals.append(actuals_np)
            all_naive.append(naive_np)
            all_decoded.append(part_ds.decoded_index)
            
            _run_alignment_audit(df_part_sliced, part_ds.decoded_index, start_day, H, actuals_np)
            del df_part_sliced
            
            for m_name, m_obj in models_eval:
                st = time.perf_counter()
                preds = get_predictions(m_obj, part_loader)
                assert preds.shape == actuals_np.shape, (
                    f"{m_name} shape mismatch: predictions={preds.shape}, targets={actuals_np.shape}"
                )
                assert np.isfinite(preds).all(), (
                    f"Non-finite predictions from {m_name} for store={store}, window={test_name}"
                )
                model_times[m_name] += time.perf_counter() - st
                model_preds[m_name].append(preds)
            
            del part_loader, part_ds
            gc.collect()
            
        concatenated_decoded = pd.concat(all_decoded, ignore_index=True)
        concatenated_decoded['id'] = concatenated_decoded['id'].astype(str)
        order = concatenated_decoded.assign(row_idx=np.arange(len(concatenated_decoded))).sort_values('id')['row_idx'].to_numpy()
        
        actuals = np.concatenate(all_actuals, axis=0)[order]
        naive_forecasts = np.concatenate(all_naive, axis=0)[order]
        for m_name in model_preds:
            model_preds[m_name] = np.concatenate(model_preds[m_name], axis=0)[order]
            
        decoded_ids_sorted = concatenated_decoded.sort_values('id')['id'].to_numpy()
        ground_truth_ids = df_test_gt['id'].drop_duplicates().to_numpy()
        assert np.array_equal(decoded_ids_sorted, ground_truth_ids), "Alignment failed!"
        
        num_series = actuals.shape[0]
        assert num_series == FULL_M5_SERIES_COUNT, f"Expected {FULL_M5_SERIES_COUNT} series, got {num_series}"
        assert naive_forecasts.shape == actuals.shape
        assert np.isfinite(actuals).all()
        assert np.isfinite(naive_forecasts).all()
        
        models_final = [("Seasonal Naive", naive_forecasts, naive_time)]
        for m_name, _ in models_eval:
            preds = model_preds[m_name]
            assert preds.shape == actuals.shape
            assert np.isfinite(preds).all()
            models_final.append((m_name, preds, model_times[m_name]))
            
        slices = [
            ("Overall (1-28)", 0, 28),
            ("Short (1-7)", 0, 7),
            ("Medium (8-14)", 7, 14),
            ("Long (15-28)", 14, 28)
        ]
        
        window_results = []
        for name, forecasts, inf_time in models_final:
            normalized_inf_time = (inf_time / num_series) * 1000.0
            
            for slice_name, start_idx, end_idx in slices:
                actuals_slice = actuals[:, start_idx:end_idx]
                forecasts_slice = forecasts[:, start_idx:end_idx]
                
                slice_start_day = start_day + start_idx
                slice_end_day = start_day + end_idx - 1
                
                df_test_gt_slice = df_test_gt[(df_test_gt['time_idx'] >= slice_start_day) & (df_test_gt['time_idx'] <= slice_end_day)].copy()
                df_preds_slice = df_test_gt_slice.copy()
                df_preds_slice['sales'] = forecasts_slice.flatten()
                
                mae, rmse, wape = compute_point_metrics(actuals_slice.flatten(), forecasts_slice.flatten())
                wrmsse, level_wrmsses = compute_hierarchical_wrmsse(df_test_gt_slice, df_preds_slice, weights_dict, scales_dict)
                mase = compute_mase(actuals_slice, forecasts_slice, scales_array)
                
                if slice_name == "Overall (1-28)":
                    diagnostics_dict[(name, test_name)] = level_wrmsses
                
                window_results.append({
                    "Window": test_name,
                    "Model": name,
                    "Horizon": slice_name,
                    "WRMSSE": float(wrmsse),
                    "MAE": float(mae),
                    "RMSE": float(rmse),
                    "MASE": float(mase),
                    "WAPE": float(wape),
                    "Inference_Time_Sec": float(inf_time),
                    "Inference_Time_Per_1k_Sec": float(normalized_inf_time)
                })
        
        results.extend(window_results)
        pd.DataFrame(results).to_csv(inc_csv, index=False)
        gc.collect()

    df_res = pd.DataFrame(results)
    
    # Calculate Summary for Overall Test Stream
    overall_rows = df_res[df_res['Window'].str.startswith('Overall Test Stream (Origin ')]
    if not overall_rows.empty:
        summary_rows = []
        for model in overall_rows['Model'].unique():
            for horizon in overall_rows['Horizon'].unique():
                subset = overall_rows[(overall_rows['Model'] == model) & (overall_rows['Horizon'] == horizon)]
                metrics = subset[['WRMSSE', 'MAE', 'RMSE', 'MASE', 'WAPE']]
                
                mean_dict = metrics.mean().to_dict()
                std_dict = metrics.std().to_dict()
                median_dict = metrics.median().to_dict()
                min_dict = metrics.min().to_dict()
                max_dict = metrics.max().to_dict()
                
                row = {"Window": "Overall Test Stream", "Model": model, "Horizon": horizon}
                for k in metrics.columns:
                    row[k] = mean_dict[k]
                    row[f"{k}_SD"] = std_dict[k]
                    row[f"{k}_Median"] = median_dict[k]
                    row[f"{k}_Min"] = min_dict[k]
                    row[f"{k}_Max"] = max_dict[k]
                summary_rows.append(row)
        
        df_summary = pd.DataFrame(summary_rows)
        df_final = pd.concat([df_res[~df_res['Window'].str.startswith('Overall Test Stream (Origin ')], df_summary], ignore_index=True)
    else:
        df_final = df_res

    csv_filepath = os.path.join(eval_exp_dir, "evaluation_results_final.csv")
    df_final.to_csv(csv_filepath, index=False)

    # --- Relative degradation: ID vs OOD ---
    degradation_rows = []
    for model_name in expected_models:
        rows = df_final[
            (df_final["Model"] == model_name) &
            (df_final["Horizon"] == "Overall (1-28)")
        ]
        id_row = rows[rows["Window"] == "ID Reference"]
        event_row = rows[rows["Window"] == "Event-Intensive OOD"]
        extended_row = rows[rows["Window"] == "Extended-Gap OOD"]
        assert len(id_row) == 1, f"Expected 1 ID Reference row for {model_name}"
        assert len(event_row) == 1, f"Expected 1 Event-Intensive OOD row for {model_name}"
        assert len(extended_row) == 1, f"Expected 1 Extended-Gap OOD row for {model_name}"
        for metric in ["WRMSSE", "MAE", "RMSE", "MASE", "WAPE"]:
            id_value = float(id_row.iloc[0][metric])
            event_value = float(event_row.iloc[0][metric])
            extended_value = float(extended_row.iloc[0][metric])
            degradation_rows.append({
                "Model": model_name,
                "Metric": metric,
                "ID_to_Event_OOD_Percent": ((event_value - id_value) / id_value) * 100,
                "ID_to_Extended_OOD_Percent": ((extended_value - id_value) / id_value) * 100,
            })
    pd.DataFrame(degradation_rows).to_csv(
        os.path.join(eval_exp_dir, "relative_degradation.csv"), index=False
    )
    
    save_metadata(
        eval_exp_dir,
        cfg.environment.seed,
        additional_fields={
            "script_hash": script_hash,
            "checkpoints": {
                "teacher": {"path": teacher_chk, "sha256": teacher_hash},
                "student_nokd": {"path": student_nokd_chk, "sha256": student_nokd_hash},
                "selected_student": {
                    "path": student_kd_chk,
                    "sha256": student_kd_hash,
                    "label": args.selected_student_label,
                }
            },
            "wrmsse_scale_diagnostics": scale_diag
        }
    )

if __name__ == "__main__":
    main()

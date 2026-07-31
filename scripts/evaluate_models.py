"""
evaluate_models.py — M5 Model Evaluation Script

Evaluates four models (Seasonal Naive, TFT Teacher, Student w/o KD, Student w/ KD)
across:
  - Three predefined fixed windows: ID Reference, Event-Intensive OOD, Extended-Gap OOD
  - Overall held-out test stream: 52 seven-day-aligned origins (d1554–d1911)

Methodology (P2 draft):
  - 90-day historical input (lookback), 28-day horizon, 7-day stride
  - Metrics: WRMSSE (primary), MAE, RMSE, MASE, WAPE
  - Separate controlled inference benchmark distinct from per-origin operational runtime
"""

import os
import sys
import json
import gc
import time
import argparse
import glob
import hashlib

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Add repository root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, save_config, save_metadata
from utils.paths import resolve_path
from utils.seed import set_seed
from data.cache import load_from_cache, load_dataset_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset
from models.student import M5TransformerStudent

# ─── Constants ────────────────────────────────────────────────────────────────

# Official 12 M5 aggregation levels
HIERARCHY_LEVELS = [
    [],                               # Level  1: All products, all stores
    ['state_id'],                     # Level  2: All products, by state
    ['store_id'],                     # Level  3: All products, by store
    ['cat_id'],                       # Level  4: All products, by category
    ['dept_id'],                      # Level  5: All products, by department
    ['state_id', 'cat_id'],           # Level  6: By state and category
    ['state_id', 'dept_id'],          # Level  7: By state and department
    ['store_id', 'cat_id'],           # Level  8: By store and category
    ['store_id', 'dept_id'],          # Level  9: By store and department
    ['item_id'],                      # Level 10: Individual product, all stores
    ['item_id', 'state_id'],          # Level 11: Individual product, by state
    ['id'],                           # Level 12: Individual product, by store
]

CAT_COLS = [
    'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
    'weekday', 'month', 'year', 'event_name_1', 'event_type_1',
]

HIERARCHY_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

HORIZON_SLICES = [
    ("Overall (1-28)", 0, 28),
    ("Short (1-7)",    0,  7),
    ("Medium (8-14)",  7, 14),
    ("Long (15-28)",  14, 28),
]

FULL_M5_SERIES_COUNT = 30490

METRICS = ["WRMSSE", "MAE", "RMSE", "MASE", "WAPE"]

# ─── Scale Helpers ─────────────────────────────────────────────────────────────

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
    reason : str — one of 'valid', 'insufficient_length', 'all_zero', 'zero_variance'
    """
    FALLBACK = 1e-4
    if len(series) < 2:
        return FALLBACK, "insufficient_length"
    first_nonzero_idx = np.argmax(series > 0)
    if series[first_nonzero_idx] == 0:
        # np.argmax returns 0 when all elements are 0 (no True found)
        return FALLBACK, "all_zero"
    trimmed = series[first_nonzero_idx:]
    if len(trimmed) < 2:
        return FALLBACK, "insufficient_length"
    sq_diffs = np.diff(trimmed.astype(float)) ** 2
    scale = float(np.mean(sq_diffs))
    if scale <= 0:
        return FALLBACK, "zero_variance"
    return scale, "valid"


def compute_wrmsse_weights_and_scales(df_train, train_end):
    """
    Pre-computes M5 hierarchy scales (naive std) and dollar-value weights.

    Applies M5-aligned corrections:
      - Reindexes each aggregate series to a full contiguous time_idx range
        (zero-fills calendar gaps) before computing the scale.
      - Strips leading zero observations per series (before first non-zero sale).

    Returns
    -------
    weights_dict     : dict  Level_k → {group_key: weight}
    scales_dict      : dict  Level_k → {group_key: scale}
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
        scale_reasons = {}  # key_str → reason string

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
            df_grouped_weight = (df_weight_window.groupby(group_cols)['dollar_value']
                                                 .sum().reset_index())

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
                w = float(row['dollar_value'] / total_dollar_sum) if total_dollar_sum > 0 else 0.0
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
        print("  WARNING: >5% of WRMSSE weight is on fallback scales — "
              "inspect wrmsse_scale_diagnostics.json before reporting results.")

    return weights_dict, scales_dict, scale_diagnostics


def compute_mase_scales(df_train):
    """
    Pre-computes per-series MASE denominator using a 28-day seasonal lag.

    MASE denominator = mean |y_t - y_{t-28}| over the training period,
    computed only on observations at or after the first non-zero sale per series.

    The 28-day seasonal lag matches the Seasonal Naive baseline used as the
    denominator and is distinct from the first-difference RMSSE scale.

    Returns dict {id_str: scale_float}
    """
    print("Pre-computing per-series MASE scales (28-day seasonal lag, "
          "leading-zero stripping)...")
    df_sorted = (df_train
                 .sort_values(by=['id', 'time_idx'])
                 .reset_index(drop=True))

    # 28-day lag shift per series (boundary-safe via groupby shift)
    prev_sales = df_sorted.groupby('id', observed=True)['sales'].shift(28)

    # Mark observations within the active sale period (at or after first non-zero sale)
    df_sorted['cummax_sales'] = (df_sorted.groupby('id', observed=True)['sales']
                                          .cummax())
    df_sorted['in_sale_period'] = df_sorted['cummax_sales'] > 0

    df_sorted['abs_diff'] = np.abs(df_sorted['sales'] - prev_sales)
    # Only count observations in sale period where lag is available
    df_sorted['valid_diff'] = df_sorted['abs_diff'].where(
        df_sorted['in_sale_period'] & prev_sales.notna()
    )

    scales_raw = df_sorted.groupby('id', observed=True)['valid_diff'].mean()
    n_missing = scales_raw.isna().sum()
    n_zero    = (scales_raw == 0).sum()
    total     = len(scales_raw)
    
    print(f"  MASE diagnostics: {n_missing} missing scales, {n_zero} zero scales. "
          f"({(n_missing + n_zero) / total * 100:.2f}% of series use fallback 1.0)")

    scales = scales_raw.fillna(1.0).replace(0.0, 1.0)
    return scales.to_dict()


# ─── Metric Functions ──────────────────────────────────────────────────────────

def compute_hierarchical_wrmsse(df_test_gt, df_test_preds, weights_dict, scales_dict):
    """
    Computes M5 WRMSSE across all 12 hierarchy levels.
    Expects df_test_gt and df_test_preds to have columns:
      time_idx, sales, and all hierarchy identifiers.
    """
    level_wrmsses = []

    for level_idx, group_cols in enumerate(HIERARCHY_LEVELS, 1):
        level_name = f"Level_{level_idx}"
        level_weights = weights_dict[level_name]
        level_scales  = scales_dict[level_name]

        rmsses  = []
        weights = []

        if len(group_cols) == 0:
            gt_agg   = df_test_gt.groupby('time_idx')['sales'].sum().sort_index().values
            pred_agg = df_test_preds.groupby('time_idx')['sales'].sum().sort_index().values
            mse = np.mean((gt_agg - pred_agg) ** 2)
            rmsses.append(np.sqrt(mse / level_scales['Total']))
            weights.append(1.0)
        else:
            df_gt_grouped   = (df_test_gt.groupby(group_cols + ['time_idx'])['sales']
                                         .sum().reset_index())
            df_pred_grouped = (df_test_preds.groupby(group_cols + ['time_idx'])['sales']
                                            .sum().reset_index())
            df_merged = df_gt_grouped.merge(
                df_pred_grouped, on=group_cols + ['time_idx'], suffixes=('_gt', '_pred')
            )
            for keys, group in df_merged.groupby(group_cols):
                if isinstance(keys, tuple):
                    key_str = "_".join(str(k) for k in keys)
                else:
                    key_str = str(keys)
                gt_vals   = group.sort_values('time_idx')['sales_gt'].values
                pred_vals = group.sort_values('time_idx')['sales_pred'].values
                mse   = np.mean((gt_vals - pred_vals) ** 2)
                
                assert key_str in level_scales, f"Missing scale for {level_name}/{key_str}"
                assert key_str in level_weights, f"Missing weight for {level_name}/{key_str}"
                
                scale = level_scales[key_str]
                w     = level_weights[key_str]
                rmsses.append(np.sqrt(mse / scale))
                weights.append(w)

        level_wrmsse = float(np.sum(np.array(rmsses) * np.array(weights)))
        level_wrmsses.append(level_wrmsse)

    overall_wrmsse = float(np.mean(level_wrmsses))
    return overall_wrmsse, level_wrmsses


def compute_point_metrics(actuals_flat, forecasts_flat):
    """Returns (MAE, RMSE, WAPE) for flat arrays."""
    mae  = float(np.mean(np.abs(actuals_flat - forecasts_flat)))
    rmse = float(np.sqrt(np.mean((actuals_flat - forecasts_flat) ** 2)))
    total_sales = float(np.sum(actuals_flat))
    wape = float(np.sum(np.abs(actuals_flat - forecasts_flat)) / total_sales) \
           if total_sales > 0 else 0.0
    return mae, rmse, wape


def compute_mase(actuals_slice, forecasts_slice, scales_array):
    """
    Computes mean MASE over all series in a slice.
    actuals_slice   : (num_series, slice_len)
    forecasts_slice : (num_series, slice_len)
    scales_array    : (num_series,) — 28-day seasonal lag denominator per series
    """
    mae_per_series  = np.mean(np.abs(actuals_slice - forecasts_slice), axis=1)
    mase_per_series = mae_per_series / scales_array
    return float(np.mean(mase_per_series))


# ─── Alignment Helpers ─────────────────────────────────────────────────────────

def _build_long_form_table(ids, actuals, naive_forecasts, model_preds_dict, origin, H):
    """
    Build a vectorized long-form prediction table for one evaluation window.

    Parameters
    ----------
    ids              : array of series ID strings, shape (N,)
    actuals          : array shape (N, H)
    naive_forecasts  : array shape (N, H)
    model_preds_dict : {model_name: array(N, H)} for neural models
    origin           : first target day index (int)
    H                : prediction horizon length (int)

    Returns
    -------
    pd.DataFrame with columns:
        id, origin, h, time_idx_target, actual, naive, pred_<model_name>...
    """
    n = len(ids)
    id_arr          = np.repeat(ids, H)               # N*H
    h_arr           = np.tile(np.arange(H), n)        # N*H
    time_idx_arr    = origin + h_arr

    df = pd.DataFrame({
        'id':             id_arr,
        'origin':         origin,
        'h':              h_arr,
        'time_idx_target': time_idx_arr,
        'actual':         actuals.flatten(),           # row-major → series i owns h_arr[i*H:(i+1)*H]
        'naive':          naive_forecasts.flatten(),
    })
    for model_name, preds in model_preds_dict.items():
        df[f'pred_{model_name}'] = preds.flatten()

    return df


def _assert_alignment(df_long, expected_rows, model_names, id_meta):
    """
    Assert structural integrity of the long-form prediction table before metric
    computation. Raises AssertionError on any violation.
    """
    assert len(df_long) == expected_rows, \
        f"Row count mismatch: {len(df_long)} != {expected_rows}"
    assert not df_long[['id', 'time_idx_target']].duplicated().any(), \
        "Duplicate (id, time_idx_target) pairs found"
    assert df_long['actual'].notna().all(), \
        "Missing actuals in long-form table"
    assert df_long['naive'].notna().all(), \
        "Missing naive predictions in long-form table"
    for model_name in model_names:
        col = f'pred_{model_name}'
        if col in df_long.columns:
            assert df_long[col].notna().all(), \
                f"Missing predictions for model: {model_name}"
    # Hierarchy metadata
    for hcol in HIERARCHY_COLS:
        if hcol in df_long.columns:
            assert df_long[hcol].notna().all(), \
                f"Missing hierarchy column after join: {hcol}"
    assert df_long['id'].isin(id_meta.index).all(), \
        "Some series IDs have no hierarchy metadata — join may be incomplete"


def _build_wrmsse_df(df_long_slice, value_col, id_meta):
    """
    Build a DataFrame suitable for compute_hierarchical_wrmsse from a slice
    of the long-form table.

    Returns df with columns: id, time_idx, sales, item_id, dept_id, cat_id,
    store_id, state_id.
    """
    df = df_long_slice[['id', 'time_idx_target', value_col]].copy()
    df = df.rename(columns={'time_idx_target': 'time_idx', value_col: 'sales'})
    available_hcols = [c for c in HIERARCHY_COLS if c in id_meta.columns]
    df = df.join(id_meta[available_hcols], on='id', how='left')
    return df



# ─── Inference Functions ───────────────────────────────────────────────────────

def run_inference_tft(model, loader, device):
    """
    Run TFT inference using model.predict() (operational runtime).
    TFT timing includes Lightning Trainer setup and prediction orchestration.
    This is labelled as operational runtime, not pure model latency.

    Returns (preds_np: ndarray (N, H), elapsed_sec: float)
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    preds = model.predict(
        loader,
        mode="prediction",
        trainer_kwargs={
            "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
            "devices": 1,
        },
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return preds.cpu().numpy(), elapsed


def run_inference_student(model, loader, device):
    """
    Run student model inference with a manual batch loop (operational runtime).

    Returns (preds_np: ndarray (N, H), elapsed_sec: float)
    """
    model.eval()
    model.to(device)
    all_preds = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            x, _ = batch
            for k in list(x.keys()):
                if isinstance(x[k], torch.Tensor):
                    x[k] = x[k].to(device)
            preds = model(x)
            all_preds.append(preds.cpu())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return torch.cat(all_preds, dim=0).numpy(), elapsed


# ─── Benchmark Forward Adapters ────────────────────────────────────────────────

def forward_tft(model, x):
    """
    TFT forward adapter for controlled benchmark.
    Calls model(x) to get a structured output dict, then model.to_prediction()
    to extract the point forecast tensor (shape: batch_size × H).
    """
    output = model(x)
    return model.to_prediction(output)


def forward_student(model, x):
    """Student forward adapter for controlled benchmark (returns shape: batch_size × H)."""
    return model(x)


# ─── Checkpoint Helpers ────────────────────────────────────────────────────────

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

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


# ─── Shared Per-Store Inference Utility ───────────────────────────────────────

def _run_store_inference(store, ds_dir, training_data, cfg, args, device,
                         models_info, origin, slice_start, slice_end, H):
    """
    Load one store partition, build a TimeSeriesDataSet for the given origin
    window, extract actuals and naive forecasts from the DataLoader, and run
    inference for every neural model.

    Parameters
    ----------
    models_info : list of (model_name, model_obj, is_tft)

    Returns
    -------
    dict with keys:
        decoded        : pd.DataFrame  (decoded_index)
        actuals        : ndarray (n_store_series, H)
        naive          : ndarray (n_store_series, H)
        preds          : {model_name: ndarray (n_store_series, H)}
        model_times    : {model_name: float}
        naive_time     : float
    Or None if the store partition is empty.
    """
    df_part = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
    if df_part is None:
        raise FileNotFoundError(f"Cache not found for store: {store}")

    df_sliced = df_part[
        (df_part['time_idx'] >= slice_start) &
        (df_part['time_idx'] <= slice_end)
    ].copy()
    del df_part

    for col in CAT_COLS:
        if col in df_sliced.columns:
            df_sliced[col] = df_sliced[col].astype(str).astype('category')

    if len(df_sliced) == 0:
        return None

    part_ds = TimeSeriesDataSet.from_dataset(
        training_data, df_sliced, predict=True, stop_randomization=True
    )
    decoded = part_ds.decoded_index

    # ── Assertions: exactly one origin, no duplicate IDs ──────────────────
    n_unique_origins = decoded["time_idx_first_prediction"].nunique()
    actual_origin    = int(decoded["time_idx_first_prediction"].iloc[0])
    assert n_unique_origins == 1, \
        (f"store={store}, origin={origin}: expected 1 unique origin, "
         f"got {n_unique_origins}")
    assert actual_origin == origin, \
        (f"store={store}: origin mismatch — "
         f"expected {origin}, got {actual_origin}")
    assert decoded["id"].is_unique, \
        f"store={store}, origin={origin}: duplicate series IDs in decoded_index"

    part_loader = part_ds.to_dataloader(
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.environment.num_workers,
    )

    # ── Actuals + naive (one pass) ─────────────────────────────────────────
    store_actuals = []
    store_naive   = []
    t0 = time.perf_counter()
    for batch_x, batch_y in part_loader:
        target = batch_y[0] if isinstance(batch_y, (tuple, list)) else batch_y
        store_actuals.append(target.cpu().numpy())
        store_naive.append(batch_x['encoder_target'][:, -28:].cpu().numpy())
    naive_time = time.perf_counter() - t0

    part_act   = np.concatenate(store_actuals, axis=0)
    part_naive = np.concatenate(store_naive,   axis=0)

    # ── Independent Alignment Audit (One-time) ─────────────────────────────
    if not hasattr(_run_store_inference, "audited"):
        print(f"  [Audit] Independent ID-to-target alignment for store={store}, origin={origin}...")
        audit_positions = [0, len(decoded) // 2, len(decoded) - 1]
        for pos in audit_positions:
            sid = str(decoded.iloc[pos]["id"])
            raw_target = (
                df_sliced[
                    (df_sliced["id"].astype(str) == sid) &
                    (df_sliced["time_idx"].between(origin, origin + H - 1))
                ]
                .sort_values("time_idx")["sales"]
                .to_numpy()
            )
            assert len(raw_target) == H, f"Independent alignment: expected {H} raw target values for {sid}, got {len(raw_target)}"
            np.testing.assert_allclose(
                part_act[pos],
                raw_target,
                err_msg=f"Independent alignment failed for {sid}"
            )
        print("  [Audit] Passed: Row order target values match source DataFrame exactly.")
        _run_store_inference.audited = True

    del df_sliced  # Clean up memory after audit

    assert part_act.shape[1] == H, \
        f"store={store}, origin={origin}: actuals shape mismatch {part_act.shape}"
    assert len(decoded) == part_act.shape[0], \
        (f"store={store}, origin={origin}: decoded rows ({len(decoded)}) "
         f"!= actuals rows ({part_act.shape[0]})")

    # ── Neural model inference (separate timed pass per model) ────────────
    preds       = {}
    model_times = {}
    for model_name, model_obj, is_tft in models_info:
        if is_tft:
            p, elapsed = run_inference_tft(model_obj, part_loader, device)
        else:
            p, elapsed = run_inference_student(model_obj, part_loader, device)
        
        assert p.shape == part_act.shape, (
            f"Prediction shape mismatch for {model_name}: "
            f"pred={p.shape}, actual={part_act.shape}, "
            f"store={store}, origin={origin}"
        )
        assert np.isfinite(p).all(), (
            f"Non-finite predictions detected for {model_name}, "
            f"store={store}, origin={origin}"
        )
        preds[model_name]       = p
        model_times[model_name] = elapsed

    assert part_naive.shape == part_act.shape
    assert np.isfinite(part_naive).all()
    assert np.isfinite(part_act).all()

    del part_loader, part_ds
    gc.collect()

    return {
        "decoded":     decoded,
        "actuals":     part_act,
        "naive":       part_naive,
        "preds":       preds,
        "model_times": model_times,
        "naive_time":  naive_time,
    }


# ─── Fixed-Window Evaluator ────────────────────────────────────────────────────

def evaluate_fixed_window(scenarios, df, cfg, models_info, training_data,
                          weights_dict, scales_dict, mase_scales_dict, id_meta,
                          ds_dir, args, device):
    """
    Evaluate all models on three predefined fixed-window scenarios.

    Parameters
    ----------
    scenarios  : list of (name, start_day, end_day)
    models_info: list of (model_name, model_obj, is_tft)

    Returns
    -------
    list of result dicts (one per model × scenario × horizon slice)
    """
    H = cfg.dataset.prediction_window
    L = cfg.dataset.lookback_window

    stores    = resolve_stores(cfg.environment.store_filter)
    max_stores = getattr(cfg.environment, "max_stores", None)
    if max_stores is not None:
        stores = stores[:max_stores]

    results = []

    for test_name, start_day, end_day in scenarios:
        print(f"\n--- Fixed Window: {test_name} "
              f"(d{start_day}–d{end_day}) ---")

        # Data slice: exactly one lookback + one prediction window ending at end_day
        # start_day is the first target day; the lookback begins at start_day - L
        slice_start = start_day - L        # first history day
        slice_end   = end_day              # last target day  (= start_day + H - 1)

        model_names   = [name for name, _, _ in models_info]
        cumul_preds   = {name: [] for name in model_names}
        all_actuals   = []
        all_naive     = []
        all_decoded   = []
        model_times   = {name: 0.0 for name in model_names}
        naive_time    = 0.0

        for store in stores:
            result = _run_store_inference(
                store, ds_dir, training_data, cfg, args, device,
                models_info, start_day, slice_start, slice_end, H
            )
            if result is None:
                continue
            all_actuals.append(result["actuals"])
            all_naive.append(result["naive"])
            all_decoded.append(result["decoded"].copy())
            naive_time += result["naive_time"]
            for name, p in result["preds"].items():
                cumul_preds[name].append(p)
                model_times[name] += result["model_times"][name]

        # ── Aggregate across stores ────────────────────────────────────────
        actuals         = np.concatenate(all_actuals, axis=0)
        naive_forecasts = np.concatenate(all_naive,   axis=0)
        concat_decoded  = pd.concat(all_decoded, ignore_index=True)
        n_series        = actuals.shape[0]

        assert n_series == FULL_M5_SERIES_COUNT, \
            f"Dataset rule violation: {test_name} has {n_series} series " \
            f"(expected {FULL_M5_SERIES_COUNT}). Final eval must use full population."

        neural_preds = {name: np.concatenate(v, axis=0)
                        for name, v in cumul_preds.items() if v}
        ids = concat_decoded['id'].astype(str).values

        # ── Long-form table ────────────────────────────────────────────────
        df_long = _build_long_form_table(
            ids, actuals, naive_forecasts, neural_preds, start_day, H
        )
        df_long = df_long.join(id_meta, on='id', how='left')

        _assert_alignment(df_long, n_series * H, list(neural_preds.keys()), id_meta)

        # ── Compute metrics per model per horizon slice ────────────────────
        model_forecast_map = {"Seasonal Naive": "naive",
                              **{name: f"pred_{name}" for name in neural_preds}}

        for model_name, pred_col in model_forecast_map.items():
            inf_time         = naive_time if model_name == "Seasonal Naive" else model_times[model_name]
            n_forecasts      = n_series
            norm_inf_time    = (inf_time / n_forecasts * 1000.0) if n_forecasts > 0 else 0.0

            ids_sorted   = sorted(df_long['id'].unique())
            missing_mase_ids = [sid for sid in ids_sorted if sid not in mase_scales_dict]
            assert not missing_mase_ids, f"Missing MASE scales for {len(missing_mase_ids)} series"
            scales_array = np.array([mase_scales_dict[sid] for sid in ids_sorted])

            for slice_name, s_h, e_h in HORIZON_SLICES:
                df_slice = df_long[df_long['h'].between(s_h, e_h - 1)].copy()

                act_pivot  = df_slice.pivot(index='id', columns='h', values='actual').sort_index()
                pred_pivot = df_slice.pivot(index='id', columns='h', values=pred_col).sort_index()
                actuals_slice   = act_pivot.values
                forecasts_slice = pred_pivot.values

                mae, rmse, wape = compute_point_metrics(
                    actuals_slice.flatten(), forecasts_slice.flatten()
                )
                mase = compute_mase(actuals_slice, forecasts_slice, scales_array)

                df_gt_w   = _build_wrmsse_df(df_slice, 'actual',   id_meta)
                df_pred_w = _build_wrmsse_df(df_slice, pred_col,   id_meta)
                wrmsse, level_wrmsses = compute_hierarchical_wrmsse(
                    df_gt_w, df_pred_w, weights_dict, scales_dict
                )

                print(f"  {model_name:26s} | {slice_name:15s} -> "
                      f"WRMSSE: {wrmsse:.4f} | MAE: {mae:.4f} | "
                      f"RMSE: {rmse:.4f} | MASE: {mase:.4f} | WAPE: {wape:.4f}")

                row = {
                    "Window":  test_name,
                    "Model":   model_name,
                    "Horizon": slice_name,
                    "n_series": n_series,
                    "n_forecasts": n_forecasts,
                    "WRMSSE": wrmsse,
                    "MAE":    mae,
                    "RMSE":   rmse,
                    "MASE":   mase,
                    "WAPE":   wape,
                    "Inference_Time_Sec":              float(inf_time),
                    "Inference_Time_Per_1k_Forecasts_Sec": float(norm_inf_time),
                }
                if model_name == "TFT Teacher":
                    row["TFT_includes_lightning_overhead"] = True
                if slice_name == "Overall (1-28)":
                    row["hierarchy_level_wrmsses"] = level_wrmsses
                results.append(row)

    return results


# ─── Overall-Stream Evaluator ──────────────────────────────────────────────────

def evaluate_overall_stream(origins, cfg, models_info, training_data,
                             weights_dict, scales_dict, mase_scales_dict, id_meta,
                             ds_dir, args, device, eval_exp_dir=None, suffix=""):
    """
    Evaluate all models separately at each of the 52 seven-day-aligned forecast
    origins in the held-out test stream.

    Metrics are computed per-origin; the summary across origins (mean, SD, median,
    min, max of all five metrics) is returned alongside a per-origin detail DataFrame.

    Parameters
    ----------
    origins     : list of int — first target day of each origin
    models_info : list of (model_name, model_obj, is_tft)

    Returns
    -------
    summary_rows     : list of result dicts for the main CSV (mean values)
    df_per_origin    : pd.DataFrame with per-origin detail rows
    """
    H = cfg.dataset.prediction_window
    L = cfg.dataset.lookback_window

    stores    = resolve_stores(cfg.environment.store_filter)
    max_stores = getattr(cfg.environment, "max_stores", None)
    if max_stores is not None:
        stores = stores[:max_stores]

    model_names = [name for name, _, _ in models_info]
    per_origin_records = []
    
    # Resume logic
    completed_origins = set()
    if eval_exp_dir:
        inc_csv = os.path.join(eval_exp_dir, f"evaluation_results_per_origin{suffix}.csv")
        if os.path.exists(inc_csv):
            try:
                df_existing = pd.read_csv(inc_csv)
                if not df_existing.empty and "Origin" in df_existing.columns:
                    expected_models = {"Seasonal Naive", *{name for name, _, _ in models_info}}
                    expected_horizons = {name for name, _, _ in HORIZON_SLICES}
                    expected_combinations = {
                        (m, h) for m in expected_models for h in expected_horizons
                    }
                    expected_rows = len(expected_combinations)
                    
                    metric_columns = ["WRMSSE", "MAE", "RMSE", "MASE", "WAPE"]
                    for origin, group in df_existing.groupby("Origin"):
                        combinations = set(zip(group["Model"], group["Horizon"]))
                        metrics_valid = (
                            group[metric_columns]
                            .replace([np.inf, -np.inf], np.nan)
                            .notna()
                            .all()
                            .all()
                        )
                        if len(group) == expected_rows and combinations == expected_combinations and metrics_valid:
                            completed_origins.add(origin)
                    
                    if completed_origins:
                        df_valid = df_existing[df_existing["Origin"].isin(completed_origins)]
                        per_origin_records = df_valid.to_dict("records")
                        print(f"Resuming from {len(completed_origins)} fully completed origins.")
            except Exception as e:
                print(f"Failed to load resume state: {e}")

    for o_idx, o in enumerate(origins):
        if o in completed_origins:
            print(f"\n--- Skipping Overall Stream Origin {o_idx+1}/{len(origins)}: "
                  f"d{o}–d{o+H-1} (Already completed) ---")
            continue

        print(f"\n--- Overall Stream Origin {o_idx+1}/{len(origins)}: "
              f"d{o}–d{o+H-1} ---")

        slice_start = o - L       # first history day (90 days before target start)
        slice_end   = o + H - 1   # last target day

        cumul_preds  = {name: [] for name in model_names}
        all_actuals  = []
        all_naive    = []
        all_decoded  = []
        model_times  = {name: 0.0 for name in model_names}
        naive_time   = 0.0

        for store in stores:
            result = _run_store_inference(
                store, ds_dir, training_data, cfg, args, device,
                models_info, o, slice_start, slice_end, H
            )
            if result is None:
                continue
            all_actuals.append(result["actuals"])
            all_naive.append(result["naive"])
            all_decoded.append(result["decoded"].copy())
            naive_time += result["naive_time"]
            for name, p in result["preds"].items():
                cumul_preds[name].append(p)
                model_times[name] += result["model_times"][name]

        # ── Aggregate across stores ────────────────────────────────────────
        actuals         = np.concatenate(all_actuals, axis=0)
        naive_forecasts = np.concatenate(all_naive,   axis=0)
        concat_decoded  = pd.concat(all_decoded, ignore_index=True)
        n_series        = actuals.shape[0]

        assert n_series == FULL_M5_SERIES_COUNT, \
            f"Dataset rule violation: origin {o} has {n_series} series " \
            f"(expected {FULL_M5_SERIES_COUNT}). Final eval must use full population."

        neural_preds = {name: np.concatenate(v, axis=0)
                        for name, v in cumul_preds.items() if v}
        ids = concat_decoded['id'].astype(str).values

        # ── Long-form table ────────────────────────────────────────────────
        df_long = _build_long_form_table(
            ids, actuals, naive_forecasts, neural_preds, o, H
        )
        df_long = df_long.join(id_meta, on='id', how='left')

        _assert_alignment(df_long, n_series * H, list(neural_preds.keys()), id_meta)

        # ── Compute metrics per model per horizon slice ────────────────────
        model_forecast_map = {"Seasonal Naive": "naive",
                              **{name: f"pred_{name}" for name in neural_preds}}

        ids_sorted   = sorted(df_long['id'].unique())
        missing_mase_ids = [sid for sid in ids_sorted if sid not in mase_scales_dict]
        assert not missing_mase_ids, f"Missing MASE scales for {len(missing_mase_ids)} series"
        scales_array = np.array([mase_scales_dict[sid] for sid in ids_sorted])

        for model_name, pred_col in model_forecast_map.items():
            inf_time      = naive_time if model_name == "Seasonal Naive" else model_times[model_name]
            n_forecasts   = n_series
            norm_inf_time = (inf_time / n_forecasts * 1000.0) if n_forecasts > 0 else 0.0

            for slice_name, s_h, e_h in HORIZON_SLICES:
                df_slice = df_long[df_long['h'].between(s_h, e_h - 1)].copy()

                act_pivot  = df_slice.pivot(index='id', columns='h', values='actual').sort_index()
                pred_pivot = df_slice.pivot(index='id', columns='h', values=pred_col).sort_index()
                actuals_slice   = act_pivot.values
                forecasts_slice = pred_pivot.values

                mae, rmse, wape = compute_point_metrics(
                    actuals_slice.flatten(), forecasts_slice.flatten()
                )
                mase = compute_mase(actuals_slice, forecasts_slice, scales_array)

                df_gt_w   = _build_wrmsse_df(df_slice, 'actual',  id_meta)
                df_pred_w = _build_wrmsse_df(df_slice, pred_col,  id_meta)
                wrmsse, _ = compute_hierarchical_wrmsse(
                    df_gt_w, df_pred_w, weights_dict, scales_dict
                )

                per_origin_records.append({
                    "Window":         "Overall Test Stream",
                    "Origin":         o,
                    "Target_Start":   o,
                    "Target_End":     o + H - 1,
                    "n_series":       n_series,
                    "n_forecasts":    n_forecasts,
                    "Model":          model_name,
                    "Horizon":        slice_name,
                    "WRMSSE":         wrmsse,
                    "MAE":            mae,
                    "RMSE":           rmse,
                    "MASE":           mase,
                    "WAPE":           wape,
                    "Operational_Runtime_Sec":               float(inf_time),
                    "Operational_Runtime_Per_1k_Forecasts_Sec": float(norm_inf_time),
                    "TFT_includes_lightning_overhead": (model_name == "TFT Teacher"),
                })
                
        # Incremental save
        if eval_exp_dir:
            inc_csv = os.path.join(eval_exp_dir, f"evaluation_results_per_origin{suffix}.csv")
            pd.DataFrame(per_origin_records).to_csv(inc_csv, index=False)

    # ── Compute summary statistics over all 52 origins ─────────────────────
    df_per_origin = pd.DataFrame(per_origin_records)
    summary_rows  = []

    for model_name in df_per_origin['Model'].unique():
        for slice_name in df_per_origin['Horizon'].unique():
            df_m_h = df_per_origin[
                (df_per_origin['Model']   == model_name) &
                (df_per_origin['Horizon'] == slice_name)
            ]
            row = {
                "Window":    "Overall Test Stream",
                "Model":     model_name,
                "Horizon":   slice_name,
                "n_origins": len(df_m_h),
                "n_series":  int(df_m_h["n_series"].mean()),
                "n_forecasts": int(df_m_h["n_forecasts"].mean()),
                "Inference_Time_Sec":
                    float(df_m_h["Operational_Runtime_Sec"].sum()),
                "Inference_Time_Per_1k_Forecasts_Sec":
                    float(df_m_h["Operational_Runtime_Per_1k_Forecasts_Sec"].mean()),
                "TFT_includes_lightning_overhead": (model_name == "TFT Teacher"),
            }
            for metric in METRICS:
                vals = df_m_h[metric].values
                row[metric]                = float(np.mean(vals))
                row[f"{metric}_SD"]        = float(np.std(vals))
                row[f"{metric}_Median"]    = float(np.median(vals))
                row[f"{metric}_Min"]       = float(np.min(vals))
                row[f"{metric}_Max"]       = float(np.max(vals))
            summary_rows.append(row)

    return summary_rows, df_per_origin


# ─── Controlled Inference Benchmark ───────────────────────────────────────────

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
    sample_batch : dict (batch_x) — fixed batch used for all timed runs
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
        # ── Seasonal Naive: array baseline, no neural forward pass ──────────
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

        # ── Neural models ────────────────────────────────────────────────────
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
              f"{float(np.mean(times)*1000):.3f} ± {float(np.std(times)*1000):.3f} ms")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate M5 Models on ID/OOD splits and overall test stream"
    )
    parser.add_argument("--env",        type=str, default="local")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--exp-name",   type=str, default=None,
                        help="Experiment name (required, e.g. exp_full_phase1)")
    parser.add_argument("--teacher-checkpoint",      type=str, default=None)
    parser.add_argument("--student-nokd-checkpoint", type=str, default=None)
    parser.add_argument("--student-kd-checkpoint",   type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Inference batch size (default: 256)")
    args = parser.parse_args()

    if args.exp_name is None:
        raise ValueError(
            "--exp-name is required. Provide a descriptive name for this run, "
            "e.g. --exp-name exp_full_phase1"
        )

    # ── 1. Load Configuration ─────────────────────────────────────────────
    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    set_seed(cfg.environment.seed)

    # ── 2. Assert Paper Boundary Values ──────────────────────────────────
    print("Asserting configuration boundaries against paper methodology...")
    assert getattr(cfg.environment, "max_stores", None) is None, \
        "Final evaluation cannot use max_stores"
    assert getattr(cfg.environment, "max_batches_per_store", None) is None, \
        "Final evaluation cannot use max_batches_per_store"
    assert cfg.dataset.splits.train.end                   == 1359, "Train end mismatch"
    assert cfg.dataset.splits.test_stream.start           == 1554, "Test stream start mismatch"
    assert cfg.dataset.splits.test_stream.end             == 1941, "Test stream end mismatch"
    assert cfg.dataset.splits.id_test.start               == 1554, "ID start mismatch"
    assert cfg.dataset.splits.id_test.end                 == 1581, "ID end mismatch"
    assert cfg.dataset.splits.ood_test.start              == 1819, "OOD start mismatch"
    assert cfg.dataset.splits.ood_test.end                == 1846, "OOD end mismatch"
    assert cfg.dataset.splits.extended_ood_test.start     == 1914, "Ext-OOD start mismatch"
    assert cfg.dataset.splits.extended_ood_test.end       == 1941, "Ext-OOD end mismatch"
    assert cfg.dataset.lookback_window   == 90, "Lookback window mismatch"
    assert cfg.dataset.prediction_window == 28, "Prediction window mismatch"
    assert cfg.dataset.window_stride     == 7,  "Stride mismatch"
    print("  All configuration assertions passed.")

    # ── 3. Resolve Checkpoints ────────────────────────────────────────────
    outputs_dir = resolve_path(cfg.environment.outputs_dir)
    from utils.paths import get_dataset_dir

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
    teacher_hash = sha256_file(teacher_chk)
    student_nokd_hash = sha256_file(student_nokd_chk)
    student_kd_hash = sha256_file(student_kd_chk)
    print(f"  Teacher:          {teacher_chk} (SHA256: {teacher_hash[:8]})")
    print(f"  Student (No KD):  {student_nokd_chk} (SHA256: {student_nokd_hash[:8]})")
    print(f"  Student (KD):     {student_kd_chk} (SHA256: {student_kd_hash[:8]})")

    # ── 4. Load Data ──────────────────────────────────────────────────────
    ds_dir = get_dataset_dir(cfg)
    df = load_dataset_from_cache(
        artifacts_dir=ds_dir,
        store_filter=cfg.environment.store_filter
    )
    if df is None:
        raise FileNotFoundError(
            f"Preprocessed cache not found for store filter: "
            f"'{cfg.environment.store_filter}'. Run prepare_dataset.py first."
        )

    # ── 5. Build Base Training Dataset ───────────────────────────────────
    print("Building base training dataset...")
    training_data = build_timeseries_dataset(df, cfg, is_train=True)

    # ── 6. Build ID Hierarchy Metadata ───────────────────────────────────
    id_meta_source = df[['id'] + HIERARCHY_COLS].copy()
    id_meta_source['id'] = id_meta_source['id'].astype(str)
    for hcol in HIERARCHY_COLS:
        if hcol in id_meta_source.columns:
            id_meta_source[hcol] = id_meta_source[hcol].astype(str)

    id_meta = (id_meta_source
               .drop_duplicates('id')
               .set_index('id'))
    print(f"  id_meta built: {len(id_meta)} unique series")

    # ── 7. Load Models ────────────────────────────────────────────────────
    print("Loading models from checkpoints...")
    teacher      = TemporalFusionTransformer.load_from_checkpoint(teacher_chk)
    student_nokd = M5TransformerStudent.load_from_checkpoint(
        student_nokd_chk, training_dataset=training_data, strict=True
    )
    student_kd = M5TransformerStudent.load_from_checkpoint(
        student_kd_chk, training_dataset=training_data, strict=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    student_nokd.to(device)
    student_kd.to(device)

    # models_info: list of (model_name, model_obj, is_tft)
    models_info = [
        ("TFT Teacher",       teacher,      True),
        ("Student Without KD", student_nokd, False),
        ("Student With KD",    student_kd,   False),
    ]

    # ── 8. Precompute Scales ──────────────────────────────────────────────
    train_end = cfg.dataset.splits.train.end
    df_train  = df[df['time_idx'] <= train_end].copy()

    weights_dict, scales_dict, scale_diagnostics = compute_wrmsse_weights_and_scales(
        df_train, train_end
    )
    mase_scales_dict = compute_mase_scales(df_train)

    # ── 9. Prepare Output Directory ──────────────────────────────────────
    eval_exp_dir = os.path.join(outputs_dir, "evaluation", args.exp_name)
    os.makedirs(eval_exp_dir, exist_ok=True)

    suffix = f"_{cfg.environment.store_filter}" if cfg.environment.store_filter else "_full"
    
    # ── 9a. Save / Check Run State for Resume Protection ───────────────────
    script_hash = sha256_file(os.path.abspath(__file__))
    current_run_state = {
        "evaluation_script_hash": script_hash,
        "teacher_hash": teacher_hash,
        "student_nokd_hash": student_nokd_hash,
        "student_kd_hash": student_kd_hash,
        "train_end": cfg.dataset.splits.train.end,
        "test_stream_start": cfg.dataset.splits.test_stream.start,
        "test_stream_end": cfg.dataset.splits.test_stream.end,
        "lookback": cfg.dataset.lookback_window,
        "horizon": cfg.dataset.prediction_window,
        "stride": cfg.dataset.window_stride,
        "seed": cfg.environment.seed,
        "expected_series": FULL_M5_SERIES_COUNT
    }
    state_path = os.path.join(eval_exp_dir, f"resume_state{suffix}.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            prev_state = json.load(f)
        if prev_state != current_run_state:
            raise ValueError(
                f"Cannot resume experiment '{args.exp_name}'. Run state configuration or "
                f"checkpoint hashes differ from the previous run.\n"
                f"Please use a new --exp-name or clear the old evaluation folder."
            )
    else:
        with open(state_path, "w") as f:
            json.dump(current_run_state, f, indent=2)

    # Save scale diagnostics for audit
    diag_path = os.path.join(eval_exp_dir, f"wrmsse_scale_diagnostics{suffix}.json")
    with open(diag_path, "w") as f:
        json.dump(scale_diagnostics, f, indent=2)
    print(f"WRMSSE scale diagnostics saved to: {diag_path}")

    # ── 10. Fixed-Window Evaluation ───────────────────────────────────────
    scenarios = [
        ("ID Reference",       cfg.dataset.splits.id_test.start,
                               cfg.dataset.splits.id_test.end),
        ("Event-Intensive OOD", cfg.dataset.splits.ood_test.start,
                                cfg.dataset.splits.ood_test.end),
        ("Extended-Gap OOD",   cfg.dataset.splits.extended_ood_test.start,
                               cfg.dataset.splits.extended_ood_test.end),
    ]

    fixed_results = evaluate_fixed_window(
        scenarios, df, cfg, models_info, training_data,
        weights_dict, scales_dict, mase_scales_dict, id_meta,
        ds_dir, args, device
    )

    # ── 11. Overall Test Stream Evaluation ────────────────────────────────
    H      = cfg.dataset.prediction_window
    stride = cfg.dataset.window_stride
    test_stream_start = cfg.dataset.splits.test_stream.start
    test_stream_end   = cfg.dataset.splits.test_stream.end

    origins = list(range(test_stream_start, test_stream_end - H + 2, stride))
    assert len(origins) == 52, \
        f"Expected 52 overall-stream origins, got {len(origins)}"
    print(f"\nOverall stream: {len(origins)} eligible origins "
          f"(d{origins[0]}–d{origins[-1]}), last target day d{origins[-1]+H-1}.")

    stream_summary, df_per_origin = evaluate_overall_stream(
        origins, cfg, models_info, training_data,
        weights_dict, scales_dict, mase_scales_dict, id_meta,
        ds_dir, args, device, eval_exp_dir, suffix
    )

    # ── 12. Collect Sample Batch for Benchmark ────────────────────────────
    print("\nCollecting sample batch for inference benchmark...")
    sample_batch = None
    stores_list  = resolve_stores(cfg.environment.store_filter)
    max_stores   = getattr(cfg.environment, "max_stores", None)
    if max_stores:
        stores_list = stores_list[:max_stores]

    L_bench = cfg.dataset.lookback_window
    for store in stores_list[:1]:
        df_part = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_part is None:
            continue
        o_ref = origins[0]
        df_sliced = df_part[
            (df_part['time_idx'] >= o_ref - L_bench) &
            (df_part['time_idx'] <= o_ref + H - 1)
        ].copy()
        for col in CAT_COLS:
            if col in df_sliced.columns:
                df_sliced[col] = df_sliced[col].astype(str).astype('category')
        if len(df_sliced) == 0:
            continue
        part_ds     = TimeSeriesDataSet.from_dataset(
            training_data, df_sliced, predict=True, stop_randomization=True
        )
        part_loader = part_ds.to_dataloader(
            train=False, batch_size=args.batch_size, shuffle=False,
            num_workers=cfg.environment.num_workers
        )
        for batch_x, _ in part_loader:
            sample_batch = {k: v for k, v in batch_x.items()}
            break
        del part_loader, part_ds
        break

    # ── 13. Controlled Inference Benchmark ────────────────────────────────
    benchmark_results = []
    if sample_batch is not None:
        print("\n--- Controlled Inference Benchmark ---")
        # Include Seasonal Naive as a named entry (no model object needed)
        benchmark_models_info = [
            ("Seasonal Naive",    None,        False),
        ] + models_info
        benchmark_results = benchmark_inference(
            benchmark_models_info, sample_batch, device,
            num_warmup=3, num_runs=10
        )
    else:
        print("WARNING: Could not collect sample batch; benchmark skipped.")

    # ── 14. Save All Results ──────────────────────────────────────────────
    # 14a. Main evaluation CSV (64 rows: 4 models × 4 windows × 4 slices)
    all_fixed_results = []
    for r in fixed_results:
        # Drop the hierarchy diagnostic from CSV rows
        row = {k: v for k, v in r.items() if k != "hierarchy_level_wrmsses"}
        all_fixed_results.append(row)

    all_stream_results = []
    for r in stream_summary:
        all_stream_results.append(r)

    df_res = pd.DataFrame(all_fixed_results + all_stream_results)
    csv_path = os.path.join(eval_exp_dir, f"evaluation_results{suffix}.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved evaluation results to: {csv_path}")

    # 14b. Per-origin CSV (832 rows: 4 models × 52 origins × 4 slices)
    per_origin_csv = os.path.join(
        eval_exp_dir, f"evaluation_results_per_origin{suffix}.csv"
    )
    df_per_origin.to_csv(per_origin_csv, index=False)
    print(f"Saved per-origin results to: {per_origin_csv}")

    # 14c. Benchmark CSV (4 rows: one per model)
    if benchmark_results:
        bench_csv = os.path.join(eval_exp_dir, f"benchmark_inference{suffix}.csv")
        pd.DataFrame(benchmark_results).to_csv(bench_csv, index=False)
        print(f"Saved inference benchmark to: {bench_csv}")

    # 14d. Config snapshot
    config_save_path = os.path.join(eval_exp_dir, "config.yaml")
    save_config(cfg, config_save_path)
    print(f"Merged config saved to: {config_save_path}")

    # ── 15. Print Relative Degradation ───────────────────────────────────
    print("\n--- Relative ID-to-OOD Performance Degradation (Overall 1-28) ---")
    all_model_names = ["Seasonal Naive", "TFT Teacher",
                       "Student Without KD", "Student With KD"]
    for m in all_model_names:
        df_m = df_res[
            (df_res["Model"] == m) & (df_res["Horizon"] == "Overall (1-28)")
        ]
        if df_m.empty:
            continue
        try:
            id_err       = df_m[df_m["Window"] == "ID Reference"]["WRMSSE"].values[0]
            event_ood_err = df_m[df_m["Window"] == "Event-Intensive OOD"]["WRMSSE"].values[0]
            ext_ood_err  = df_m[df_m["Window"] == "Extended-Gap OOD"]["WRMSSE"].values[0]
            deg_event = ((event_ood_err - id_err) / id_err) * 100
            deg_ext   = ((ext_ood_err   - id_err) / id_err) * 100
            print(f"  {m:26s} -> ID: {id_err:.4f} | "
                  f"Event-OOD: {event_ood_err:.4f} ({deg_event:+.2f}%) | "
                  f"Ext-OOD: {ext_ood_err:.4f} ({deg_ext:+.2f}%)")
        except IndexError:
            print(f"  {m}: insufficient data for degradation calculation")

    # ── 16. Print Deployment Complexity ──────────────────────────────────
    print("\n--- Model Deployment Complexity ---")
    t_params  = sum(p.numel() for p in teacher.parameters())
    t_size_mb = os.path.getsize(teacher_chk) / 1e6
    print(f"  TFT Teacher          -> Parameters: {t_params/1e3:.1f}k | "
          f"Checkpoint: {t_size_mb:.2f} MB")
    s_params  = sum(p.numel() for p in student_nokd.parameters())
    s_size_mb = os.path.getsize(student_nokd_chk) / 1e6
    print(f"  Transformer Student  -> Parameters: {s_params/1e3:.1f}k | "
          f"Checkpoint: {s_size_mb:.2f} MB")
    
    parameter_ratio = t_params / s_params
    parameter_reduction_pct = (1.0 - s_params / t_params) * 100.0
    print(f"  Teacher/Student Parameter Ratio -> {parameter_ratio:.2f}x")
    print(f"  Student Parameter Reduction     -> {parameter_reduction_pct:.1f}%")

    # ── 17. Save Metadata ─────────────────────────────────────────────────
    models_summary = {}
    for m in all_model_names:
        models_summary[m] = {}
        for w in ["ID Reference", "Event-Intensive OOD",
                  "Extended-Gap OOD", "Overall Test Stream"]:
            df_m_w = df_res[
                (df_res["Model"]   == m) &
                (df_res["Window"]  == w) &
                (df_res["Horizon"] == "Overall (1-28)")
            ]
            if df_m_w.empty:
                continue
            models_summary[m][w] = {
                metric: float(df_m_w[metric].values[0])
                for metric in METRICS
                if metric in df_m_w.columns
            }

    save_metadata(
        eval_exp_dir,
        cfg.environment.seed,
        additional_fields={
            "checkpoints": {
                "teacher":     {"path": teacher_chk, "sha256": teacher_hash},
                "student_nokd": {"path": student_nokd_chk, "sha256": student_nokd_hash},
                "student_kd":  {"path": student_kd_chk, "sha256": student_kd_hash},
            },
            "overall_stream_origins": {
                "count":      len(origins),
                "first":      origins[0],
                "last":       origins[-1],
                "last_target_day": origins[-1] + H - 1,
            },
            "metrics_summary": models_summary,
        },
    )
    print(f"\nEvaluation complete. Outputs saved to: {eval_exp_dir}")


if __name__ == "__main__":
    main()

import os
import sys
import glob
import time
import json
import resource
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import load_config
from utils.paths import resolve_path, get_dataset_dir
from data.cache import load_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset
from pytorch_forecasting import TimeSeriesDataSet
from models.teacher import M5TemporalFusionTransformer
from models.student import M5TransformerStudent
from scripts.evaluate_models import (
    compute_wrmsse_weights_and_scales,
    compute_hierarchical_wrmsse,
    compute_mase_scales,
    compute_mase
)

def print_memory(label):
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"[MEMORY] {label}: {rss_kb / 1024 / 1024:.2f} GB")

def format_report_bool(cond):
    if cond is None:
        return "UNRESOLVED"
    return "Yes" if cond else "No"

def smoke_test(model_name, model, val_dl, device):
    print(f"\n--- Smoke Test: {model_name} ---")
    batch = next(iter(val_dl))
    x, y = batch
    x_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in x.items()}
    actual = y[0]
    
    with torch.no_grad():
        out = model(x_dev)
        if "Student" in model_name:
            preds = out
            if preds.ndim == 3:
                preds = preds.squeeze(-1)
        else:
            preds = model.to_prediction(out)
            
    print(f"  Raw output shape: {getattr(out, 'shape', 'Complex output (tuples/dicts)')}")
    print(f"  Final prediction shape: {preds.shape}")
    print(f"  Target shape: {actual.shape}")
    assert preds.shape == actual.shape, f"Shape mismatch: preds {preds.shape} != target {actual.shape}"
    assert torch.isfinite(preds).all(), "Non-finite predictions in smoke test!"
    assert torch.isfinite(actual).all(), "Non-finite actuals in smoke test!"
    
    _decoded = val_dl.dataset.x_to_index(x)
    print(f"  Decoded key sample: Series {_decoded.iloc[0]['id']} Store {_decoded.iloc[0].get('store_id', 'UNKNOWN')}")
    print(f"  Actual (first 5): {actual[0, :5].cpu().numpy()}")
    print(f"  Preds  (first 5): {preds[0, :5].cpu().numpy()}")
    print("---------------------------------")


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting audit on device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = resolve_path(f"artifacts/teacher_student_comparability_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    cfg = load_config(experiment="full")
    ds_dir = get_dataset_dir(cfg)
    
    stores = resolve_stores(getattr(cfg.environment, "store_filter", None))
    print(f"Resolved stores: {stores}")
    print(f"Number of resolved stores: {len(stores)}")
    
    train_end = cfg.dataset.splits.train.end
    val_end = cfg.dataset.splits.validation.end
    L = cfg.dataset.lookback_window
    H = cfg.dataset.prediction_window
    val_start = val_end - H + 1
    min_idx = val_start - L
    
    print(f"Split boundaries: train_end={train_end}, val_start={val_start}, val_end={val_end}")
    assert train_end < val_start, (
        f"Invalid split chronology: train_end={train_end} must be earlier than "
        f"val_start={val_start}. Otherwise the training dataset includes the "
        f"validation forecast period."
    )
    
    print("Building original training schema and target normalizer...")
    train_dfs = []
    for store in stores:
        df_s = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_s is not None:
            train_dfs.append(df_s[df_s["time_idx"] <= train_end])
    df_train_full = pd.concat(train_dfs, ignore_index=True)
    for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday", "month", "year", "event_name_1", "event_type_1"]:
        if col in df_train_full.columns:
            df_train_full[col] = df_train_full[col].astype(str).astype("category")
            
    print_memory("df_train_full")
    
    origin_for_stats = val_start - 1
    df_hist_stats = (
        df_train_full[df_train_full["time_idx"] <= origin_for_stats]
        .groupby("id")
        .agg(
            mean_sales=("sales", "mean"),
            total_sales=("sales", "sum"),
            zero_pct=("sales", lambda x: (x == 0).mean()),
        )
        .reset_index()
    )
    
    train_ds = build_timeseries_dataset(df_train_full, cfg, is_train=True)
    del df_train_full
    print_memory("train_ds")
    
    print("Building validation dataset...")
    val_dfs = []
    for store in stores:
        df_s = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_s is not None:
            val_dfs.append(df_s[(df_s["time_idx"] >= min_idx) & (df_s["time_idx"] <= val_end)])
    df_val_full = pd.concat(val_dfs, ignore_index=True)
    for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday", "month", "year", "event_name_1", "event_type_1"]:
        if col in df_val_full.columns:
            df_val_full[col] = df_val_full[col].astype(str).astype("category")
            
    print(f"Validation series count: {df_val_full['id'].nunique()}")
    print("Validation time range:", df_val_full["time_idx"].min(), df_val_full["time_idx"].max())
    
    assert len(stores) == 10, f"Full audit requires 10 stores, but resolved {len(stores)}: {stores}"
    expected_series_count = 30490
    actual_series_count = df_val_full["id"].nunique()
    assert actual_series_count == expected_series_count, (
        f"Expected {expected_series_count} series, found {actual_series_count}. Resolved stores: {stores}"
    )
        
    print_memory("df_val_full")
            
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val_full, predict=True, stop_randomization=True)
    val_dl = val_ds.to_dataloader(train=False, batch_size=1024, num_workers=4)
    
    student_ckpt = "outputs/student/no_kd/exp_full_phase1/best_student.ckpt"
    teacher_opt_ckpt = "outputs/teacher/tft64_optimized/tft64-opt-epoch=epoch=05-val_loss=val_loss=0.474301.ckpt"
    teacher_hub_ckpt = "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=05-val_loss=val_loss=0.606112.ckpt"
    
    models_to_evaluate = {
        "Student (No KD)": (M5TransformerStudent, student_ckpt),
        "Teacher (Quantile)": (M5TemporalFusionTransformer, teacher_opt_ckpt),
        "Teacher (Huber)": (M5TemporalFusionTransformer, teacher_hub_ckpt)
    }
    
    print("Loading models and performing smoke tests...")
    loaded_models = {}
    for m_name, (MClass, ckpt) in models_to_evaluate.items():
        if os.path.exists(ckpt):
            if "Student" in m_name:
                model = MClass.load_from_checkpoint(ckpt, training_dataset=train_ds, map_location="cpu")
            else:
                model = MClass.load_from_checkpoint(ckpt, map_location="cpu")
            model.to(device)
            model.eval()
            smoke_test(m_name, model, val_dl, device)
            loaded_models[m_name] = model
        else:
            print(f"[MISSING] {ckpt}")
            
    required_models = set(models_to_evaluate)
    loaded_model_names = set(loaded_models)
    missing_models = required_models - loaded_model_names
    if missing_models:
        raise FileNotFoundError(f"Required models were not loaded: {sorted(missing_models)}")
            
    all_preds_dfs = []
    
    for model_name, model in loaded_models.items():
        print(f"\nEvaluating {model_name}...")
        model_start = time.time()
        
        preds_list = []
        actuals_list = []
        series_ids = []
        store_ids = []
        origins = []
        dec_time_idxs = []
        horizons = []
        
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Inference {model_name}"):
                x, y = batch
                x_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in x.items()}
                actual = y[0].numpy()
                
                out = model(x_dev)
                if "Student" in model_name:
                    preds = out
                    if preds.ndim == 3:
                        preds = preds.squeeze(-1)
                    preds = preds.cpu().numpy()
                else:
                    preds = model.to_prediction(out).cpu().numpy()
                    
                _decoded = val_ds.x_to_index(x)
                
                for b_idx in range(len(preds)):
                    series_id = _decoded.iloc[b_idx]["id"]
                    store_id = _decoded.iloc[b_idx].get("store_id", "UNKNOWN")
                    dec_times = x["decoder_time_idx"][b_idx].numpy()
                    origin = dec_times[0] - 1
                    
                    for h_idx in range(len(dec_times)):
                        series_ids.append(series_id)
                        store_ids.append(store_id)
                        origins.append(origin)
                        dec_time_idxs.append(dec_times[h_idx])
                        horizons.append(h_idx + 1)
                        preds_list.append(preds[b_idx, h_idx])
                        actuals_list.append(actual[b_idx, h_idx])
                        
        df_preds = pd.DataFrame({
            "model": model_name,
            "series_id": series_ids,
            "store_id": store_ids,
            "origin": origins,
            "decoder_time_idx_or_date": dec_time_idxs,
            "horizon": horizons,
            "prediction": preds_list,
            "actual": actuals_list
        })
        
        all_preds_dfs.append(df_preds)
        print(f"Time for {model_name}: {time.time() - model_start:.2f}s")
        
    df_all_preds = pd.concat(all_preds_dfs, ignore_index=True)
    print_memory("df_all_preds")
    
    # ------------------------------------------------------------------
    # EXPLICIT ALIGNMENT CHECKS
    # ------------------------------------------------------------------
    print("\nVerifying key-set equality and row counts...")
    model_names = df_all_preds['model'].unique()
    num_series = df_all_preds['series_id'].nunique()
    num_origins = df_all_preds['origin'].nunique()
    
    assert num_origins == 1, (
        f"This evaluator expects one validation origin, found {num_origins}: "
        f"{sorted(df_all_preds['origin'].unique())}"
    )
    expected_rows = num_series * H
    
    key_cols = ["series_id", "origin", "decoder_time_idx_or_date", "horizon"]
    key_sets = {}
    alignment_results = {"expected_rows": expected_rows, "num_series": num_series, "num_origins": num_origins, "models": {}}
    
    for m in model_names:
        df_m = df_all_preds[df_all_preds['model'] == m]
        
        duplicate_count = int(df_m.duplicated(key_cols).sum())
        row_count = len(df_m)
        alignment_results["models"][m] = {"duplicate_keys": duplicate_count, "row_count": row_count}
        assert duplicate_count == 0, f"Duplicate prediction keys for {m}: {duplicate_count}"
        assert row_count == expected_rows, f"Row count mismatch for {m}"
        
        key_sets[m] = set(map(tuple, df_m[key_cols].to_numpy()))
        
    base_keys = key_sets[model_names[0]]
    all_keys_match = True
    for m in model_names[1:]:
        missing = base_keys - key_sets[m]
        extra = key_sets[m] - base_keys
        if missing or extra:
            all_keys_match = False
            alignment_results["models"][m]["missing_keys"] = len(missing)
            alignment_results["models"][m]["extra_keys"] = len(extra)
        else:
            alignment_results["models"][m]["missing_keys"] = 0
            alignment_results["models"][m]["extra_keys"] = 0
            alignment_results["models"][m]["keys_match_base"] = True
            
    assert all_keys_match, "Model prediction key sets do NOT perfectly match!"
    
    actual_pivot = df_all_preds.pivot(index=key_cols, columns="model", values="actual")
    assert not actual_pivot.isna().any().any(), "Missing actual values for one or more aligned model keys"
    
    actual_range = actual_pivot.max(axis=1) - actual_pivot.min(axis=1)
    mismatched_actuals = int((actual_range > 1e-5).sum())
    assert mismatched_actuals == 0, "Models have mismatched actual values!"
    alignment_results["actuals_match"] = True
    
    with open(os.path.join(out_dir, "alignment_checks.json"), "w") as f:
        json.dump(alignment_results, f, indent=4)
        
    df_all_preds.head(1000).to_csv(os.path.join(out_dir, "aligned_predictions_sample.csv"), index=False)
    
    # ------------------------------------------------------------------
    # AUTHORITATIVE METRICS CALCULATION
    # ------------------------------------------------------------------
    print("\nCalculating authoritative WRMSSE and MASE...")
    print("Loading weights and scales...")
    weights_dict, scales_dict = compute_wrmsse_weights_and_scales(ds_dir, train_end)
    mase_scales_dict = compute_mase_scales(ds_dir, train_end)
    
    df_val_gt = df_val_full[(df_val_full["time_idx"] >= val_start) & (df_val_full["time_idx"] <= val_end)].copy()
    
    metrics_records = []
    for m in model_names:
        df_m = df_all_preds[df_all_preds['model'] == m]
        
        pred_for_merge = df_m.rename(
            columns={
                "series_id": "id",
                "decoder_time_idx_or_date": "time_idx",
            }
        )[["id", "time_idx", "prediction"]]
        
        assert pred_for_merge.duplicated(["id", "time_idx"]).sum() == 0, "Duplicates in prediction merge"
        
        df_preds_wrmsse = df_val_gt.merge(
            pred_for_merge,
            on=["id", "time_idx"],
            how="left",
            validate="one_to_one"
        )
        
        assert df_preds_wrmsse["prediction"].notna().all(), "Merge failed: missing predictions"
        assert len(df_preds_wrmsse) == len(df_val_gt), "Merge failed: row count mismatch"
        
        df_preds_wrmsse = df_preds_wrmsse.drop(columns=["sales"], errors="ignore").rename(columns={"prediction": "sales"})
        
        sort_cols = ["id", "time_idx"]
        df_val_metric = df_val_gt.sort_values(sort_cols).reset_index(drop=True)
        df_pred_metric = df_preds_wrmsse.sort_values(sort_cols).reset_index(drop=True)
        assert df_val_metric[sort_cols].equals(df_pred_metric[sort_cols]), "Ground-truth and prediction metric rows are misaligned"
        
        wrmsse = compute_hierarchical_wrmsse(df_val_metric, df_pred_metric, weights_dict, scales_dict)
        mase = compute_mase(df_val_metric, df_pred_metric, mase_scales_dict)
        
        sum_abs_err = np.abs(df_m['prediction'] - df_m['actual']).sum()
        sum_actual = df_m['actual'].sum()
        wape = sum_abs_err / sum_actual
        bias = (df_m['prediction'].sum() - df_m['actual'].sum()) / sum_actual
        mae = np.mean(np.abs(df_m['prediction'] - df_m['actual']))
        rmse = np.sqrt(np.mean((df_m['prediction'] - df_m['actual'])**2))
        
        metrics_records.append({
            "model": m, "WRMSSE": wrmsse, "MAE": mae, "RMSE": rmse, "WAPE": wape, "MASE": mase, "Bias": bias
        })
        
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(os.path.join(out_dir, "common_validation_metrics.csv"), index=False)
    
    # ------------------------------------------------------------------
    # FEATURE LEAKAGE TRACE
    # ------------------------------------------------------------------
    print("\nTracing decoder feature leakage...")
    leakage_records = []
    
    df_hist_stats_sorted = df_hist_stats.sort_values(by="total_sales")
    high_series = df_hist_stats_sorted["id"].iloc[-1]
    med_series = df_hist_stats_sorted["id"].iloc[len(df_hist_stats_sorted)//2]
    intermittent_series = df_hist_stats_sorted[df_hist_stats_sorted["total_sales"] == 0]["id"]
    intermittent_series = intermittent_series.iloc[0] if len(intermittent_series) > 0 else df_hist_stats_sorted["id"].iloc[0]
    
    sample_series = [intermittent_series, med_series, high_series]
    
    target_derived_features = {
        "sales",
        "lag_7",
        "lag_28",
        "rolling_mean_7",
        "rolling_std_7",
        "zero_sales_indicator",
    }

    student_model = loaded_models.get("Student (No KD)")
    if student_model is not None and hasattr(student_model, "known_real_indices"):
        student_decoder_reals = [train_ds.reals[int(i)] for i in student_model.known_real_indices]
        print("Dataset real-variable order:", list(train_ds.reals))
        print("Student known_real_indices:", student_model.known_real_indices)
        print("Student decoder real variables:", student_decoder_reals)
        
        configured_target_features = sorted(target_derived_features.intersection(set(train_ds.reals)))
        missing_expected_names = sorted(target_derived_features - set(train_ds.reals))
        print("Matched target-derived features:", configured_target_features)
        print("Expected names absent from dataset:", missing_expected_names)
    else:
        student_decoder_reals = None
        configured_target_features = []
        
    has_effective_leakage = False
    
    for s_id in sample_series:
        df_s = df_val_full[df_val_full["id"] == s_id].copy().sort_values("time_idx")
        s_val_start = df_s["time_idx"].max() - H + 1
        s_origin = s_val_start - 1
        
        for dec_t in range(s_val_start, s_val_start + H):
            features_to_check = [
                ("lag_7", dec_t - 7),
                ("lag_28", dec_t - 28),
                ("rolling_mean_7", dec_t - 1),
                ("rolling_std_7", dec_t - 1),
                ("zero_sales_indicator", dec_t)
            ]
            for f_name, max_target_idx in features_to_check:
                conceptually_uses_future = max_target_idx > s_origin
                
                if student_decoder_reals is not None:
                    enters_decoder = f_name in student_decoder_reals
                else:
                    enters_decoder = None
                    
                effective_leakage = (enters_decoder is True) and conceptually_uses_future
                if effective_leakage:
                    has_effective_leakage = True
                    
                leakage_records.append({
                    "series_id": s_id, "horizon": dec_t - s_origin, "origin": s_origin,
                    "decoder_time_idx": dec_t, "feature": f_name, 
                    "target_timestamp_used": max_target_idx,
                    "conceptually_uses_future": bool(conceptually_uses_future),
                    "enters_decoder": enters_decoder,
                    "effective_leakage": effective_leakage
                })
            
    df_leakage = pd.DataFrame(leakage_records)
    df_leakage.to_csv(os.path.join(out_dir, "feature_availability.csv"), index=False)
    
    if student_decoder_reals is None:
        leak_status = "UNRESOLVED"
    elif not configured_target_features:
        leak_status = "UNRESOLVED"
    else:
        leak_status = "LEAKED" if has_effective_leakage else "SAFE"
    
    # ------------------------------------------------------------------
    # DEMAND REGIME CALCULATION
    # ------------------------------------------------------------------
    print("\nComputing metrics by demand regime...")
    q33 = df_hist_stats["mean_sales"].quantile(0.33)
    q66 = df_hist_stats["mean_sales"].quantile(0.66)
    
    def classify_regime(row):
        if row["zero_pct"] > 0.8: return "Intermittent"
        elif row["mean_sales"] < q33: return "Low Volume"
        elif row["mean_sales"] < q66: return "Medium Volume"
        else: return "High Volume"
            
    df_hist_stats["regime"] = df_hist_stats.apply(classify_regime, axis=1)
    df_all_preds = df_all_preds.merge(df_hist_stats[["id", "regime"]].rename(columns={"id": "series_id"}), on="series_id", how="left")
    
    regime_metrics = df_all_preds.groupby(["model", "regime"]).apply(
        lambda x: pd.Series({
            "MAE": np.mean(np.abs(x['prediction'] - x['actual'])),
            "RMSE": np.sqrt(np.mean((x['prediction'] - x['actual'])**2))
        })
    ).reset_index()
    regime_metrics.to_csv(os.path.join(out_dir, "metrics_by_demand_regime.csv"), index=False)
    
    # ------------------------------------------------------------------
    # REPORT CLASSIFICATION LOGIC
    # ------------------------------------------------------------------
    if leak_status == "LEAKED":
        final_class = "B. TARGET OR FUTURE-INFORMATION LEAKAGE IN STUDENT"
        final_rec = "B. FIX STUDENT LEAKAGE AND RETRAIN SUPERVISED BASELINE"
    elif leak_status == "UNRESOLVED":
        final_class = "G. DECODER FEATURE AVAILABILITY UNRESOLVED"
        final_rec = "F. RESOLVE STUDENT DECODER FEATURE MAPPING"
    elif not all_keys_match or mismatched_actuals > 0:
        final_class = "E. EVALUATION-PATH OR ALIGNMENT MISMATCH"
        final_rec = "D. REPLACE EARLIER METRICS AFTER FIXING ALIGNMENT"
    else:
        final_class = "A. NO LEAKAGE OR ALIGNMENT DEFECT FOUND IN CHECKED PATHS"
        final_rec = "A. USE COMMON-EVALUATOR RESULTS FOR THE RESEARCH DECISION"

    total_duplicates = sum(result.get("duplicate_keys", 0) for result in alignment_results["models"].values())
    total_missing = sum(result.get("missing_keys", 0) for result in alignment_results["models"].values())
    
    actual_decoder_start = df_all_preds['decoder_time_idx_or_date'].min()
    actual_decoder_end = df_all_preds['decoder_time_idx_or_date'].max()

    conceptual_future_exists = bool(df_leakage["conceptually_uses_future"].any())

    report_content = f"""# Teacher-Student Comparability Audit Report

## 1. Execution Overview
- Device: {device}
- Elapsed Time: {time.time() - start_time:.2f}s
- Configured Validation Boundary: {val_end - H + 1} to {val_end}
- Actual Decoder Start/End: {actual_decoder_start} to {actual_decoder_end}
- Validation Origins: {num_origins}
- Series Count: {num_series}
- Expected Rows per Model: {expected_rows}
- Expected Rows Matched: {format_report_bool(all_keys_match)}

## 2. Leakage and Inputs
- Did target-derived variables logically use post-origin timestamps? {format_report_bool(conceptual_future_exists)}
- Did these leaked features effectively enter the student decoder? {format_report_bool(has_effective_leakage) if leak_status != 'UNRESOLVED' else 'UNRESOLVED'}

## 3. Future Price Assumption
The audit treats future calendar, SNAP, and sell-price variables as known future covariates according to the repository's modelling assumptions. Price-derived variables were not classified as target-derived sales features.

## 4. Final Classifications
**FINAL CLASSIFICATION:** {final_class}
**FINAL RECOMMENDATION:** {final_rec}
"""
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report_content)
        
    print(f"\nAudit complete. Artifacts saved to: {out_dir}")
    print(f"Duplicate Keys: {total_duplicates}")
    print(f"Missing Keys: {total_missing}")
    print(f"Leakage status: {leak_status}")
    print(f"FINAL CLASSIFICATION: {final_class}")
    print(f"FINAL RECOMMENDATION: {final_rec}")

if __name__ == "__main__":
    main()

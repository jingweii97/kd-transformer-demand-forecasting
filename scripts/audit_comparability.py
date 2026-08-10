import os
import sys
import time
import json
import gc
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import load_config
from utils.paths import resolve_path, get_dataset_dir
from data.cache import load_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from models.student import M5TransformerStudent
from scripts.evaluate_models import (
    compute_wrmsse_weights_and_scales,
    compute_hierarchical_wrmsse,
    compute_mase_scales,
    compute_mase,
    get_predictions,
)

try:
    import resource
except ImportError:
    resource = None

def print_memory(label):
    if resource is not None:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"[MEMORY] {label}: {rss_kb / 1024 / 1024:.2f} GB")
    else:
        print(f"[MEMORY] {label}: (cross-platform)")

def format_report_bool(cond):
    if cond is None:
        return "UNRESOLVED"
    return "Yes" if cond else "No"

def smoke_test(model_name, model, sample_dl, device):
    print(f"\n--- Smoke Test: {model_name} ---")
    batch = next(iter(sample_dl))
    x, y = batch
    x_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in x.items()}
    actual = y[0] if isinstance(y, (tuple, list)) else y
    
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
    print("---------------------------------")


def audit_store_target_alignment(df_part, decoded, origin, H, actuals, store):
    audit_positions = [0, len(decoded) // 2, len(decoded) - 1]
    for pos in audit_positions:
        sid = str(decoded.iloc[pos]["id"])
        raw_target = (
            df_part[
                (df_part["id"].astype(str) == sid)
                & (df_part["time_idx"].between(origin, origin + H - 1))
            ]
            .sort_values("time_idx")["sales"]
            .values
        )
        batch_target = actuals[pos]
        assert raw_target.shape == batch_target.shape, (
            f"Alignment audit shape mismatch for store={store}, id={sid}: "
            f"raw={raw_target.shape}, batch={batch_target.shape}"
        )
        assert np.allclose(raw_target, batch_target), (
            f"Alignment audit failed for store={store}, id={sid}, origin={origin}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dicc")
    parser.add_argument("--experiment", type=str, default="full")
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("TYPE", "LABEL", "CHECKPOINT"),
        help=(
            "Model to evaluate. TYPE must be 'student' or 'tft'. May be repeated. "
            "When supplied, replaces the legacy hard-coded comparison list."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="New directory for audit outputs. Refuses to reuse an existing directory.",
    )
    args = parser.parse_args()

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting audit on device: {device}")
    
    if args.output_dir:
        out_dir = resolve_path(args.output_dir)
        if os.path.exists(out_dir):
            raise FileExistsError(
                f"Refusing to reuse existing audit output directory: {out_dir}"
            )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = resolve_path(f"artifacts/teacher_student_comparability_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    ds_dir = get_dataset_dir(cfg)
    
    stores = resolve_stores(getattr(cfg.environment, "store_filter", None))
    print(f"Resolved stores: {stores}")
    print(f"Number of resolved stores: {len(stores)}")
    assert len(stores) == 10, f"Full audit requires 10 stores, but resolved {len(stores)}: {stores}"
    
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
    
    # ------------------------------------------------------------------
    # STORE-BY-STORE STREAMING SETUP FOR MINIMAL RAM USAGE
    # ------------------------------------------------------------------
    print("\n[1/5] Loading training schema and computing historical stats store-by-store...")
    hist_stats_list = []
    origin_for_stats = val_start - 1
    
    for store in stores:
        df_s = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_s is not None:
            df_train_part = df_s[df_s["time_idx"] <= train_end].copy()
            
            # Historical statistics per series for demand regime classification
            h_stats = (
                df_train_part[df_train_part["time_idx"] <= origin_for_stats]
                .groupby("id")
                .agg(
                    mean_sales=("sales", "mean"),
                    total_sales=("sales", "sum"),
                    zero_pct=("sales", lambda x: (x == 0).mean()),
                )
                .reset_index()
            )
            hist_stats_list.append(h_stats)
            del df_train_part
        del df_s
        gc.collect()

    if not hist_stats_list:
        raise RuntimeError("No training partitions were loaded for historical stats.")
    df_hist_stats = pd.concat(hist_stats_list, ignore_index=True)
    df_hist_stats["id"] = df_hist_stats["id"].astype(str)
    del hist_stats_list
    
    # build_timeseries_dataset(..., is_train=True) loads the cached global metadata
    # builder and ignores the dataframe argument, so do not concatenate full train.
    train_ds = build_timeseries_dataset(None, cfg, is_train=True)
    gc.collect()
    print_memory("train_ds created")

    # The default remains the original comparison. --model only changes which
    # serialized models are supplied; all prediction, alignment, and metric code
    # below is shared unchanged.
    if args.model:
        models_to_evaluate = {}
        for model_type, label, checkpoint in args.model:
            if model_type == "student":
                model_class = M5TransformerStudent
            elif model_type == "tft":
                model_class = TemporalFusionTransformer
            else:
                raise ValueError(
                    f"Unsupported --model TYPE '{model_type}'. Use 'student' or 'tft'."
                )
            if label in models_to_evaluate:
                raise ValueError(f"Duplicate --model LABEL: {label}")
            models_to_evaluate[label] = (model_class, checkpoint)
    else:
        models_to_evaluate = {
            "Student (No KD)": (
                M5TransformerStudent,
                "outputs/student/no_kd/exp_full_phase1/best_student.ckpt",
            ),
            "Teacher (Quantile)": (
                TemporalFusionTransformer,
                "outputs/teacher/tft64_optimized/tft64-opt-epoch=epoch=05-val_loss=val_loss=0.474301.ckpt",
            ),
            "Teacher (Huber)": (
                TemporalFusionTransformer,
                "outputs/teacher/tft64_huber/tft64-huber-epoch=epoch=05-val_loss=val_loss=0.606112.ckpt",
            ),
        }
    
    print("\n[2/5] Loading models...")
    loaded_models = {}
    for m_name, (MClass, ckpt) in models_to_evaluate.items():
        if os.path.exists(ckpt):
            if MClass is M5TransformerStudent:
                model = MClass.load_from_checkpoint(ckpt, training_dataset=train_ds, map_location="cpu")
            else:
                model = MClass.load_from_checkpoint(ckpt, map_location="cpu")
            model.to(device)
            model.eval()
            loaded_models[m_name] = model
        else:
            print(f"[MISSING] {ckpt}")
            
    required_models = set(models_to_evaluate)
    loaded_model_names = set(loaded_models)
    missing_models = required_models - loaded_model_names
    if missing_models:
        raise FileNotFoundError(f"Required models were not loaded: {sorted(missing_models)}")

    # Store-by-store prediction streaming loop
    print("\n[3/5] Running store-by-store batch prediction streaming...")
    all_preds_dfs = []
    total_series_count = 0

    for store_idx, store in enumerate(stores, 1):
        print(f"  --> Processing Store {store_idx}/{len(stores)}: {store}")
        df_s = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_s is None:
            continue
            
        df_val_slice = df_s[(df_s["time_idx"] >= min_idx) & (df_s["time_idx"] <= val_end)].copy()
        del df_s
        
        for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday", "month", "year", "event_name_1", "event_type_1"]:
            if col in df_val_slice.columns:
                df_val_slice[col] = df_val_slice[col].astype(str).astype("category")

        store_series_num = df_val_slice["id"].nunique()
        total_series_count += store_series_num

        if len(df_val_slice) == 0:
            continue

        val_ds_store = TimeSeriesDataSet.from_dataset(train_ds, df_val_slice, predict=True, stop_randomization=True)
        decoded_store = val_ds_store.decoded_index
        assert decoded_store["time_idx_first_prediction"].nunique() == 1, (
            f"Multiple prediction starts in store partition: {store}"
        )
        assert int(decoded_store["time_idx_first_prediction"].iloc[0]) == val_start, (
            f"Prediction start mismatch for {store}: expected {val_start}, "
            f"got {decoded_store['time_idx_first_prediction'].iloc[0]}"
        )
        assert decoded_store["id"].is_unique, f"Duplicate series IDs in store partition: {store}"

        val_dl_store = val_ds_store.to_dataloader(
            train=False,
            batch_size=cfg.teacher.batch_size,
            shuffle=False,
            num_workers=getattr(cfg.environment, "num_workers", 0),
        )

        if store_idx == 1:
            for m_name, model in loaded_models.items():
                smoke_test(m_name, model, val_dl_store, device)

        store_actuals = []
        for _, batch_y in val_dl_store:
            target = batch_y[0] if isinstance(batch_y, (tuple, list)) else batch_y
            store_actuals.append(target.cpu().numpy())
        actuals_np = np.concatenate(store_actuals, axis=0)
        assert actuals_np.shape[1] == H, f"Expected H={H} targets for {store}, got {actuals_np.shape[1]}"
        assert np.isfinite(actuals_np).all(), f"Non-finite actuals for store={store}"
        audit_store_target_alignment(df_val_slice, decoded_store, val_start, H, actuals_np, store)

        for model_name, model in loaded_models.items():
            preds_np = get_predictions(model, val_dl_store)
            assert preds_np.shape == actuals_np.shape, (
                f"{model_name} shape mismatch for store={store}: "
                f"predictions={preds_np.shape}, targets={actuals_np.shape}"
            )
            assert np.isfinite(preds_np).all(), (
                f"Non-finite predictions from {model_name} for store={store}"
            )

            preds_list = []
            actuals_list = []
            series_ids = []
            store_ids = []
            origins = []
            dec_time_idxs = []
            horizons = []

            for row_idx, row in decoded_store.reset_index(drop=True).iterrows():
                series_id = row["id"]
                store_id = row.get("store_id", store)
                first_prediction = int(row["time_idx_first_prediction"])
                origin = first_prediction - 1

                for h_idx in range(H):
                    series_ids.append(series_id)
                    store_ids.append(store_id)
                    origins.append(origin)
                    dec_time_idxs.append(first_prediction + h_idx)
                    horizons.append(h_idx + 1)
                    preds_list.append(preds_np[row_idx, h_idx])
                    actuals_list.append(actuals_np[row_idx, h_idx])

            df_preds_store = pd.DataFrame({
                "model": model_name,
                "series_id": series_ids,
                "store_id": store_ids,
                "origin": origins,
                "decoder_time_idx_or_date": dec_time_idxs,
                "horizon": horizons,
                "prediction": preds_list,
                "actual": actuals_list
            })
            all_preds_dfs.append(df_preds_store)

        del df_val_slice, val_ds_store, val_dl_store
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_all_preds = pd.concat(all_preds_dfs, ignore_index=True)
    del all_preds_dfs
    gc.collect()
    print_memory("df_all_preds compiled")

    expected_series_count = 30490
    print(f"Total validation series count across stores: {total_series_count}")
    assert total_series_count == expected_series_count, (
        f"Expected {expected_series_count} series, but found {total_series_count}"
    )

    # ------------------------------------------------------------------
    # EXPLICIT ALIGNMENT CHECKS
    # ------------------------------------------------------------------
    print("\n[4/5] Verifying key-set equality and row counts...")
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
    print("\n[5/5] Calculating authoritative WRMSSE and MASE...")
    print("Loading weights and scales from training data...")

    train_dfs_metrics = []
    for store in stores:
        df_s = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_s is not None:
            train_dfs_metrics.append(df_s[df_s["time_idx"] <= train_end].copy())
        del df_s
    df_train_for_metrics = pd.concat(train_dfs_metrics, ignore_index=True)
    del train_dfs_metrics

    weights_dict, scales_dict, scale_diag = compute_wrmsse_weights_and_scales(df_train_for_metrics, train_end)
    mase_scales_dict = compute_mase_scales(df_train_for_metrics, train_end)
    del df_train_for_metrics
    gc.collect()

    # Load validation ground truth store by store
    val_gt_dfs = []
    for store in stores:
        df_s = load_from_cache(artifacts_dir=ds_dir, store_filter=store)
        if df_s is not None:
            val_gt_dfs.append(df_s[(df_s["time_idx"] >= val_start) & (df_s["time_idx"] <= val_end)].copy())
        del df_s
    df_val_gt = pd.concat(val_gt_dfs, ignore_index=True)
    del val_gt_dfs
    
    df_val_gt["id"] = df_val_gt["id"].astype(str)
    df_val_gt = df_val_gt.sort_values(["id", "time_idx"]).reset_index(drop=True)

    series_ids = df_val_gt["id"].drop_duplicates().to_numpy()
    assert len(series_ids) == expected_series_count, f"Series count mismatch: {len(series_ids)}"
    missing_mase = [sid for sid in series_ids if sid not in mase_scales_dict]
    assert not missing_mase, f"Missing MASE scales for {len(missing_mase)} series"
    scales_array = np.array([mase_scales_dict[sid] for sid in series_ids])

    metrics_records = []
    for m in model_names:
        df_m = df_all_preds[df_all_preds['model'] == m].copy()
        df_m["series_id"] = df_m["series_id"].astype(str)

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

        df_preds_for_wrmsse = df_preds_wrmsse.drop(columns=["sales"], errors="ignore").rename(columns={"prediction": "sales"})

        sort_cols = ["id", "time_idx"]
        df_val_metric = df_val_gt.sort_values(sort_cols).reset_index(drop=True)
        df_pred_metric = df_preds_for_wrmsse.sort_values(sort_cols).reset_index(drop=True)
        assert df_val_metric[sort_cols].equals(df_pred_metric[sort_cols]), "Ground-truth and prediction metric rows are misaligned"

        wrmsse_result, level_wrmsses = compute_hierarchical_wrmsse(
            df_val_metric, df_pred_metric, weights_dict, scales_dict
        )

        actuals_wide = (
            df_val_gt.pivot(index="id", columns="time_idx", values="sales")
            .reindex(series_ids)
            .values
        )
        preds_wide = (
            df_preds_for_wrmsse.pivot(index="id", columns="time_idx", values="sales")
            .reindex(series_ids)
            .values
        )
        assert actuals_wide.shape == (len(series_ids), H), f"actuals_wide shape mismatch: {actuals_wide.shape}"
        assert preds_wide.shape == actuals_wide.shape, f"preds_wide shape mismatch: {preds_wide.shape}"
        assert np.isfinite(actuals_wide).all(), "Non-finite actuals in MASE input"
        assert np.isfinite(preds_wide).all(), "Non-finite predictions in MASE input"

        mase = compute_mase(actuals_wide, preds_wide, scales_array)

        sum_abs_err = np.abs(df_m['prediction'] - df_m['actual']).sum()
        sum_actual = df_m['actual'].sum()
        wape = sum_abs_err / sum_actual
        bias = (df_m['prediction'].sum() - df_m['actual'].sum()) / sum_actual
        mae = np.mean(np.abs(df_m['prediction'] - df_m['actual']))
        rmse = np.sqrt(np.mean((df_m['prediction'] - df_m['actual'])**2))

        metric_record = {
            "model": m, "WRMSSE": float(wrmsse_result), "MAE": float(mae), "RMSE": float(rmse),
            "WAPE": float(wape), "MASE": float(mase), "Bias": float(bias),
            "Actual_Total": float(sum_actual), "Predicted_Total": float(df_m['prediction'].sum()),
        }
        metric_record.update({
            f"WRMSSE_Level_{level}": float(value)
            for level, value in enumerate(level_wrmsses, start=1)
        })
        metrics_records.append(metric_record)

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

    student_model = next(
        (
            loaded_models[name]
            for name, (model_class, _) in models_to_evaluate.items()
            if model_class is M5TransformerStudent and name in loaded_models
        ),
        None,
    )
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
        s_val_start = val_start
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

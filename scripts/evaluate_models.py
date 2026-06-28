import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Add repository root to python path to allow importing packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from utils.config import load_config, save_config, save_metadata
from utils.paths import resolve_path
from utils.seed import set_seed
from data.cache import load_from_cache, load_dataset_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset
from models.student import M5TransformerStudent

# Define official 12 M5 aggregation levels
HIERARCHY_LEVELS = [
    [],                               # Level 1: All products, all stores
    ['state_id'],                     # Level 2: All products, by state
    ['store_id'],                     # Level 3: All products, by store
    ['cat_id'],                       # Level 4: All products, by category
    ['dept_id'],                      # Level 5: All products, by department
    ['state_id', 'cat_id'],           # Level 6: Products by state and category
    ['state_id', 'dept_id'],          # Level 7: Products by state and department
    ['store_id', 'cat_id'],           # Level 8: Products by store and category
    ['store_id', 'dept_id'],          # Level 9: Products by store and department
    ['item_id'],                      # Level 10: Individual product, all stores
    ['item_id', 'state_id'],          # Level 11: Individual product, by state
    ['id']                            # Level 12: Individual product, by store (Level 12 series)
]

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

def compute_wrmsse_weights_and_scales(df_train, train_end):
    """
    Pre-computes M5 hierarchy scales (naive standard deviation) and value weights.
    """
    print("Pre-computing scale factors and value weights for the WRMSSE calculation...")
    df_train = df_train.copy()
    df_train['dollar_value'] = df_train['sales'] * df_train['sell_price']
    
    # Filter for the last 28 days of train to compute weights
    df_weight_window = df_train[df_train['time_idx'] > (train_end - 28)].copy()
    total_dollar_sum = df_weight_window['dollar_value'].sum()
    
    weights_dict = {}
    scales_dict = {}
    
    for level_idx, group_cols in enumerate(HIERARCHY_LEVELS, 1):
        level_name = f"Level_{level_idx}"
        weights_dict[level_name] = {}
        scales_dict[level_name] = {}
        
        # 1. Compute in-sample naive scale factor (RMSSE denominator)
        if len(group_cols) == 0:
            # Level 1: Aggregate sum
            agg_series = df_train.groupby('time_idx')['sales'].sum().sort_index().values
            scale = np.mean(np.diff(agg_series) ** 2)
            scales_dict[level_name]['Total'] = scale if scale > 0 else 1e-4
            
            # Level 1 weight is always 1.0
            weights_dict[level_name]['Total'] = 1.0
        else:
            # Group by level attributes
            df_grouped_train = df_train.groupby(group_cols + ['time_idx'])['sales'].sum().reset_index()
            # Loop groups
            for keys, group in df_grouped_train.groupby(group_cols):
                key_str = "_".join(keys) if isinstance(keys, tuple) else str(keys)
                agg_series = group.sort_values(by='time_idx')['sales'].values
                scale = np.mean(np.diff(agg_series) ** 2)
                scales_dict[level_name][key_str] = scale if scale > 0 else 1e-4
                
            # 2. Compute value weights
            df_grouped_weight = df_weight_window.groupby(group_cols)['dollar_value'].sum().reset_index()
            for _, row in df_grouped_weight.iterrows():
                keys = row[group_cols].values
                key_str = "_".join(keys) if len(group_cols) > 1 else str(keys[0])
                weights_dict[level_name][key_str] = row['dollar_value'] / total_dollar_sum if total_dollar_sum > 0 else 0.0
                
    return weights_dict, scales_dict

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
                scale = level_scales.get(key_str, 1e-4)
                w = level_weights.get(key_str, 0.0)
                
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate M5 Models on ID and OOD splits")
    parser.add_argument("--env", type=str, default="local", help="Environment configuration name")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment configuration name")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name (required — e.g. exp_full_phase1)")
    
    # Model checkpoint paths (optional overrides)
    parser.add_argument("--teacher-checkpoint", type=str, default=None, help="TFT teacher checkpoint path")
    parser.add_argument("--student-nokd-checkpoint", type=str, default=None, help="Student (No KD) checkpoint path")
    parser.add_argument("--student-kd-checkpoint", type=str, default=None, help="Student (With KD) checkpoint path")
    # B-3: Inference batch size is now configurable. Default 256 preserves existing behaviour.
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Inference batch size for model evaluation (default: 256)")
    args = parser.parse_args()

    # B-4: Require an explicit experiment name to avoid accidentally overwriting
    # existing artifacts (e.g. the pre-existing exp_001 evaluation results).
    if args.exp_name is None:
        raise ValueError(
            "--exp-name is required. Provide a descriptive name for this run, "
            "e.g. --exp-name exp_full_phase1"
        )

    # 1. Load Configurations
    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    set_seed(cfg.environment.seed)

    # Determine checkpoint paths, with defaults in outputs_dir
    outputs_dir = resolve_path(cfg.environment.outputs_dir)
    
    from utils.paths import find_checkpoint, get_dataset_dir
 
    teacher_chk = args.teacher_checkpoint or cfg.evaluation.teacher_checkpoint
    if not teacher_chk:
        teacher_chk = find_checkpoint(
            cfg,
            os.path.join(outputs_dir, "teacher", args.exp_name, "best_tft_teacher.ckpt"),
            f"teacher/{args.exp_name}/best_tft_teacher.ckpt"
        )
    else:
        teacher_chk = resolve_path(teacher_chk)
        
    student_nokd_chk = args.student_nokd_checkpoint or cfg.evaluation.student_nokd_checkpoint
    if not student_nokd_chk:
        student_nokd_chk = find_checkpoint(
            cfg,
            os.path.join(outputs_dir, "student", "no_kd", args.exp_name, "best_student.ckpt"),
            f"student/no_kd/{args.exp_name}/best_student.ckpt"
        )
    else:
        student_nokd_chk = resolve_path(student_nokd_chk)
        
    student_kd_chk = args.student_kd_checkpoint or cfg.evaluation.student_kd_checkpoint
    if not student_kd_chk:
        student_kd_chk = find_checkpoint(
            cfg,
            os.path.join(outputs_dir, "student", "kd", args.exp_name, "best_student.ckpt"),
            f"student/kd/{args.exp_name}/best_student.ckpt"
        )
    else:
        student_kd_chk = resolve_path(student_kd_chk)

    # 2. Load Preprocessed Data
    ds_dir = get_dataset_dir(cfg)
    df = load_dataset_from_cache(
        artifacts_dir=ds_dir,
        store_filter=cfg.environment.store_filter
    )
    if df is None:
        raise FileNotFoundError(
            f"Preprocessed cache not found for store filter: '{cfg.environment.store_filter}'. "
            "Please run prepare_dataset.py first."
        )

    # 3. Build Base Datasets and Dataloaders
    print("Building datasets and dataloaders...")
    training_data = build_timeseries_dataset(df, cfg, is_train=True)
    
    id_start = cfg.dataset.splits.id_test.start
    id_end = cfg.dataset.splits.id_test.end
    ood_start = cfg.dataset.splits.ood_test.start
    ood_end = cfg.dataset.splits.ood_test.end
    
    from data.dataset import StorePartitionManager
    partition_manager = StorePartitionManager(training_data, cfg)
    
    # 4. Load Models
    print("Loading models from checkpoints...")
    teacher = TemporalFusionTransformer.load_from_checkpoint(teacher_chk)
    student_nokd = M5TransformerStudent.load_from_checkpoint(student_nokd_chk, training_dataset=training_data, strict=False)
    student_kd = M5TransformerStudent.load_from_checkpoint(student_kd_chk, training_dataset=training_data, strict=False)

    # 5. Precompute WRMSSE weights and scales, and MASE scales
    train_end = cfg.dataset.splits.train.end
    df_train = df[df['time_idx'] <= train_end].copy()
    weights_dict, scales_dict = compute_wrmsse_weights_and_scales(df_train, train_end)
    mase_scales_dict = compute_mase_scales(df_train, train_end)

    results = []

    # Loop ID and OOD windows
    for test_name, start_day, end_day in [
        ("ID Test", id_start, id_end),
        ("OOD Test", ood_start, ood_end)
    ]:
        print(f"\n--- Evaluating Models on {test_name} (Days {start_day} to {end_day}) ---")
        
        # Sliced test actuals and categoricals for indexing
        df_test_gt = df[(df['time_idx'] >= start_day) & (df['time_idx'] <= end_day)].copy()
        df_test_gt = df_test_gt.sort_values(by=['id', 'time_idx']).reset_index(drop=True)
        
        # 5.1 Seasonal Naive Predictions
        print("Generating forecasts from Seasonal Naive...")
        start_t = time.perf_counter()
        df_naive_source = df[(df['time_idx'] >= (start_day - 28)) & (df['time_idx'] < start_day)].copy()
        df_naive_source = df_naive_source.sort_values(by=['id', 'time_idx']).reset_index(drop=True)
        naive_time = time.perf_counter() - start_t
        
        # 5.2 Model Predictions store-by-store
        print("Generating model forecasts store-by-store...")
        import gc
        from data.cache import STORES
        
        # Determine the stores to load
        stores = resolve_stores(cfg.environment.store_filter)
        
        max_stores = getattr(cfg.environment, "max_stores", None)
        if max_stores is not None:
            stores = stores[:max_stores]
            
        max_encoder_length = cfg.dataset.lookback_window
        max_prediction_length = cfg.dataset.prediction_window
        min_idx = end_day - max_encoder_length - max_prediction_length + 1
        
        all_teacher_preds = []
        all_student_nokd_preds = []
        all_student_kd_preds = []
        all_decoded = []
        
        teacher_time = 0.0
        student_nokd_time = 0.0
        student_kd_time = 0.0
        
        for store in stores:
            print(f"Evaluating store: {store}")
            df_part = load_from_cache(
                artifacts_dir=ds_dir,
                store_filter=store
            )
            if df_part is None:
                raise FileNotFoundError(f"Cache not found for store: {store}")
                
            df_part_sliced = df_part[(df_part['time_idx'] >= min_idx) & (df_part['time_idx'] <= end_day)].copy()
            del df_part
            
            cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                        'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
            for col in cat_cols:
                if col in df_part_sliced.columns:
                    df_part_sliced[col] = df_part_sliced[col].astype(str).astype('category')
                    
            if len(df_part_sliced) == 0:
                continue
                
            part_ds = TimeSeriesDataSet.from_dataset(
                training_data,
                df_part_sliced,
                predict=True,
                stop_randomization=True
            )
            del df_part_sliced
            
            part_loader = part_ds.to_dataloader(
                train=False,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=cfg.environment.num_workers
            )
            
            start_t = time.perf_counter()
            part_teacher = get_predictions(teacher, part_loader)
            teacher_time += time.perf_counter() - start_t
            
            start_t = time.perf_counter()
            part_student_nokd = get_predictions(student_nokd, part_loader)
            student_nokd_time += time.perf_counter() - start_t
            
            start_t = time.perf_counter()
            part_student_kd = get_predictions(student_kd, part_loader)
            student_kd_time += time.perf_counter() - start_t
            
            # Limit for debug mode — uses args.batch_size so the limit is
            # consistent with the actual DataLoader batch size above.
            max_batches_per_store = getattr(cfg.environment, "max_batches_per_store", None)
            if max_batches_per_store is not None:
                limit_samples = max_batches_per_store * args.batch_size
                part_teacher = part_teacher[:limit_samples]
                part_student_nokd = part_student_nokd[:limit_samples]
                part_student_kd = part_student_kd[:limit_samples]
                part_decoded = part_ds.decoded_index.head(limit_samples)
            else:
                part_decoded = part_ds.decoded_index
                
            all_teacher_preds.append(part_teacher)
            all_student_nokd_preds.append(part_student_nokd)
            all_student_kd_preds.append(part_student_kd)
            all_decoded.append(part_decoded)
            
            del part_loader
            del part_ds
            gc.collect()
            
        # Aggregate predictions across stores
        teacher_forecasts = np.concatenate(all_teacher_preds, axis=0)
        student_nokd_forecasts = np.concatenate(all_student_nokd_preds, axis=0)
        student_kd_forecasts = np.concatenate(all_student_kd_preds, axis=0)
        concatenated_decoded = pd.concat(all_decoded, ignore_index=True)
        
        # In debug mode (or when limiting stores/batches), filter df_test_gt to match decoded index series
        active_ids = set(concatenated_decoded['id'].unique())
        df_test_gt = df_test_gt[df_test_gt['id'].isin(active_ids)].reset_index(drop=True)
        
        actuals = df_test_gt['sales'].values.reshape(-1, 28)
        num_series = actuals.shape[0]
        
        # Re-get series ids and naive forecasts for the active subset
        series_ids = df_test_gt['id'].drop_duplicates().values
        assert len(series_ids) == num_series, "Mismatch in series count and scales count."
        scales_array = np.array([mase_scales_dict[sid] for sid in series_ids])
        
        df_naive_source = df_naive_source[df_naive_source['id'].isin(active_ids)].reset_index(drop=True)
        naive_forecasts = df_naive_source['sales'].values.reshape(-1, 28)
        
        # Sort predictions alphabetically by id to match the filtered df_test_gt
        concatenated_decoded['pred_idx'] = np.arange(len(concatenated_decoded))
        decoded_sorted = concatenated_decoded.sort_values(by=['id', 'time_idx_first_prediction'])
        target_indices = decoded_sorted['pred_idx'].values
        
        teacher_forecasts = teacher_forecasts[target_indices]
        student_nokd_forecasts = student_nokd_forecasts[target_indices]
        student_kd_forecasts = student_kd_forecasts[target_indices]

        # Shape integrity check
        assert actuals.shape == naive_forecasts.shape == teacher_forecasts.shape == student_nokd_forecasts.shape == student_kd_forecasts.shape
        
        # Evaluate each model
        models_eval = [
            ("Seasonal Naive", naive_forecasts, naive_time),
            ("TFT Teacher", teacher_forecasts, teacher_time),
            ("Student Without KD", student_nokd_forecasts, student_nokd_time),
            ("Student With KD", student_kd_forecasts, student_kd_time)
        ]
        
        slices = [
            ("Overall (1-28)", 0, 28),
            ("Short (1-7)", 0, 7),
            ("Medium (8-14)", 7, 14),
            ("Long (15-28)", 14, 28)
        ]
        
        for name, forecasts, inf_time in models_eval:
            normalized_inf_time = (inf_time / num_series) * 1000.0  # normalized per 1,000 series
            print(f"\n  Model: {name} (Total Inf: {inf_time:.3f}s, Per 1k: {normalized_inf_time:.3f}s)")
            
            for slice_name, start_idx, end_idx in slices:
                # Slice actuals and forecasts
                actuals_slice = actuals[:, start_idx:end_idx]
                forecasts_slice = forecasts[:, start_idx:end_idx]
                
                # Sliced days indices relative to start_day
                slice_start_day = start_day + start_idx
                slice_end_day = start_day + end_idx - 1
                
                # Slice DataFrames for WRMSSE
                df_test_gt_slice = df_test_gt[(df_test_gt['time_idx'] >= slice_start_day) & (df_test_gt['time_idx'] <= slice_end_day)].copy()
                df_preds_slice = df_test_gt_slice.copy()
                df_preds_slice['sales'] = forecasts_slice.flatten()
                
                # Compute metrics
                mae, rmse, wape = compute_point_metrics(actuals_slice.flatten(), forecasts_slice.flatten())
                wrmsse, _ = compute_hierarchical_wrmsse(df_test_gt_slice, df_preds_slice, weights_dict, scales_dict)
                mase = compute_mase(actuals_slice, forecasts_slice, scales_array)
                
                print(f"    {slice_name:15s} -> WRMSSE: {wrmsse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MASE: {mase:.4f} | WAPE: {wape:.4f}")
                
                results.append({
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
            
    # 6. Save results
    eval_exp_dir = os.path.join(outputs_dir, "evaluation", args.exp_name)
    os.makedirs(eval_exp_dir, exist_ok=True)
    
    # Save the fully merged configuration into experiment folder for complete reproducibility
    config_save_path = os.path.join(eval_exp_dir, "config.yaml")
    save_config(cfg, config_save_path)
    print(f"\nMerged config saved to {config_save_path}")

    # Save metrics csv
    suffix = f"_{cfg.environment.store_filter}" if cfg.environment.store_filter else "_full"
    csv_filename = f"evaluation_results{suffix}.csv"
    csv_filepath = os.path.join(eval_exp_dir, csv_filename)
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(csv_filepath, index=False)
    print(f"Saved evaluation metrics to: {csv_filepath}")

    # Save metadata JSON file for traceability
    models = ["Seasonal Naive", "TFT Teacher", "Student Without KD", "Student With KD"]
    summary_metrics = {}
    for m in models:
        summary_metrics[m] = {}
        for w in ["ID Test", "OOD Test"]:
            df_m_w = df_res[(df_res["Model"] == m) & (df_res["Window"] == w) & (df_res["Horizon"] == "Overall (1-28)")]
            summary_metrics[m][w] = {
                "WRMSSE": float(df_m_w["WRMSSE"].values[0]),
                "MASE": float(df_m_w["MASE"].values[0]),
                "MAE": float(df_m_w["MAE"].values[0]),
                "Inference_Time_Sec": float(df_m_w["Inference_Time_Sec"].values[0]),
                "Inference_Time_Per_1k_Sec": float(df_m_w["Inference_Time_Per_1k_Sec"].values[0])
            }
            
    save_metadata(
        eval_exp_dir,
        cfg.environment.seed,
        additional_fields={
            "checkpoints": {
                "teacher": teacher_chk,
                "student_nokd": student_nokd_chk,
                "student_kd": student_kd_chk
            },
            "metrics_summary": summary_metrics
        }
    )

    # 7. Print Relative Degradation
    print("\n--- Relative ID-to-OOD Performance Degradation (Overall 1-28 Horizon) ---")
    for m in models:
        df_m = df_res[(df_res["Model"] == m) & (df_res["Horizon"] == "Overall (1-28)")]
        id_err = df_m[df_m["Window"] == "ID Test"]["WRMSSE"].values[0]
        ood_err = df_m[df_m["Window"] == "OOD Test"]["WRMSSE"].values[0]
        
        degradation = ((ood_err - id_err) / id_err) * 100
        print(f"  {m:25s} -> ID WRMSSE: {id_err:.4f} | OOD WRMSSE: {ood_err:.4f} | Degradation: {degradation:+.2f}%")

    # 8. Print Deployment Complexity
    print("\n--- Model Deployment Efficiency (Complexity) ---")
    t_params = sum(p.numel() for p in teacher.parameters())
    t_size_mb = os.path.getsize(teacher_chk) / 1e6
    print(f"  TFT Teacher        -> Parameters: {t_params/1e3:.1f}k | Saved Checkpoint Size: {t_size_mb:.2f} MB")
    
    s_params = sum(p.numel() for p in student_nokd.parameters())
    s_size_mb = os.path.getsize(student_nokd_chk) / 1e6
    print(f"  Transformer Student -> Parameters: {s_params/1e3:.1f}k | Saved Checkpoint Size: {s_size_mb:.2f} MB")
    print(f"  Parameter Reduction -> {t_params / s_params:.1f}x smaller student")

if __name__ == "__main__":
    main()

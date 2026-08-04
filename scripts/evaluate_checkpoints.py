import os
import sys
import argparse
import time
import hashlib
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from data.cache import load_from_cache, load_dataset_from_cache, resolve_stores
from models.teacher import TemporalFusionTransformer
from scripts.evaluate_models import compute_wrmsse_weights_and_scales, HIERARCHY_LEVELS, _compute_scale

def evaluate_checkpoint(ckpt_path, val_dataloader, weights, scales, cfg):
    print(f"\nEvaluating: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"File not found: {ckpt_path}")
        return None
    
    with open(ckpt_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    ckpt_size = os.path.getsize(ckpt_path)
    
    # Load model to extract details
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load {ckpt_path}: {e}")
        return None
        
    model.eval()
    
    # Get parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Check if Huber or Quantile
    is_quantile = hasattr(model.loss, "quantiles")
    objective = "Quantile" if is_quantile else "Huber"
    
    if is_quantile:
        quantiles = model.loss.quantiles
        q50_idx = quantiles.index(0.5)
    else:
        quantiles = None
        q50_idx = 0
        
    internal_epoch = model.current_epoch
    global_step = model.global_step
    hidden_size = model.hparams.hidden_size
    hidden_cont = model.hparams.hidden_continuous_size
    
    print("Running validation inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Metrics
    total_abs_err = 0.0
    total_sq_err = 0.0
    total_actual = 0.0
    total_pred = 0.0
    total_wrmsse_sq_err = {level: {group: 0.0 for group in scales[level]} for level in scales}
    total_count = 0
    
    quantile_loss_sum = 0.0
    num_batches = 0
    
    val_origins = set()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            x, y = batch
            # Move to device
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            y = (y[0].to(device), y[1].to(device) if y[1] is not None else None)
            
            # Predict
            out = model(x)
            
            if is_quantile:
                # Quantile loss
                q_loss = model.loss.loss(out.prediction, y[0])
                quantile_loss_sum += q_loss.mean().item()
                preds = out.prediction[:, :, q50_idx].cpu().numpy()
            else:
                preds = out.prediction.cpu().numpy()
                if preds.ndim == 3 and preds.shape[-1] == 1:
                    preds = preds.squeeze(-1)
            
            actuals = y[0].cpu().numpy()
            
            num_batches += 1
            
            # For each sample in batch
            for b in range(preds.shape[0]):
                pred_seq = preds[b]
                act_seq = actuals[b]
                
                total_abs_err += np.sum(np.abs(pred_seq - act_seq))
                total_sq_err += np.sum((pred_seq - act_seq)**2)
                total_pred += np.sum(pred_seq)
                total_actual += np.sum(act_seq)
                total_count += len(pred_seq)
                
                # We need full hierarchy IDs to compute WRMSSE correctly.
                # In PyTorch Forecasting, we can map back if needed.
                # However, full WRMSSE needs exact ID matching. 
                # For this streamlined validation evaluation script, we will skip the exact
                # group matching here and log that this is an approximation unless we use
                # the exact evaluator logic from evaluate_models.py (which iterates DataFrames).
                # To perfectly align with user request "Authoritative Evaluator", 
                # we must run the model.predict(val_dataloader) and build a DataFrame.
                
    # We will implement the DataFrame based authoritative evaluation inside the main function.
    return {
        "checkpoint": os.path.basename(ckpt_path),
        "path": ckpt_path,
        "internal_epoch": internal_epoch,
        "global_step": global_step,
        "objective": objective,
        "parameter_count": total_params,
        "checkpoint_size": ckpt_size,
        "checkpoint_SHA256": file_hash,
        "model": model,
        "is_quantile": is_quantile,
        "q50_idx": q50_idx
    }

def authoritative_evaluation(predictions_df, df_val, weights, scales):
    """
    Applies authoritative hierarchy WRMSSE / WAPE / MASE over the predictions.
    """
    print("Running authoritative validation evaluation...")
    
    # Merge predictions with actuals
    df_merged = df_val.merge(predictions_df, on=['id', 'time_idx', 'origin'], how='inner')
    
    if len(df_merged) != len(predictions_df):
        print(f"Warning: Merged length {len(df_merged)} != Predictions length {len(predictions_df)}")
        
    actuals = df_merged['sales'].values
    preds = df_merged['pred_q50'].values
    
    mae = np.mean(np.abs(actuals - preds))
    rmse = np.sqrt(np.mean((actuals - preds)**2))
    
    total_actual = np.sum(actuals)
    total_pred = np.sum(preds)
    
    wape = np.sum(np.abs(actuals - preds)) / (total_actual + 1e-9)
    agg_bias = (total_pred / (total_actual + 1e-9)) - 1
    
    # Calculate WRMSSE / MASE
    wrmsse_total = 0.0
    mase_total = 0.0
    
    origins = df_merged['origin'].unique()
    
    for level_idx, group_cols in enumerate(HIERARCHY_LEVELS, 1):
        level_name = f"Level_{level_idx}"
        level_weights = weights[level_name]
        level_scales = scales[level_name]
        
        if len(group_cols) == 0:
            agg_actual = df_merged.groupby('time_idx')['sales'].sum()
            agg_pred = df_merged.groupby('time_idx')['pred_q50'].sum()
            
            sq_err = np.mean((agg_actual - agg_pred)**2)
            abs_err = np.mean(np.abs(agg_actual - agg_pred))
            scale = level_scales.get('ALL', 1.0)
            weight = level_weights.get('ALL', 1.0)
            
            wrmsse_total += weight * np.sqrt(sq_err / scale)
            mase_total += weight * (abs_err / scale)
        else:
            agg_actual = df_merged.groupby(group_cols + ['time_idx'])['sales'].sum().reset_index()
            agg_pred = df_merged.groupby(group_cols + ['time_idx'])['pred_q50'].sum().reset_index()
            
            for k, group_data in agg_actual.groupby(group_cols):
                if isinstance(k, str) or not isinstance(k, tuple):
                    k_tuple = (k,)
                else:
                    k_tuple = k
                    
                key_str = "_".join(str(x) for x in k_tuple)
                
                scale = level_scales.get(key_str, 1.0)
                weight = level_weights.get(key_str, 1.0)
                
                act_g = group_data['sales'].values
                pred_g_data = agg_pred.copy()
                for c, v in zip(group_cols, k_tuple):
                    pred_g_data = pred_g_data[pred_g_data[c] == v]
                pred_g = pred_g_data['pred_q50'].values
                
                if len(act_g) == len(pred_g) and len(act_g) > 0:
                    sq_err = np.mean((act_g - pred_g)**2)
                    abs_err = np.mean(np.abs(act_g - pred_g))
                    
                    wrmsse_total += weight * np.sqrt(sq_err / scale)
                    mase_total += weight * (abs_err / scale)
    
    wrmsse_final = wrmsse_total / 12.0
    mase_final = mase_total / 12.0
    
    return {
        "validation_MAE": mae,
        "validation_RMSE": rmse,
        "validation_WAPE": wape,
        "validation_seasonal_MASE": mase_final,
        "validation_WRMSSE": wrmsse_final,
        "aggregate_percentage_bias": agg_bias,
        "actual_total": total_actual,
        "predicted_total": total_pred
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
        "outputs/teacher/tft64_optimized/tft64-opt-epoch=epoch=08-val_loss=val_loss=0.474549.ckpt"
    ]
    
    valid_checkpoints = [c for c in checkpoints if os.path.exists(c)]
    
    if not valid_checkpoints:
        print("No valid checkpoints found!")
        return
        
    print("Loading datasets...")
    stores = resolve_stores(getattr(cfg.environment, "store_filter", None))
    df_full = load_from_cache(cfg.dataset.processed_data_path, stores)
    df_train, df_val, df_test = load_dataset_from_cache(cfg.dataset.dataset_cache_path)
    
    train_end = df_train['time_idx'].max()
    weights, scales, diag = compute_wrmsse_weights_and_scales(df_full[df_full['time_idx'] <= train_end], train_end)
    
    from pytorch_forecasting import TimeSeriesDataSet
    train_ds = TimeSeriesDataSet.load(os.path.join(cfg.dataset.dataset_cache_path, "train_dataset_params.pt"))
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val, predict=True, stop_randomization=True)
    val_dl = val_ds.to_dataloader(train=False, batch_size=cfg.teacher.batch_size, num_workers=0)
    
    results = []
    
    for ckpt in valid_checkpoints:
        meta = evaluate_checkpoint(ckpt, val_dl, weights, scales, cfg)
        if not meta:
            continue
            
        model = meta.pop("model")
        
        # Build predictions DataFrame for authoritative evaluation
        predictions = model.predict(val_dl, mode="quantiles" if meta['is_quantile'] else "prediction", return_x=True)
        
        # Unpack predictions
        x_data = predictions.x
        preds = predictions.output
        
        if meta['is_quantile']:
            preds_q50 = preds[:, :, meta['q50_idx']].numpy()
            meta['validation_quantile_loss'] = float(model.loss.loss(preds, predictions.y[0]).mean().item())
        else:
            preds_q50 = preds.squeeze(-1).numpy() if preds.ndim == 3 else preds.numpy()
            meta['validation_quantile_loss'] = None
            
        # Reconstruct prediction dataframe
        dec_len = preds_q50.shape[1]
        
        rows = []
        for b in range(preds_q50.shape[0]):
            sid = val_ds.decoded_index.iloc[b]['id']
            origin = int(x_data['time_idx'][b, 0].item()) - 1
            for h in range(dec_len):
                t_idx = origin + h + 1
                rows.append({
                    "id": sid,
                    "time_idx": t_idx,
                    "origin": origin,
                    "pred_q50": preds_q50[b, h]
                })
                
        df_preds = pd.DataFrame(rows)
        metrics = authoritative_evaluation(df_preds, df_val, weights, scales)
        
        meta.update(metrics)
        meta['teacher_version'] = "Teacher-v1" if "exp_full" in ckpt else "Teacher-v2"
        results.append(meta)
        
    df_results = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"artifacts/teacher_checkpoint_validation_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    columns = [
        "teacher_version", "checkpoint", "internal_epoch", "global_step", "objective", 
        "parameter_count", "checkpoint_size", "validation_quantile_loss", "validation_MAE",
        "validation_RMSE", "validation_seasonal_MASE", "validation_WAPE", "validation_WRMSSE",
        "aggregate_percentage_bias", "actual_total", "predicted_total", "checkpoint_SHA256"
    ]
    df_results = df_results[[c for c in columns if c in df_results.columns]]
    df_results.to_csv(os.path.join(out_dir, "checkpoint_metrics.csv"), index=False)
    
    # Save md
    with open(os.path.join(out_dir, "checkpoint_validation_report.md"), "w") as f:
        f.write("# Checkpoint Validation Report\n\n")
        f.write(df_results.to_markdown(index=False))
        
    print(f"\nSaved results to {out_dir}")

if __name__ == "__main__":
    main()

import os
import sys
import argparse
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Add repository root to python path to allow importing packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, save_config, save_metadata
from utils.paths import resolve_path
from utils.seed import set_seed
from data.cache import load_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset, StorePartitionedDataset
from models.student import M5TransformerStudent
from scripts.evaluate_models import HIERARCHY_LEVELS, compute_wrmsse_weights_and_scales, compute_hierarchical_wrmsse

def get_predictions_and_targets(model, loader):
    """
    Evaluates model on validation loader and returns predictions, targets, and time indices.
    """
    is_tft = isinstance(model, TemporalFusionTransformer)
    if not is_tft:
        model.eval()
        
    all_preds = []
    all_targets = []
    all_times = []
    all_groups = []
    
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            if isinstance(y, (tuple, list)):
                y = y[0]
                
            if is_tft:
                # Move tensors to device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                x_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                out = model(x_device)
                preds = model.to_prediction(out).cpu()
            else:
                if hasattr(model, "device"):
                    for k in x.keys():
                        if isinstance(x[k], torch.Tensor):
                            x[k] = x[k].to(model.device)
                preds = model(x).cpu()
                
            all_preds.append(preds)
            all_targets.append(y.cpu())
            all_times.append(x['decoder_time_idx'][:, 0].cpu())
            all_groups.append(x['groups'][:, 0].cpu())
            
    preds_cat = torch.cat(all_preds, dim=0).numpy()
    targets_cat = torch.cat(all_targets, dim=0).numpy()
    times_cat = torch.cat(all_times, dim=0).numpy()
    groups_cat = torch.cat(all_groups, dim=0).numpy()
    
    return preds_cat, targets_cat, times_cat, groups_cat

def main():
    parser = argparse.ArgumentParser(description="Focused Causal Diagnostic Analysis for Teacher & Student Models")
    parser.add_argument("--env", type=str, default="local", help="Environment configuration name")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment configuration name")
    parser.add_argument("--exp-name", type=str, default="exp_full_phase1", help="Experiment name")
    args = parser.parse_args()

    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    set_seed(cfg.environment.seed)
    
    print("=" * 80)
    print("FOCUSED CAUSAL DIAGNOSTIC ANALYSIS")
    print(f"Experiment Name : {args.exp_name}")
    print(f"Environment     : {args.env}")
    print("Evaluation Split: Validation Partition (d_1360 - d_1553)")
    print("=" * 80)

    val_max_idx = cfg.dataset.splits.validation.end # d_1553
    train_end = cfg.dataset.splits.train.end # d_1359

    # 1. Resolve Checkpoint Paths
    outputs_dir = resolve_path(cfg.environment.outputs_dir)
    teacher_chk = os.path.join(outputs_dir, "teacher", args.exp_name, "best_tft_teacher.ckpt")
    student_nokd_chk = os.path.join(outputs_dir, "student", "no_kd", args.exp_name, "best_student.ckpt")
    student_kd_chk = os.path.join(outputs_dir, "student", "kd", args.exp_name, "best_student.ckpt")
    
    for path, name in [(teacher_chk, "Teacher"), (student_nokd_chk, "Student No-KD"), (student_kd_chk, "Student KD")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required checkpoint for {name} not found at: {path}")

    # 2. Build Base Dataset & Load Models
    print("\n1. Loading Base Dataset & Trained Model Checkpoints...")
    base_ds = build_timeseries_dataset(None, cfg, is_train=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    teacher = TemporalFusionTransformer.load_from_checkpoint(teacher_chk).to(device).eval()
    student_nokd = M5TransformerStudent.load_from_checkpoint(student_nokd_chk, training_dataset=base_ds).to(device).eval()
    student_kd = M5TransformerStudent.load_from_checkpoint(student_kd_chk, training_dataset=base_ds).to(device).eval()

    # 3. Build Validation DataLoader
    print("2. Building Validation DataLoader (d_1360 - d_1553)...")
    val_dataset_iter = StorePartitionedDataset(
        base_dataset=base_ds,
        cfg=cfg,
        batch_size=cfg.student.batch_size,
        is_train=False,
        max_idx=val_max_idx,
        predict=True,
        shuffle=False,
        exp_name=args.exp_name
    )
    val_loader = torch.utils.data.DataLoader(val_dataset_iter, batch_size=None, num_workers=0)

    # 4. Generate Validation Predictions for all 3 models
    print("3. Generating Validation Predictions...")
    print("   - Evaluating TFT Teacher...")
    t_preds, targets, times, groups = get_predictions_and_targets(teacher, val_loader)
    
    print("   - Evaluating Student Without KD...")
    snokd_preds, _, _, _ = get_predictions_and_targets(student_nokd, val_loader)
    
    print("   - Evaluating Student With KD...")
    skd_preds, _, _, _ = get_predictions_and_targets(student_kd, val_loader)

    huber = nn.HuberLoss()

    # 5. Compute Detailed Diagnostics for Each Model
    models = {
        "TFT Teacher": t_preds,
        "Student Without KD": snokd_preds,
        "Student With KD": skd_preds
    }

    total_actual_demand = np.sum(targets)
    horizon = cfg.dataset.prediction_window

    model_metrics = {}
    for name, preds in models.items():
        # 1. Supervised Huber Loss
        p_tensor = torch.tensor(preds, dtype=torch.float32)
        t_tensor = torch.tensor(targets, dtype=torch.float32)
        huber_loss = huber(p_tensor, t_tensor).item()
        
        # 2. MAE & Mean Signed Error
        mae = np.mean(np.abs(preds - targets))
        mean_signed_error = np.mean(preds - targets)
        
        # 3. Total Predicted Demand
        total_pred_demand = np.sum(preds)
        
        # 4. Horizon-wise Aggregate Signed Bias (sum(preds) - sum(actual))
        horizon_signed_bias = np.sum(preds - targets, axis=0)
        
        # 5. Cumulative Aggregate Bias across 28-day horizon
        cumulative_bias = np.sum(horizon_signed_bias)
        
        # 6. Horizon-wise MAE
        horizon_mae = np.mean(np.abs(preds - targets), axis=0)
        
        # 8. Prediction Statistics
        pred_mean = np.mean(preds)
        pred_std = np.std(preds)
        exact_zero_freq = np.mean(preds == 0.0)

        model_metrics[name] = {
            "huber_loss": huber_loss,
            "mae": mae,
            "mean_signed_error": mean_signed_error,
            "total_pred_demand": total_pred_demand,
            "total_actual_demand": total_actual_demand,
            "horizon_signed_bias": horizon_signed_bias,
            "cumulative_bias": cumulative_bias,
            "horizon_mae": horizon_mae,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "exact_zero_freq": exact_zero_freq
        }

    # 6. Pre-compute M5 Hierarchy Scales and Value Weights for Validation WRMSSE
    print("4. Computing WRMSSE across all 12 M5 Hierarchy Levels on Validation Set...")
    # Load training dataset cache to compute scales and weights
    stores = resolve_stores(cfg.environment.store_filter)
    df_train_list = []
    from utils.paths import get_dataset_dir
    ds_dir = get_dataset_dir(cfg)
    for s in stores:
        df_s = load_from_cache(ds_dir, s)
        df_s_sliced = df_s[df_s['time_idx'] <= train_end].copy()
        df_train_list.append(df_s_sliced)
    df_train_all = pd.concat(df_train_list, ignore_index=True)
    
    weights_dict, scales_dict = compute_wrmsse_weights_and_scales(df_train_all, train_end)

    # Build validation ground truth and prediction DataFrames for WRMSSE
    # Map group_id index back to strings for hierarchy grouping
    group_encoder = base_ds._categorical_encoders['id']
    group_names = group_encoder.inverse_transform(groups)

    # Construct DataFrame for Ground Truth
    N, H = targets.shape
    records_gt = []
    records_teacher = []
    records_nokd = []
    records_kd = []
    
    # Fast vectorized creation of evaluation DataFrames
    df_meta = pd.DataFrame({
        'id': group_names,
        'group_code': groups,
        'time_idx_first': times
    })
    
    # Extract item_id, store_id, state_id, cat_id, dept_id mapping from metadata builder
    id_metadata = df_train_all[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates().set_index('id')
    df_meta = df_meta.join(id_metadata, on='id')
    
    # Flatten horizon data for WRMSSE
    rows = []
    for h in range(H):
        df_h = df_meta.copy()
        df_h['time_idx'] = df_h['time_idx_first'] + h
        df_h['sales_gt'] = targets[:, h]
        df_h['sales_teacher'] = t_preds[:, h]
        df_h['sales_nokd'] = snokd_preds[:, h]
        df_h['sales_kd'] = skd_preds[:, h]
        rows.append(df_h)
        
    df_eval = pd.concat(rows, ignore_index=True)

    # Compute 12-level WRMSSE for each model
    df_gt = df_eval[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'time_idx', 'sales_gt']].rename(columns={'sales_gt': 'sales'})
    
    hierarchy_wrmsse = {}
    for name, col in [("TFT Teacher", "sales_teacher"), ("Student Without KD", "sales_nokd"), ("Student With KD", "sales_kd")]:
        df_p = df_eval[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'time_idx', col]].rename(columns={col: 'sales'})
        _, level_wrmsses = compute_hierarchical_wrmsse(df_gt, df_p, weights_dict, scales_dict)
        hierarchy_wrmsse[name] = level_wrmsses

    # 7. Additional Calculations for KD Student
    print("5. Calculating Loss Decomposition for Student With KD...")
    skd_tensor = torch.tensor(skd_preds, dtype=torch.float32)
    t_tensor = torch.tensor(t_preds, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.float32)

    raw_L_sup = huber(skd_tensor, y_tensor).item()
    raw_L_dist = huber(skd_tensor, t_tensor).item()
    
    alpha = getattr(cfg.student, "alpha", 0.5)
    weighted_L_sup = alpha * raw_L_sup
    weighted_L_dist = (1.0 - alpha) * raw_L_dist
    loss_imbalance_ratio = weighted_L_dist / weighted_L_sup if weighted_L_sup > 0 else 0.0

    kd_diagnostics = {
        "alpha": alpha,
        "alpha_convention": "loss = alpha * L_sup + (1 - alpha) * L_dist",
        "raw_L_sup": raw_L_sup,
        "raw_L_dist": raw_L_dist,
        "weighted_L_sup": weighted_L_sup,
        "weighted_L_dist": weighted_L_dist,
        "loss_imbalance_ratio": loss_imbalance_ratio
    }

    # 8. Generate Plots
    print("6. Generating Required Plot Artifacts...")
    fig_dir = os.path.join(outputs_dir, "evaluation", args.exp_name)
    os.makedirs(fig_dir, exist_ok=True)

    # Plot 1: Aggregate Signed Bias by Horizon (1-28)
    plt.figure(figsize=(10, 5))
    horizons_x = np.arange(1, horizon + 1)
    plt.plot(horizons_x, model_metrics["TFT Teacher"]["horizon_signed_bias"], label="TFT Teacher", marker='o', linewidth=2)
    plt.plot(horizons_x, model_metrics["Student Without KD"]["horizon_signed_bias"], label="Student Without KD", marker='s', linewidth=2)
    plt.plot(horizons_x, model_metrics["Student With KD"]["horizon_signed_bias"], label="Student With KD", marker='^', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.title("Aggregate Signed Bias by Forecast Horizon (sum(preds) - sum(actual))", fontsize=12, fontweight='bold')
    plt.xlabel("Forecast Horizon (Day 1 - 28)", fontsize=11)
    plt.ylabel("Aggregate Signed Bias (Units)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    bias_plot_path = os.path.join(fig_dir, "aggregate_bias_by_horizon.png")
    plt.savefig(bias_plot_path, dpi=300)
    plt.close()

    # Plot 2: Hierarchy-Level WRMSSE across 12 levels
    plt.figure(figsize=(12, 6))
    x_levels = np.arange(1, 13)
    bar_width = 0.25
    plt.bar(x_levels - bar_width, hierarchy_wrmsse["TFT Teacher"], width=bar_width, label="TFT Teacher", alpha=0.85)
    plt.bar(x_levels, hierarchy_wrmsse["Student Without KD"], width=bar_width, label="Student Without KD", alpha=0.85)
    plt.bar(x_levels + bar_width, hierarchy_wrmsse["Student With KD"], width=bar_width, label="Student With KD", alpha=0.85)
    plt.title("M5 Hierarchy WRMSSE by Level (Levels 1 to 12)", fontsize=12, fontweight='bold')
    plt.xlabel("M5 Hierarchy Level (1 = Total Sum, 12 = Item-Store Level)", fontsize=11)
    plt.ylabel("WRMSSE", fontsize=11)
    plt.xticks(x_levels, [f"L{i}" for i in range(1, 13)])
    plt.grid(True, linestyle=':', alpha=0.5, axis='y')
    plt.legend(fontsize=10)
    plt.tight_layout()
    wrmsse_plot_path = os.path.join(fig_dir, "hierarchy_wrmsse_by_level.png")
    plt.savefig(wrmsse_plot_path, dpi=300)
    plt.close()

    # Plot 3: Loss Contributions for KD Student
    plt.figure(figsize=(7, 5))
    bars = plt.bar(["Supervised (alpha * L_sup)", "Distillation ((1-alpha) * L_dist)"], 
                   [weighted_L_sup, weighted_L_dist], color=['#2ca02c', '#d62728'], width=0.5)
    plt.title("Weighted Loss Contributions for Student With KD", fontsize=12, fontweight='bold')
    plt.ylabel("Weighted Huber Loss Value", fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5, axis='y')
    plt.tight_layout()
    loss_plot_path = os.path.join(fig_dir, "loss_contributions_kd.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()

    # Plot 4: Actual vs Predicted Aggregate Demand
    plt.figure(figsize=(8, 5))
    model_names_plot = ["Actual Demand", "TFT Teacher", "Student No-KD", "Student KD"]
    demands_plot = [total_actual_demand, 
                    model_metrics["TFT Teacher"]["total_pred_demand"], 
                    model_metrics["Student Without KD"]["total_pred_demand"], 
                    model_metrics["Student With KD"]["total_pred_demand"]]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(model_names_plot, demands_plot, color=colors, width=0.5)
    plt.title("Total Demand Comparison: Actual vs Predicted (Validation Partition)", fontsize=12, fontweight='bold')
    plt.ylabel("Total Demand (Units)", fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (total_actual_demand * 0.01), f"{yval:,.0f}", ha='center', va='bottom', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5, axis='y')
    plt.tight_layout()
    demand_plot_path = os.path.join(fig_dir, "actual_vs_predicted_demand.png")
    plt.savefig(demand_plot_path, dpi=300)
    plt.close()

    # 9. Evaluate Hypotheses A-E strictly based on empirical evidence
    # Evidence evaluation logic:
    # A. Systematic teacher aggregate bias: Is Teacher cumulative bias > 5% of actual demand?
    teacher_bias_pct = (model_metrics["TFT Teacher"]["cumulative_bias"] / total_actual_demand) * 100
    hypothesis_A = teacher_bias_pct > 2.0 or teacher_bias_pct < -2.0
    
    # B. KD Student inheritance/amplification: Is KD student bias matching or amplifying teacher bias?
    kd_bias_pct = (model_metrics["Student With KD"]["cumulative_bias"] / total_actual_demand) * 100
    nokd_bias_pct = (model_metrics["Student Without KD"]["cumulative_bias"] / total_actual_demand) * 100
    hypothesis_B = abs(kd_bias_pct) > abs(nokd_bias_pct) and (np.sign(kd_bias_pct) == np.sign(teacher_bias_pct))
    
    # C. Loss-scale imbalance: Is weighted distillation contribution significantly larger than weighted supervised contribution?
    hypothesis_C = loss_imbalance_ratio > 1.5 or loss_imbalance_ratio < 0.5
    
    # D. Successful teacher imitation but worse ground-truth accuracy: Is L_dist low while L_sup is high?
    hypothesis_D = (raw_L_dist < raw_L_sup) and (model_metrics["Student With KD"]["huber_loss"] > model_metrics["Student Without KD"]["huber_loss"])
    
    # E. Optimisation failure: Are both L_dist and L_sup high?
    hypothesis_E = (raw_L_dist > 0.5) and (raw_L_sup > 0.5)

    # 10. Generate Markdown Diagnostic Report
    report_path = os.path.join(fig_dir, "diagnostic_report.md")
    with open(report_path, "w") as f:
        f.write("# Focused Causal Diagnostic Analysis Report\n\n")
        f.write("## Overview\n")
        f.write(f"- **Experiment Name**: `{args.exp_name}`\n")
        f.write(f"- **Evaluation Partition**: Validation Split (Days 1360 to 1553)\n")
        f.write(f"- **Lookback / Horizon**: L = {cfg.dataset.lookback_window}, H = {cfg.dataset.prediction_window}, Stride = {cfg.dataset.window_stride}\n\n")
        
        f.write("## 1. Per-Model Diagnostics Table\n\n")
        f.write("| Diagnostic Metric | TFT Teacher | Student Without KD | Student With KD |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Supervised Huber Loss** | {model_metrics['TFT Teacher']['huber_loss']:.4f} | {model_metrics['Student Without KD']['huber_loss']:.4f} | {model_metrics['Student With KD']['huber_loss']:.4f} |\n")
        f.write(f"| **MAE** | {model_metrics['TFT Teacher']['mae']:.4f} | {model_metrics['Student Without KD']['mae']:.4f} | {model_metrics['Student With KD']['mae']:.4f} |\n")
        f.write(f"| **Mean Signed Error** | {model_metrics['TFT Teacher']['mean_signed_error']:+.4f} | {model_metrics['Student Without KD']['mean_signed_error']:+.4f} | {model_metrics['Student With KD']['mean_signed_error']:+.4f} |\n")
        f.write(f"| **Total Predicted Demand** | {model_metrics['TFT Teacher']['total_pred_demand']:,.0f} | {model_metrics['Student Without KD']['total_pred_demand']:,.0f} | {model_metrics['Student With KD']['total_pred_demand']:,.0f} |\n")
        f.write(f"| **Actual Ground-Truth Demand** | {total_actual_demand:,.0f} | {total_actual_demand:,.0f} | {total_actual_demand:,.0f} |\n")
        f.write(f"| **Cumulative Horizon Bias** | {model_metrics['TFT Teacher']['cumulative_bias']:+,.0f} | {model_metrics['Student Without KD']['cumulative_bias']:+,.0f} | {model_metrics['Student With KD']['cumulative_bias']:+,.0f} |\n")
        f.write(f"| **Prediction Mean** | {model_metrics['TFT Teacher']['pred_mean']:.4f} | {model_metrics['Student Without KD']['pred_mean']:.4f} | {model_metrics['Student With KD']['pred_mean']:.4f} |\n")
        f.write(f"| **Prediction Std Dev** | {model_metrics['TFT Teacher']['pred_std']:.4f} | {model_metrics['Student Without KD']['pred_std']:.4f} | {model_metrics['Student With KD']['pred_std']:.4f} |\n")
        f.write(f"| **Exact Zero Frequency P(y=0)** | {model_metrics['TFT Teacher']['exact_zero_freq']*100:.2f}% | {model_metrics['Student Without KD']['exact_zero_freq']*100:.2f}% | {model_metrics['Student With KD']['exact_zero_freq']*100:.2f}% |\n\n")

        f.write("## 2. Knowledge Distillation Loss Breakdown (Student With KD)\n\n")
        f.write(f"- **Implementation Alpha Convention**: `L_total = alpha * L_sup + (1 - alpha) * L_dist` (with $\\alpha = {alpha}$)\n")
        f.write(f"- **Raw Supervised Loss ($L_{{sup}}$)**: `{raw_L_sup:.4f}`\n")
        f.write(f"- **Raw Distillation Loss ($L_{{dist}}$)**: `{raw_L_dist:.4f}`\n")
        f.write(f"- **Weighted Supervised Contribution ($\\\\alpha \\\\times L_{{sup}}$)**: `{weighted_L_sup:.4f}`\n")
        f.write(f"- **Weighted Distillation Contribution ($(1 - \\\\alpha) \\\\times L_{{dist}}$)**: `{weighted_L_dist:.4f}`\n")
        f.write(f"- **Loss Imbalance Ratio ($(1-\\\\alpha)L_{{dist}} / \\\\alpha L_{{sup}}$)**: `{loss_imbalance_ratio:.4f}`\n\n")

        f.write("## 3. Hierarchy-Level WRMSSE Breakdown\n\n")
        f.write("| Level | Level Name | TFT Teacher | Student Without KD | Student With KD |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: |\n")
        for i in range(12):
            level_str = f"Level {i+1}"
            f.write(f"| L{i+1} | {level_str} | {hierarchy_wrmsse['TFT Teacher'][i]:.4f} | {hierarchy_wrmsse['Student Without KD'][i]:.4f} | {hierarchy_wrmsse['Student With KD'][i]:.4f} |\n")
        f.write("\n")

        f.write("## 4. Empirical Evaluation of Hypotheses A–E\n\n")
        f.write(f"- **A. Systematic Teacher Aggregate Bias**: **{'SUPPORTED' if hypothesis_A else 'NOT SUPPORTED'}**\n")
        f.write(f"  * Evidence: Teacher cumulative bias is `{model_metrics['TFT Teacher']['cumulative_bias']:+,.0f}` units ({teacher_bias_pct:+.2f}% relative to actual demand).\n\n")
        f.write(f"- **B. KD Student Inheritance or Amplification of Teacher Bias**: **{'SUPPORTED' if hypothesis_B else 'NOT SUPPORTED'}**\n")
        f.write(f"  * Evidence: Student KD cumulative bias is `{model_metrics['Student With KD']['cumulative_bias']:+,.0f}` units ({kd_bias_pct:+.2f}%) compared to Student No-KD (`{model_metrics['Student Without KD']['cumulative_bias']:+,.0f}` units, {nokd_bias_pct:+.2f}%).\n\n")
        f.write(f"- **C. Loss-Scale Imbalance**: **{'SUPPORTED' if hypothesis_C else 'NOT SUPPORTED'}**\n")
        f.write(f"  * Evidence: Ratio of weighted distillation to weighted supervised loss is `{loss_imbalance_ratio:.4f}`.\n\n")
        f.write(f"- **D. Successful Teacher Imitation but Worse Ground-Truth Accuracy**: **{'SUPPORTED' if hypothesis_D else 'NOT SUPPORTED'}**\n")
        f.write(f"  * Evidence: Raw $L_{{dist}}$ (`{raw_L_dist:.4f}`) vs $L_{{sup}}$ (`{raw_L_sup:.4f}`).\n\n")
        f.write(f"- **E. Optimisation Failure to Match Both**: **{'SUPPORTED' if hypothesis_E else 'NOT SUPPORTED'}**\n")
        f.write(f"  * Evidence: Both $L_{{dist}}$ and $L_{{sup}}$ trajectory values.\n\n")

        f.write("## Generated Diagnostic Visualizations\n")
        f.write(f"1. `aggregate_bias_by_horizon.png`\n")
        f.write(f"2. `hierarchy_wrmsse_by_level.png`\n")
        f.write(f"3. `loss_contributions_kd.png`\n")
        f.write(f"4. `actual_vs_predicted_demand.png`\n")

    print(f"\n7. Diagnostic Report and Plots successfully generated in: {fig_dir}")
    print("   - Report: diagnostic_report.md")
    print("=" * 80)

if __name__ == "__main__":
    main()

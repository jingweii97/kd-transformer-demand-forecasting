import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

# Add repository root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.paths import resolve_path
from data.cache import resolve_stores
from data.dataset import build_timeseries_dataset, StorePartitionedDataset
from pytorch_forecasting import TemporalFusionTransformer
from models.student import M5TransformerStudent

def main():
    exp_name = "exp_full_phase1"
    env_name = "local"
    cfg = load_config(env_name=env_name)
    outputs_dir = resolve_path(cfg.environment.outputs_dir)
    
    print("=" * 80)
    print("RUNNING CRITICAL SANITY & ALIGNMENT CHECKS FOR KD DIAGNOSTICS")
    print("=" * 80)

    # 1. Load Checkpoints
    teacher_chk = os.path.join(outputs_dir, "teacher", exp_name, "best_tft_teacher.ckpt")
    student_nokd_chk = os.path.join(outputs_dir, "student", "no_kd", exp_name, "best_student.ckpt")
    student_kd_chk = os.path.join(outputs_dir, "student", "kd", exp_name, "best_student.ckpt")
    
    base_ds = build_timeseries_dataset(None, cfg, is_train=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    teacher = TemporalFusionTransformer.load_from_checkpoint(teacher_chk).to(device).eval()
    student_nokd = M5TransformerStudent.load_from_checkpoint(student_nokd_chk, training_dataset=base_ds).to(device).eval()
    student_kd = M5TransformerStudent.load_from_checkpoint(student_kd_chk, training_dataset=base_ds).to(device).eval()

    # 2. Get Validation Predictions
    val_max_idx = cfg.dataset.splits.validation.end
    val_dataset_iter = StorePartitionedDataset(
        base_dataset=base_ds,
        cfg=cfg,
        batch_size=cfg.student.batch_size,
        is_train=False,
        max_idx=val_max_idx,
        predict=True,
        shuffle=False,
        exp_name=exp_name
    )
    val_loader = torch.utils.data.DataLoader(val_dataset_iter, batch_size=None, num_workers=0)

    def get_preds(model):
        all_preds, all_y = [], []
        is_tft = isinstance(model, TemporalFusionTransformer)
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                if isinstance(y, (tuple, list)): y = y[0]
                if is_tft:
                    x_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                    out = model(x_dev)
                    preds = model.to_prediction(out).cpu()
                else:
                    if hasattr(model, "device"):
                        for k in x.keys():
                            if isinstance(x[k], torch.Tensor): x[k] = x[k].to(model.device)
                    preds = model(x).cpu()
                all_preds.append(preds)
                all_y.append(y.cpu())
        return torch.cat(all_preds, dim=0).numpy(), torch.cat(all_y, dim=0).numpy()

    print("Extracting validation predictions for all models...")
    t_preds, y_true = get_preds(teacher)
    nokd_preds, _ = get_preds(student_nokd)
    kd_preds, _ = get_preds(student_kd)

    huber = nn.HuberLoss()
    t_tensor = torch.tensor(t_preds, dtype=torch.float32)
    nokd_tensor = torch.tensor(nokd_preds, dtype=torch.float32)
    kd_tensor = torch.tensor(kd_preds, dtype=torch.float32)
    y_tensor = torch.tensor(y_true, dtype=torch.float32)
    zero_tensor = torch.zeros_like(y_tensor)

    print("\n--- CHECK 1: Direct KD Student vs Teacher Comparison ---")
    corr = np.corrcoef(kd_preds.flatten(), t_preds.flatten())[0, 1]
    mae_kd_t = np.mean(np.abs(kd_preds - t_preds))
    huber_kd_t = huber(kd_tensor, t_tensor).item()
    mean_signed_diff_kd_t = np.mean(kd_preds - t_preds)
    total_diff_kd_t = np.sum(kd_preds) - np.sum(t_preds)
    
    print(f"  * Pearson Correlation (KD Student vs Teacher) : {corr:.4f}")
    print(f"  * MAE(KD Student, Teacher)                   : {mae_kd_t:.4f}")
    print(f"  * Huber(KD Student, Teacher)                 : {huber_kd_t:.4f}")
    print(f"  * Mean Signed Diff E[KD - Teacher]            : {mean_signed_diff_kd_t:+.4f}")
    print(f"  * Total Demand Diff (KD - Teacher)            : {total_diff_kd_t:+,.0f} units")

    print("\n--- CHECK 2: Reference Baseline Comparisons ---")
    huber_nokd_t = huber(nokd_tensor, t_tensor).item()
    huber_t_y = huber(t_tensor, y_tensor).item()
    huber_nokd_y = huber(nokd_tensor, y_tensor).item()
    huber_kd_y = huber(kd_tensor, y_tensor).item()
    huber_zero_y = huber(zero_tensor, y_tensor).item()

    print(f"  * Huber(Supervised Student No-KD, Teacher)    : {huber_nokd_t:.4f}")
    print(f"  * Huber(Teacher, Ground Truth y)             : {huber_t_y:.4f}")
    print(f"  * Huber(Supervised Student No-KD, GT y)      : {huber_nokd_y:.4f}")
    print(f"  * Huber(KD Student, Ground Truth y)          : {huber_kd_y:.4f}")
    print(f"  * Huber(All-Zero Forecast, Ground Truth y)   : {huber_zero_y:.4f}")

    print("\n--- CHECK 3: Output Scaling & Distribution Audit ---")
    print(f"  * Ground Truth Sales Min / Max / Mean / Std   : {np.min(y_true):.2f} / {np.max(y_true):.2f} / {np.mean(y_true):.4f} / {np.std(y_true):.4f}")
    print(f"  * Teacher Preds Min / Max / Mean / Std        : {np.min(t_preds):.2f} / {np.max(t_preds):.2f} / {np.mean(t_preds):.4f} / {np.std(t_preds):.4f}")
    print(f"  * Student No-KD Min / Max / Mean / Std        : {np.min(nokd_preds):.2f} / {np.max(nokd_preds):.2f} / {np.mean(nokd_preds):.4f} / {np.std(nokd_preds):.4f}")
    print(f"  * Student KD Min / Max / Mean / Std           : {np.min(kd_preds):.2f} / {np.max(kd_preds):.2f} / {np.mean(kd_preds):.4f} / {np.std(kd_preds):.4f}")

    print("\n--- CHECK 4: Training Trajectory Audit from metrics.csv ---")
    metrics_kd_file = os.path.join(outputs_dir, "student", "kd", exp_name, "metrics.csv")
    if os.path.exists(metrics_kd_file):
        df_metrics = pd.read_csv(metrics_kd_file)
        print("  * Student KD Training Epoch Summary:")
        if 'epoch' in df_metrics.columns:
            epochs_df = df_metrics.groupby('epoch').mean(numeric_only=True)
            cols_to_print = [c for c in ['train_loss', 'train_loss_sup', 'train_loss_dist', 'val_loss'] if c in epochs_df.columns]
            print(epochs_df[cols_to_print].tail(10))

    print("\n=================================================================")
    print("SANITY CHECKS COMPLETE")
    print("=================================================================")

if __name__ == "__main__":
    main()

import os
import sys
import argparse
import pandas as pd

def plot_curves(exp_dir=None, metrics_path=None):
    """
    Plots training and validation loss curves from PyTorch Lightning CSVLogger output.
    Saves 'training_curves.png' in the experiment output folder.
    """
    if metrics_path is None:
        if exp_dir is None:
            raise ValueError("Must provide either --exp-dir or --metrics-path")
        metrics_path = os.path.join(exp_dir, "metrics.csv")

    if not os.path.exists(metrics_path):
        print(f"[Plotting Warning] metrics.csv not found at: {metrics_path}")
        return False

    df = pd.read_csv(metrics_path)
    if df.empty or "epoch" not in df.columns:
        print("[Plotting Warning] Empty or invalid metrics.csv file.")
        return False

    # Aggregate metrics by epoch
    epoch_df = df.groupby("epoch").mean(numeric_only=True).reset_index()
    
    if "val_loss" not in epoch_df.columns:
        print("[Plotting Warning] 'val_loss' not present in metrics.csv.")
        return False

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    
    # Plot train loss if available
    train_col = None
    for col in ["train_loss_epoch", "train_loss"]:
        if col in epoch_df.columns and not epoch_df[col].dropna().empty:
            train_col = col
            break

    if train_col:
        plt.plot(epoch_df["epoch"] + 1, epoch_df[train_col], label="Train Loss", color="#1f77b4", linewidth=2)

    plt.plot(epoch_df["epoch"] + 1, epoch_df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")

    # Highlight best epoch
    best_idx = epoch_df["val_loss"].idxmin()
    best_epoch = epoch_df.loc[best_idx, "epoch"] + 1
    best_val_loss = epoch_df.loc[best_idx, "val_loss"]

    plt.scatter([best_epoch], [best_val_loss], color="red", s=80, zorder=5, label=f"Best (Epoch {int(best_epoch)}: {best_val_loss:.4f})")
    
    plt.title("Training & Validation Loss Curves", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    out_dir = os.path.dirname(metrics_path)
    out_file = os.path.join(out_dir, "training_curves.png")
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"Training curves plot saved to: {out_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Plot Training & Validation Curves from metrics.csv")
    parser.add_argument("--exp-dir", type=str, default=None, help="Experiment directory containing metrics.csv")
    parser.add_argument("--metrics-path", type=str, default=None, help="Direct path to metrics.csv")
    args = parser.parse_args()

    plot_curves(exp_dir=args.exp_dir, metrics_path=args.metrics_path)

if __name__ == "__main__":
    main()

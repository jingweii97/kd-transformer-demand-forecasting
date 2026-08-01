import os
import sys
import json
import torch
import argparse
from pytorch_forecasting import TemporalFusionTransformer

def main():
    parser = argparse.ArgumentParser(description="Verify TFT training outputs")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name")
    parser.add_argument("--hidden-size", type=int, required=True, help="Expected hidden_size")
    args = parser.parse_args()

    exp_dir = os.path.join("outputs", "teacher", args.exp_name)
    
    # 1. Verify output directory exists
    if not os.path.isdir(exp_dir):
        print(f"ERROR: Output directory {exp_dir} does not exist.")
        sys.exit(1)

    # 3. Effective configuration exists
    config_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.isfile(config_path):
        print(f"ERROR: Configuration file {config_path} missing.")
        sys.exit(1)

    # 4. Metadata exists
    metadata_path = os.path.join(exp_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        print(f"ERROR: Metadata file {metadata_path} missing.")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # 2. Metrics file exists (Assuming CSV logger is used)
    metrics_dir = os.path.join("outputs", "teacher", args.exp_name, "metrics")
    if not os.path.isdir(metrics_dir):
        metrics_dir = exp_dir # sometimes it's saved in exp_dir
        
    metrics_files = []
    for root, dirs, files in os.walk(exp_dir):
        if "metrics.csv" in files:
            metrics_files.append(os.path.join(root, "metrics.csv"))
            
    if not metrics_files:
        print("ERROR: Metrics file (metrics.csv) missing.")
        sys.exit(1)

    # 6. At least one checkpoint exists
    checkpoints = [f for f in os.listdir(exp_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        print("ERROR: No checkpoints found.")
        sys.exit(1)

    # 13. Best checkpoint hash is recorded
    if "checkpoint_hashes" not in metadata or not metadata["checkpoint_hashes"]:
        print("ERROR: Checkpoint hashes missing from metadata.")
        sys.exit(1)

    best_ckpt = metadata.get("best_checkpoint_path")
    if not best_ckpt or not os.path.isfile(best_ckpt):
        print("ERROR: Best checkpoint path missing or invalid.")
        sys.exit(1)

    # 7. Best checkpoint loads on CPU
    print(f"Loading checkpoint {best_ckpt} on CPU...")
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt, map_location="cpu")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        sys.exit(1)

    # 8. Checkpoint belongs to the correct experiment
    if metadata.get("experiment_identifier") != args.exp_name:
        print(f"ERROR: Metadata experiment_identifier does not match {args.exp_name}")
        sys.exit(1)

    # 9. & 10. Checkpoint hidden_size & hidden_continuous_size
    h_size = model.hparams.hidden_size
    if h_size != args.hidden_size:
        print(f"ERROR: Expected hidden_size {args.hidden_size}, got {h_size}")
        sys.exit(1)
        
    hc_size = getattr(model.hparams, "hidden_continuous_size", None)
    if hc_size is None and hasattr(model.hparams, "get"):
        hc_size = model.hparams.get("hidden_continuous_size", 8)
    if hc_size != 8:
        print(f"ERROR: Expected hidden_continuous_size 8, got {hc_size}")
        sys.exit(1)


    # 11. Checkpoint epoch and global step
    try:
        ckpt_data = torch.load(best_ckpt, map_location="cpu")
        epoch = ckpt_data.get("epoch")
        global_step = ckpt_data.get("global_step")
        if epoch is None or global_step is None:
            print("ERROR: Epoch or global_step not available in checkpoint.")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read epoch/global_step: {e}")
        sys.exit(1)

    # 12. Best validation score is finite
    best_score = metadata.get("best_validation_loss")
    if best_score is None or not (isinstance(best_score, (float, int)) and best_score == best_score and best_score != float('inf') and best_score != float('-inf')):
        print("ERROR: Best validation score is not finite.")
        sys.exit(1)

    # 14. The run did not resume from another experiment
    if "resumed_from" in metadata and metadata["resumed_from"]:
        print("ERROR: Run resumed from another experiment.")
        sys.exit(1)

    print("SUCCESS: Post-training verification passed.")

if __name__ == "__main__":
    main()

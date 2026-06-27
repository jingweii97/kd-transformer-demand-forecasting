import os
import sys
import argparse
import torch
import numpy as np

# Add repository root to python path to allow importing packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.config import load_config
from utils.paths import resolve_path
from utils.seed import set_seed
from data.cache import load_from_cache
from data.dataset import build_timeseries_dataset
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

def main():
    parser = argparse.ArgumentParser(description="Verify alignment of precomputed soft targets with teacher predictions.")
    parser.add_argument("--env", type=str, default="local", help="Environment configuration name")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the trained TFT teacher checkpoint")
    parser.add_argument("--soft-targets-path", type=str, default=None, help="Path to soft targets tensor file")
    parser.add_argument("--max-day", type=int, default=None, help="Limit random verification day range (default: end of training)")
    args = parser.parse_args()

    # Load configurations
    cfg = load_config(env_name=args.env)
    set_seed(cfg.environment.seed)

    # 1. Load preprocessed Parquet dataset
    from utils.paths import get_dataset_dir, get_experiment_dir
    ds_dir = get_dataset_dir(cfg)
    df = load_from_cache(
        artifacts_dir=ds_dir,
        store_filter=cfg.environment.store_filter
    )
    if df is None:
        raise FileNotFoundError(
            f"Preprocessed cache not found for store filter: {cfg.environment.store_filter}. "
            "Please run prepare_dataset.py first."
        )

    # 2. Build training base dataset (for encoder mappings)
    print("Building base training dataset...")
    training_data = build_timeseries_dataset(df, cfg, is_train=True)

    # 3. Load soft targets
    soft_targets_path = args.soft_targets_path
    if not soft_targets_path:
        exp_dir = getattr(cfg.environment, "experiment_artifacts_dir", None)
        if exp_dir is not None:
            exp_art_dir = get_experiment_dir(cfg)
            path1 = os.path.join(exp_art_dir, "soft_targets", "exp_001.pt")
            path2 = os.path.join(exp_art_dir, "outputs", "soft_targets", "exp_001.pt")
            if os.path.exists(path1):
                soft_targets_path = path1
            elif os.path.exists(path2):
                soft_targets_path = path2
            else:
                raise FileNotFoundError(
                    f"Soft targets file not found under configured experiment_artifacts_dir at '{exp_art_dir}'"
                )
        else:
            artifacts_dir = resolve_path(cfg.environment.artifacts_dir)
            soft_targets_path = os.path.join(artifacts_dir, "soft_targets", "exp_001.pt")
    
    soft_targets_path_abs = resolve_path(soft_targets_path)
    print(f"Loading precomputed soft targets from: {soft_targets_path_abs}")
    if not os.path.exists(soft_targets_path_abs):
        raise FileNotFoundError(f"Soft targets file not found at: {soft_targets_path_abs}")
    soft_targets = torch.load(soft_targets_path_abs)

    # 4. Load Teacher model checkpoint
    checkpoint_path_abs = resolve_path(args.checkpoint_path)
    print(f"Loading TFT teacher checkpoint from: {checkpoint_path_abs}")
    teacher = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path_abs)
    teacher.eval()

    # 5. Engineering Validation alignment checks
    print("\n--- Starting Alignment Validation ---")
    
    # We sample 5 random series and starting times within the training window
    group_encoder = training_data._categorical_encoders['id']
    present_ids = set(df['id'].unique())
    unique_ids = [uid for uid in group_encoder.classes_ if uid in present_ids]
    
    # Determine valid start day range (from L=90 to train_end - H=28 or max_day override)
    train_end = cfg.dataset.splits.train.end
    min_time = cfg.dataset.lookback_window + 1  # 91
    if args.max_day is not None:
        max_time = args.max_day - cfg.dataset.prediction_window + 1
    else:
        max_time = train_end - cfg.dataset.prediction_window + 1
    
    # Set seed for random selection reproducibility
    np.random.seed(12345)
    
    num_samples = 5
    mismatches = 0

    for sample_idx in range(num_samples):
        # Select random series and random window start day
        group_name = np.random.choice(unique_ids)
        group_id = group_encoder.transform([group_name])[0]
        start_time = np.random.randint(min_time, max_time)
        
        print(f"Sample {sample_idx + 1}/{num_samples}: Series ID: '{group_name}' (encoded: {group_id}), First Predict Step: Day {start_time}")

        # Slice data for this group up to the end of prediction horizon (start_time + 27)
        # to construct the input sequence
        df_sample = df[(df['id'] == group_name) & (df['time_idx'] <= start_time + 27)].copy()
        
        # Build evaluation dataset for a single predict sequence
        sample_ds = TimeSeriesDataSet.from_dataset(
            training_data,
            df_sample,
            predict=True,
            stop_randomization=True
        )
        
        loader = sample_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
        
        # Run teacher predict
        with torch.no_grad():
            teacher_pred = teacher.predict(loader, mode="prediction")[0] # (28,)
            
        # Get corresponding precomputed soft targets
        precomputed_target = soft_targets[group_id, start_time] # (28,)
        
        # Check alignment (difference should be zero or negligible due to numeric float precision)
        diff = torch.abs(teacher_pred - precomputed_target).max().item()
        
        if diff < 1e-4:
            print(f"  Result: ALIGNMENT PASSED (Max absolute difference: {diff:.6f})")
        else:
            print(f"  Result: ALIGNMENT FAILED! (Max absolute difference: {diff:.6f})")
            print(f"    Teacher prediction: {teacher_pred}")
            print(f"    Precomputed target: {precomputed_target}")
            mismatches += 1

    print("\n--- Alignment Check Summary ---")
    if mismatches == 0:
        print("SUCCESS: All sampled soft targets are perfectly aligned with teacher forecasts!")
        sys.exit(0)
    else:
        print(f"FAILURE: Found {mismatches} mismatches during verification.")
        sys.exit(1)

if __name__ == "__main__":
    main()

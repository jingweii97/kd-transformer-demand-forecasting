import os
import sys
import argparse
import torch

# Add repository root to python path to allow importing packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.paths import resolve_path
from utils.seed import set_seed
from data.cache import load_from_cache, load_all_from_cache
from data.dataset import build_timeseries_dataset
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

def main():
    parser = argparse.ArgumentParser(description="Generate and Save TFT Teacher Forecasts as Soft Targets")
    parser.add_argument("--env", type=str, default="local", help="Environment configuration name")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the trained TFT teacher checkpoint")
    parser.add_argument("--exp-name", type=str, default="exp_001", help="Experiment name")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--max-day", type=int, default=None, 
                        help="Limit inference day range for fast verification (default: end of Validation)")
    args = parser.parse_args()

    # Load Configurations
    cfg = load_config(env_name=args.env)
    set_seed(cfg.environment.seed)

    # Determine default max day for soft target generation
    max_day = args.max_day if args.max_day is not None else cfg.dataset.splits.validation.end

    # Define output file path under artifacts/soft_targets/
    artifacts_dir = resolve_path(cfg.environment.artifacts_dir)
    output_dir = os.path.join(artifacts_dir, "soft_targets")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.exp_name}.pt")

    # 1. Load Preprocessed Data
    # Single store (local dev / Phase-1): use store_filter directly.
    # Full dataset (Phase-2, store_filter empty): concatenate all per-store Parquet files.
    if cfg.environment.store_filter:
        df = load_from_cache(
            artifacts_dir=cfg.environment.artifacts_dir,
            store_filter=cfg.environment.store_filter
        )
    else:
        df = load_all_from_cache(artifacts_dir=cfg.environment.artifacts_dir)
    if df is None:
        raise FileNotFoundError(
            f"Preprocessed cache not found for store filter: '{cfg.environment.store_filter}'. "
            "Please run prepare_dataset.py first."
        )

    # 2. Build Base Dataset (to inherit encoders and normalizers)
    print("Building base training dataset...")
    training_data = build_timeseries_dataset(df, cfg, is_train=True)

    # 3. Load Frozen TFT Model
    checkpoint_path_abs = resolve_path(args.checkpoint_path)
    print(f"Loading TFT teacher model from checkpoint: {checkpoint_path_abs}")
    teacher = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path_abs)
    teacher.eval()  # Set to evaluation mode

    # 4. Build Inference Dataloader via Partition Manager
    from data.dataset import StorePartitionManager
    partition_manager = StorePartitionManager(training_data, cfg)
    inference_loader = partition_manager.test_dataloader(
        batch_size=args.batch_size,
        max_idx=max_day,
        predict=False  # sliding windows
    )

    # 5. Generate Point Forecasts (Median/0.5 Quantile predictions)
    print("Generating teacher forecasts over all sliding windows...")
    with torch.no_grad():
        preds = teacher.predict(inference_loader, mode="prediction")
    
    print(f"Generated predictions tensor shape: {preds.shape}")

    # 6. Map Predictions to Group Codes and Start Times Vectorially
    print("Mapping and organizing predictions into a 3D lookup tensor...")
    group_encoder = training_data._categorical_encoders['id']
    decoded_index = partition_manager.get_decoded_index()
    group_names = decoded_index['id'].values
    group_codes = group_encoder.transform(group_names)
    start_times = decoded_index['time_idx_first_prediction'].values

    # Check mapping alignment
    assert len(preds) == len(group_codes) == len(start_times), "Mismatch in prediction shapes and index lengths."

    # Allocate tensor: (num_groups, max_days, forecast_horizon)
    num_groups = len(group_encoder.classes_)
    max_days = cfg.dataset.splits.ood_test.end + 1
    forecast_horizon = cfg.dataset.prediction_window
    
    soft_targets = torch.zeros((num_groups, max_days, forecast_horizon), dtype=torch.float32)

    # Vectorized assignment
    group_codes_tensor = torch.tensor(group_codes, dtype=torch.long)
    start_times_tensor = torch.tensor(start_times, dtype=torch.long)
    
    soft_targets[group_codes_tensor, start_times_tensor] = preds.cpu()

    # 8. Save soft targets
    print(f"Saving soft targets lookup tensor to: {output_file}")
    torch.save(soft_targets, output_file)
    print("Soft targets generation completed successfully!")

if __name__ == "__main__":
    main()

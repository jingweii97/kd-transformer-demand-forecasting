import os
import sys
import json
import argparse
import datetime
import torch
import numpy as np

# Add repository root to python path to allow importing packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, get_git_commit_hash
from utils.paths import resolve_path
from utils.seed import set_seed
from data.cache import load_from_cache, load_all_from_cache, FEATURE_VERSION
from data.dataset import build_timeseries_dataset
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

def main():
    parser = argparse.ArgumentParser(description="Generate and Save TFT Teacher Forecasts as Soft Targets")
    parser.add_argument("--env", type=str, default="local", help="Environment configuration name")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the trained TFT teacher checkpoint")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name (required — e.g. exp_full_phase1)")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--max-day", type=int, default=None, 
                        help="Limit inference day range for fast verification (default: end of Validation)")
    args = parser.parse_args()

    # B-4: Require an explicit experiment name.
    if args.exp_name is None:
        raise ValueError(
            "--exp-name is required. Provide a descriptive name for this run, "
            "e.g. --exp-name exp_full_phase1"
        )

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
    from utils.paths import get_dataset_dir
    ds_dir = get_dataset_dir(cfg)
    if cfg.environment.store_filter:
        df = load_from_cache(
            artifacts_dir=ds_dir,
            store_filter=cfg.environment.store_filter
        )
    else:
        df = load_all_from_cache(artifacts_dir=ds_dir)
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

    import gc
    from data.cache import STORES
    
    # 4. Loop over store partitions and generate predictions
    print(f"Generating teacher forecasts store-by-store up to Day {max_day}...")
    
    # Determine the stores to load
    store_filter = cfg.environment.store_filter
    stores = [store_filter] if store_filter else list(STORES)
    
    # Debug limits
    max_stores = getattr(cfg.environment, "max_stores", None)
    if max_stores is not None:
        stores = stores[:max_stores]
        
    max_encoder_length = cfg.dataset.lookback_window
    max_prediction_length = cfg.dataset.prediction_window
    min_idx = 1
    
    all_preds = []
    all_group_names = []
    all_start_times = []
    
    for store in stores:
        print(f"Generating forecasts for store: {store}")
        df_part = load_from_cache(
            artifacts_dir=cfg.environment.artifacts_dir,
            store_filter=store
        )
        if df_part is None:
            raise FileNotFoundError(f"Cache not found for store: {store}")
            
        # Slicing evaluation window
        df_part_sliced = df_part[(df_part['time_idx'] >= min_idx) & (df_part['time_idx'] <= max_day)].copy()
        del df_part
        
        # Re-convert to category columns for consistency
        cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                    'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
        for col in cat_cols:
            if col in df_part_sliced.columns:
                df_part_sliced[col] = df_part_sliced[col].astype(str).astype('category')
                
        if len(df_part_sliced) == 0:
            continue
            
        # Construct standard TimeSeriesDataSet from training_data base dataset
        part_ds = TimeSeriesDataSet.from_dataset(
            training_data,
            df_part_sliced,
            predict=False,  # sliding windows
            stop_randomization=True
        )
        del df_part_sliced
        
        # Create standard DataLoader
        part_loader = part_ds.to_dataloader(
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cfg.environment.num_workers
        )
        
        # Generate predictions for this partition
        with torch.no_grad():
            part_preds = teacher.predict(
                part_loader,
                mode="prediction",
                trainer_kwargs={
                    "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
                    "devices": 1
                }
            )
            
        # Limit predictions in debug mode
        max_batches_per_store = getattr(cfg.environment, "max_batches_per_store", None)
        if max_batches_per_store is not None:
            limit_samples = max_batches_per_store * args.batch_size
            part_preds = part_preds[:limit_samples]
            part_decoded = part_ds.decoded_index.head(limit_samples)
        else:
            part_decoded = part_ds.decoded_index
            
        # Collect predictions and index details
        all_preds.append(part_preds.cpu())
        all_group_names.extend(part_decoded['id'].values)
        all_start_times.extend(part_decoded['time_idx_first_prediction'].values)
        
        # Memory cleanup
        del part_loader
        del part_ds
        gc.collect()
        
    # Aggregate predictions across stores
    preds = torch.cat(all_preds, dim=0)
    group_encoder = training_data._categorical_encoders['id']
    group_codes = group_encoder.transform(all_group_names)
    start_times = np.array(all_start_times)

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

    # 8. Save soft targets tensor
    print(f"Saving soft targets lookup tensor to: {output_file}")
    torch.save(soft_targets, output_file)

    # A-3: Save a JSON provenance sidecar alongside the .pt so that the tensor's
    # origin (checkpoint, scope, feature version, generation parameters) is always
    # traceable without inspecting tensor contents.
    provenance = {
        "exp_name": args.exp_name,
        "checkpoint_path": str(checkpoint_path_abs),
        "store_filter": store_filter or "full",
        "max_day": int(max_day),
        "batch_size": int(args.batch_size),
        "feature_version": int(FEATURE_VERSION),
        "tensor_shape": list(soft_targets.shape),
        "git_commit": get_git_commit_hash(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    provenance_path = output_file.replace(".pt", ".json")
    print(f"Saving soft targets provenance to: {provenance_path}")
    with open(provenance_path, "w") as _pf:
        json.dump(provenance, _pf, indent=4)

    print("Soft targets generation completed successfully!")

if __name__ == "__main__":
    main()

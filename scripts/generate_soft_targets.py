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
from data.cache import load_from_cache, load_dataset_from_cache, resolve_stores, FEATURE_VERSION
from data.dataset import build_timeseries_dataset
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

def main():
    parser = argparse.ArgumentParser(description="Generate and Save TFT Teacher Forecasts as Soft Targets")
    parser.add_argument("--env", type=str, default="local", help="Environment configuration name")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment configuration name")
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
    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    set_seed(cfg.environment.seed)

    # Determine default max day for soft target generation
    train_end = cfg.dataset.splits.train.end
    prediction_window = cfg.dataset.prediction_window
    max_prediction_start = train_end - prediction_window + 1
    max_day = args.max_day if args.max_day is not None else max_prediction_start

    # Define output file path under artifacts/soft_targets/
    artifacts_dir = resolve_path(cfg.environment.artifacts_dir)
    output_dir = os.path.join(artifacts_dir, "soft_targets")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Preprocessed Data
    from utils.paths import get_dataset_dir
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
    stores = resolve_stores(cfg.environment.store_filter)
    
    # Debug limits
    max_stores = getattr(cfg.environment, "max_stores", None)
    if max_stores is not None:
        stores = stores[:max_stores]
        
    max_encoder_length = cfg.dataset.lookback_window
    max_prediction_length = cfg.dataset.prediction_window
    min_idx = 1
    
    forecast_horizon = cfg.dataset.prediction_window
    chunk_size = getattr(cfg.environment, "soft_targets_chunk_size", 500)
    print(f"Using soft targets chunk size: {chunk_size}")

    for store in stores:
        print(f"Generating forecasts for store: {store}")
        df_part = load_from_cache(
            artifacts_dir=ds_dir,
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

        # Resolve all unique groups and mapping for the entire store partition first
        group_encoder = training_data._categorical_encoders['id']
        group_names_all = df_part_sliced['id'].unique()
        group_codes_all = group_encoder.transform(group_names_all)
        
        unique_groups = sorted(list(set(group_codes_all)))
        group_to_local = {g: idx for idx, g in enumerate(unique_groups)}
        
        # Allocate store local lookup tensor: (num_store_groups, max_day + 1, forecast_horizon)
        store_soft_targets = torch.zeros((len(unique_groups), max_day + 1, forecast_horizon), dtype=torch.float32)
        
        # Chunk items to control TimeSeriesDataSet RAM footprint
        unique_items = df_part_sliced['item_id'].unique()
        batches_processed = 0
        max_batches_per_store = getattr(cfg.environment, "max_batches_per_store", None)
        
        for i in range(0, len(unique_items), chunk_size):
            chunk_items = unique_items[i : i + chunk_size]
            df_chunk = df_part_sliced[df_part_sliced['item_id'].isin(chunk_items)].copy()
            if len(df_chunk) == 0:
                continue
                
            # Construct dataset for chunk
            chunk_ds = TimeSeriesDataSet.from_dataset(
                training_data,
                df_chunk,
                predict=False,  # sliding windows
                stop_randomization=True
            )
            del df_chunk
            
            # Create DataLoader for chunk
            chunk_loader = chunk_ds.to_dataloader(
                train=False,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=cfg.environment.num_workers
            )
            
            num_batches = len(chunk_loader)
            if max_batches_per_store is not None:
                if batches_processed >= max_batches_per_store:
                    del chunk_ds
                    del chunk_loader
                    break
                if batches_processed + num_batches > max_batches_per_store:
                    limit_samples = (max_batches_per_store - batches_processed) * args.batch_size
                else:
                    limit_samples = None
            else:
                limit_samples = None
            
            # Generate predictions for this chunk
            with torch.no_grad():
                chunk_preds = teacher.predict(
                    chunk_loader,
                    mode="prediction",
                    trainer_kwargs={
                        "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
                        "devices": 1
                    }
                )
                
            if limit_samples is not None:
                chunk_preds = chunk_preds[:limit_samples]
                chunk_decoded = chunk_ds.decoded_index.head(limit_samples)
            else:
                chunk_decoded = chunk_ds.decoded_index
                
            chunk_group_names = chunk_decoded['id'].values
            chunk_start_times = chunk_decoded['time_idx_first_prediction'].values
            
            # Map codes and assign directly to the store-local tensor
            chunk_group_codes = group_encoder.transform(chunk_group_names)
            chunk_local_codes = np.array([group_to_local[g] for g in chunk_group_codes])
            
            local_codes_tensor = torch.tensor(chunk_local_codes, dtype=torch.long)
            start_times_tensor = torch.tensor(chunk_start_times, dtype=torch.long)
            store_soft_targets[local_codes_tensor, start_times_tensor] = chunk_preds.cpu()
            
            batches_processed += num_batches
            
            # Reclaim chunk memory immediately
            del chunk_ds
            del chunk_loader
            del chunk_preds
            gc.collect()
            
        # Save soft targets store partition
        output_file = os.path.join(output_dir, f"{args.exp_name}_{store}.pt")
        print(f"Saving soft targets partition to: {output_file}")
        torch.save({
            "unique_groups": unique_groups,
            "tensor": store_soft_targets
        }, output_file)
        
        # Save a JSON provenance sidecar alongside each store partition
        provenance = {
            "exp_name": args.exp_name,
            "store": store,
            "checkpoint_path": str(checkpoint_path_abs),
            "max_day": int(max_day),
            "batch_size": int(args.batch_size),
            "feature_version": int(FEATURE_VERSION),
            "tensor_shape": list(store_soft_targets.shape),
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        provenance_path = output_file.replace(".pt", ".json")
        print(f"Saving soft targets provenance to: {provenance_path}")
        with open(provenance_path, "w") as _pf:
            json.dump(provenance, _pf, indent=4)
            
        # Reclaim store memory
        del store_soft_targets
        gc.collect()

    print("Soft targets generation completed successfully!")

if __name__ == "__main__":
    main()

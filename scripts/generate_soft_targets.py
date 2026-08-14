import os
import sys
import json
import time
import argparse
import datetime
import hashlib
import pickle
import platform
from importlib import metadata as importlib_metadata
import torch
import numpy as np

# Add repository root to python path to allow importing packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, get_git_commit_hash
from utils.paths import resolve_path
from utils.seed import set_seed
from data.cache import load_from_cache, resolve_stores, FEATURE_VERSION, is_cache_valid
from data.dataset import build_timeseries_dataset
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet


def _sha256_file(path, chunk_size=1024 * 1024):
    """Return a streaming SHA-256 without loading a checkpoint or parquet into RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_object(value):
    """Fingerprint serialized dataset state for provenance, not cross-version equality."""
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def _package_version(distribution):
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _assert_dataset_state_matches_checkpoint(training_data, teacher):
    """Fail before inference if current metadata changes teacher input/target scaling.

    Soft-target generation uses the metadata-backed TimeSeriesDataSet to produce
    model inputs and ``target_scale``.  A valid parquet feature-version alone is
    insufficient: a stale metadata pickle can change encoders, normalizers, or
    real-feature scalers while retaining that version number.
    """
    checkpoint_parameters = teacher.hparams.get("dataset_parameters")
    if checkpoint_parameters is None:
        raise ValueError("Teacher checkpoint does not contain dataset_parameters for compatibility validation")

    current_parameters = training_data.get_parameters()

    current_id_encoder = current_parameters["categorical_encoders"]["id"]
    checkpoint_id_encoder = checkpoint_parameters["categorical_encoders"]["id"]
    if not np.array_equal(current_id_encoder.classes_, checkpoint_id_encoder.classes_):
        raise ValueError("Generation metadata public id encoder differs from the teacher checkpoint")

    current_normalizer = current_parameters["target_normalizer"]
    checkpoint_normalizer = checkpoint_parameters["target_normalizer"]
    current_norm = current_normalizer.norm_.sort_index()
    checkpoint_norm = checkpoint_normalizer.norm_.sort_index()
    if (
        not current_norm.index.equals(checkpoint_norm.index)
        or list(current_norm.columns) != list(checkpoint_norm.columns)
        or current_norm.shape != checkpoint_norm.shape
        or not np.allclose(current_norm.to_numpy(), checkpoint_norm.to_numpy(), rtol=1e-7, atol=1e-8)
    ):
        raise ValueError("Generation target normalizer differs from the teacher checkpoint")

    for name, checkpoint_scaler in checkpoint_parameters["scalers"].items():
        current_scaler = current_parameters["scalers"].get(name)
        if current_scaler is None:
            raise ValueError(f"Generation metadata is missing teacher scaler: {name}")
        for attribute in ("mean_", "scale_", "var_"):
            if not np.allclose(
                np.asarray(getattr(current_scaler, attribute)),
                np.asarray(getattr(checkpoint_scaler, attribute)),
                rtol=1e-7,
                atol=1e-8,
            ):
                raise ValueError(
                    f"Generation scaler '{name}' differs from the teacher checkpoint ({attribute})"
                )

    return current_parameters, checkpoint_parameters

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
    parser.add_argument(
        "--origins-file",
        default=None,
        help="Optional JSON file containing an explicit {'origins': [...]} training-start schedule.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing cache partition for this experiment name",
    )
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

    # Determine default max day for soft target generation (training split end)
    train_end = cfg.dataset.splits.train.end
    max_day = args.max_day if args.max_day is not None else train_end
    requested_origins = None
    origins_file_abs = None
    if args.origins_file:
        origins_file_abs = resolve_path(args.origins_file)
        with open(origins_file_abs, "r", encoding="utf-8") as handle:
            requested_origins = json.load(handle).get("origins")
        if (
            not isinstance(requested_origins, list)
            or not requested_origins
            or not all(isinstance(origin, int) for origin in requested_origins)
            or requested_origins != sorted(set(requested_origins))
        ):
            raise ValueError("--origins-file must contain a non-empty, sorted unique integer 'origins' list")
        if requested_origins[0] < 1 or requested_origins[-1] + cfg.dataset.prediction_window - 1 > max_day:
            raise ValueError("--max-day does not cover the full requested origin schedule and horizon")

    # Define output file path under artifacts/soft_targets/
    artifacts_dir = resolve_path(cfg.environment.artifacts_dir)
    output_dir = os.path.join(artifacts_dir, "soft_targets")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Verify preprocessed dataset caches exist
    from utils.paths import get_dataset_dir
    ds_dir = get_dataset_dir(cfg)
    metadata_path = os.path.join(ds_dir, "metadata", "global_metadata.pkl")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Generation metadata file is missing: {metadata_path}")
    stores = resolve_stores(cfg.environment.store_filter)
    for store in stores:
        if not is_cache_valid(ds_dir, store):
            raise FileNotFoundError(
                f"Valid cache not found for store '{store}' under '{ds_dir}'. "
                "Please run prepare_dataset.py first."
            )

    # 2. Build Base Dataset (to inherit encoders and normalizers)
    print("Building base training dataset...")
    training_data = build_timeseries_dataset(None, cfg, is_train=True)

    # 3. Load Frozen TFT Model
    checkpoint_path_abs = resolve_path(args.checkpoint_path)
    if not os.path.isfile(checkpoint_path_abs):
        raise FileNotFoundError(f"Teacher checkpoint is missing: {checkpoint_path_abs}")
    checkpoint_sha256 = _sha256_file(checkpoint_path_abs)
    print(f"Teacher checkpoint SHA-256: {checkpoint_sha256}")
    print(f"Loading TFT teacher model from checkpoint: {checkpoint_path_abs}")
    teacher = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path_abs)
    
    # Run in evaluation mode and move to device
    teacher.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = teacher.to(device)

    current_dataset_parameters, checkpoint_dataset_parameters = _assert_dataset_state_matches_checkpoint(
        training_data, teacher
    )
    print("Generation metadata matches the teacher checkpoint's public encoder, normalizer, and scalers.")
    common_provenance = {
        "checkpoint_path": str(checkpoint_path_abs),
        "checkpoint_sha256": checkpoint_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "generation_dataset_state_sha256": _sha256_object(current_dataset_parameters),
        "teacher_dataset_state_sha256": _sha256_object(checkpoint_dataset_parameters),
        "target_normalizer_state_sha256": _sha256_object(current_dataset_parameters["target_normalizer"]),
        "categorical_encoders_state_sha256": _sha256_object(current_dataset_parameters["categorical_encoders"]),
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pytorch_forecasting": _package_version("pytorch-forecasting"),
            "lightning": _package_version("lightning"),
        },
    }

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
        parquet_path = os.path.join(ds_dir, "data", f"preprocessed_{store}.parquet")
        version_path = os.path.join(ds_dir, "data", f"preprocessed_{store}.version")
        if not os.path.isfile(parquet_path) or not os.path.isfile(version_path):
            raise FileNotFoundError(f"Missing preprocessing artifact(s) for store {store}")
        with open(version_path, "r", encoding="utf-8") as version_handle:
            feature_version_value = version_handle.read().strip()

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
        
        # Allocate store local lookup tensor: (num_store_groups, max_day + 1, forecast_horizon) initialized to NaN
        store_soft_targets = torch.full((len(unique_groups), max_day + 1, forecast_horizon), float('nan'), dtype=torch.float32)
        
        # Chunk items to control TimeSeriesDataSet RAM footprint
        unique_items = df_part_sliced['item_id'].unique()
        batches_processed = 0
        max_batches_per_store = getattr(cfg.environment, "max_batches_per_store", None)
        
        num_items = len(unique_items)
        num_chunks = int(np.ceil(num_items / chunk_size))
        print(f"Store {store} | Total unique items: {num_items} | Configured chunk size: {chunk_size} | Total chunks: {num_chunks}")
        
        chunk_idx = 1
        for i in range(0, len(unique_items), chunk_size):
            chunk_items = unique_items[i : i + chunk_size]
            df_chunk = df_part_sliced[df_part_sliced['item_id'].isin(chunk_items)].copy()
            if len(df_chunk) == 0:
                continue
                
            # Construct dataset for chunk
            t0 = time.time()
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Building TimeSeriesDataSet...")
            chunk_ds = TimeSeriesDataSet.from_dataset(
                training_data,
                df_chunk,
                predict=False,  # sliding windows
                stop_randomization=True
            )
            if requested_origins is not None:
                requested_set = set(requested_origins)
                chunk_ds = chunk_ds.filter(
                    lambda idx: idx["time_idx_first_prediction"].isin(requested_set)
                )
                realized = sorted(chunk_ds.decoded_index["time_idx_first_prediction"].unique().tolist())
                if realized != requested_origins:
                    raise AssertionError(
                        f"{store} chunk {chunk_idx} did not realize the exact requested origins"
                    )
            del df_chunk
            t_dataset = time.time() - t0
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Building TimeSeriesDataSet completed in {t_dataset:.2f}s")
            
            # Create DataLoader for chunk
            t0 = time.time()
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Building DataLoader...")
            chunk_loader = chunk_ds.to_dataloader(
                train=False,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=cfg.environment.num_workers
            )
            t_loader = time.time() - t0
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Building DataLoader completed in {t_loader:.2f}s")
            
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
            
            # Generate predictions for this chunk using native PyTorch forward pass
            t0 = time.time()
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Running teacher inference...")
            chunk_preds_list = []
            with torch.no_grad():
                for batch in chunk_loader:
                    x, _ = batch
                    x_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                    out = teacher(x_device)
                    pred_val = teacher.to_prediction(out)
                    chunk_preds_list.append(pred_val.cpu())
            chunk_preds = torch.cat(chunk_preds_list, dim=0)
            t_inference = time.time() - t0
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Running teacher inference completed in {t_inference:.2f}s")
            
            # Post-processing predictions
            t0 = time.time()
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Post-processing predictions...")
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
            t_postprocess = time.time() - t0
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Post-processing predictions completed in {t_postprocess:.2f}s")
            
            # Writing into lookup tensor
            t0 = time.time()
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Writing into lookup tensor...")
            local_codes_tensor = torch.tensor(chunk_local_codes, dtype=torch.long)
            start_times_tensor = torch.tensor(chunk_start_times, dtype=torch.long)
            store_soft_targets[local_codes_tensor, start_times_tensor] = chunk_preds.cpu()
            t_assign = time.time() - t0
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Writing into lookup tensor completed in {t_assign:.2f}s")
            
            print(f"Store {store} | Chunk {chunk_idx}/{num_chunks} | Saving complete.")
            batches_processed += num_batches
            chunk_idx += 1
            
            # Reclaim chunk memory immediately
            del chunk_ds
            del chunk_loader
            del chunk_preds
            gc.collect()
            
        # Save soft targets store partition
        output_file = os.path.join(output_dir, f"{args.exp_name}_{store}.pt")
        provenance_path = output_file.replace(".pt", ".json")
        if (os.path.exists(output_file) or os.path.exists(provenance_path)) and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing soft-target artifact: {output_file}. "
                "Use a new --exp-name for a fresh cache, or pass --overwrite deliberately."
            )
        t0 = time.time()
        print(f"Saving soft targets partition to: {output_file}")
        torch.save({
            "unique_groups": unique_groups,
            "tensor": store_soft_targets
        }, output_file)
        t_save = time.time() - t0
        print(f"Saving soft targets partition completed in {t_save:.2f}s")
        
        # Save a JSON provenance sidecar alongside each store partition
        provenance = {
            "exp_name": args.exp_name,
            "store": store,
            "max_day": int(max_day),
            "batch_size": int(args.batch_size),
            "chunk_size": int(chunk_size),
            "feature_version": int(FEATURE_VERSION),
            "preprocessed_parquet_path": str(parquet_path),
            "preprocessed_parquet_sha256": _sha256_file(parquet_path),
            "feature_version_path": str(version_path),
            "feature_version_value": feature_version_value,
            "feature_version_sha256": _sha256_file(version_path),
            "tensor_shape": list(store_soft_targets.shape),
            "requested_origins_file": str(origins_file_abs) if origins_file_abs else None,
            "requested_origins_sha256": _sha256_file(origins_file_abs) if origins_file_abs else None,
            "requested_origin_count": len(requested_origins) if requested_origins else None,
            "requested_origins": requested_origins,
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        provenance.update(common_provenance)
        print(f"Saving soft targets provenance to: {provenance_path}")
        with open(provenance_path, "w") as _pf:
            json.dump(provenance, _pf, indent=4)
            
        # Reclaim store memory
        del store_soft_targets
        gc.collect()

    print("Soft targets generation completed successfully!")

if __name__ == "__main__":
    main()

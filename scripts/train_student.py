import os
import sys
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Add repository root to python path to allow importing packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, save_config, save_metadata
from utils.paths import resolve_path
from utils.seed import set_seed
from utils.logging import get_csv_logger
from data.cache import load_dataset_from_cache
from data.dataset import build_timeseries_dataset
from models.student import M5TransformerStudent

def main():
    parser = argparse.ArgumentParser(description="Train Compact Transformer Student Model on M5 Dataset")
    parser.add_argument("--env", type=str, default="local", help="Environment configuration name")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment configuration name")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name directory (required — e.g. exp_full_phase1)")
    
    # Overrides
    parser.add_argument("--kd", action="store_true", help="Enable teacher-student Knowledge Distillation (KD)")
    parser.add_argument("--no-kd", dest="kd", action="store_false", help="Disable teacher-student KD")
    parser.set_defaults(kd=None) # Use config setting if not specified on CLI
    
    parser.add_argument("--alpha", type=float, default=None, help="Supervised loss weight (1-alpha is distillation loss weight)")
    parser.add_argument("--soft-targets-path", type=str, default=None, 
                        help="Path to the pre-computed teacher soft targets tensor (.pt file)")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--limit-train-batches", type=float, default=None, help="Limit train batches per epoch")
    parser.add_argument("--limit-val-batches", type=float, default=None, help="Limit validation batches per epoch")
    parser.add_argument("--max-stores", type=int, default=None, help="Limit maximum number of store partitions to stream")
    parser.add_argument("--max-batches-per-store", type=int, default=None, help="Limit maximum batches per store partition")
    args = parser.parse_args()

    # B-4: Require an explicit experiment name to avoid accidentally overwriting
    # existing artifacts (e.g. the pre-existing exp_001 checkpoints).
    if args.exp_name is None:
        raise ValueError(
            "--exp-name is required. Provide a descriptive name for this run, "
            "e.g. --exp-name exp_full_phase1"
        )

    # 1. Load Configurations
    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    
    # Apply debug flags directly to config environment settings
    if args.max_stores is not None:
        cfg.environment.max_stores = args.max_stores
    if args.max_batches_per_store is not None:
        cfg.environment.max_batches_per_store = args.max_batches_per_store
    
    # Set seed
    set_seed(cfg.environment.seed)

    # Determine KD flag and alpha
    kd_enabled = args.kd if args.kd is not None else cfg.student.kd
    alpha = args.alpha if args.alpha is not None else cfg.student.alpha
    epochs = args.epochs if args.epochs is not None else cfg.student.epochs
    batch_size = args.batch_size if args.batch_size is not None else cfg.student.batch_size
    limit_train_batches = args.limit_train_batches if args.limit_train_batches is not None else cfg.student.limit_train_batches
    limit_val_batches = args.limit_val_batches if args.limit_val_batches is not None else cfg.student.limit_val_batches

    # 2. Load Preprocessed Data
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

    # 3. Build Datasets
    print("Building TimeSeriesDataSet objects...")
    training_data = build_timeseries_dataset(df, cfg, is_train=True)
    
    from data.dataset import StorePartitionManager
    partition_manager = StorePartitionManager(training_data, cfg)

    # 4. Create DataLoaders via Partition Manager
    train_loader = partition_manager.train_dataloader(batch_size=batch_size)
    val_loader = partition_manager.val_dataloader(
        batch_size=batch_size, 
        max_idx=cfg.dataset.splits.validation.end
    )

    # 5. Load Soft Targets if running under KD
    soft_targets = None
    if kd_enabled:
        soft_targets_path = args.soft_targets_path
        if not soft_targets_path:
            exp_dir = getattr(cfg.environment, "experiment_artifacts_dir", None)
            if exp_dir is not None:
                from utils.paths import get_experiment_dir
                exp_art_dir = get_experiment_dir(cfg)
                path1 = os.path.join(exp_art_dir, "soft_targets", f"{args.exp_name}.pt")
                path2 = os.path.join(exp_art_dir, "outputs", "soft_targets", f"{args.exp_name}.pt")
                if os.path.exists(path1):
                    soft_targets_path = path1
                elif os.path.exists(path2):
                    soft_targets_path = path2
                else:
                    raise FileNotFoundError(
                        f"Soft targets file for '{args.exp_name}' not found under configured experiment_artifacts_dir at '{exp_art_dir}'"
                    )
            else:
                artifacts_dir = resolve_path(cfg.environment.artifacts_dir)
                soft_targets_path = os.path.join(artifacts_dir, "soft_targets", f"{args.exp_name}.pt")
            
        soft_targets_path_abs = resolve_path(soft_targets_path)
        print(f"Loading pre-computed teacher forecasts from: {soft_targets_path_abs}")
        if not os.path.exists(soft_targets_path_abs):
            raise FileNotFoundError(
                f"Soft targets file not found at {soft_targets_path_abs}. "
                "Run generate_soft_targets.py first."
            )
        
        soft_targets = torch.load(soft_targets_path_abs)
        print(f"Loaded soft targets tensor of shape: {soft_targets.shape}")

        # A-3: Validate tensor dimensions against the fitted dataset and config.
        # Catches scope mismatches (e.g. single-store .pt loaded for a full run)
        # before training starts rather than silently producing wrong gradients.
        expected_groups = len(training_data._categorical_encoders['id'].classes_)
        if soft_targets.shape[0] != expected_groups:
            raise RuntimeError(
                f"Soft targets group dimension ({soft_targets.shape[0]}) does not "
                f"match the expected number of series ({expected_groups}). "
                "The file may have been generated for a different dataset scope "
                "(e.g. single-store vs full dataset). Re-run generate_soft_targets.py."
            )
        if soft_targets.shape[2] != cfg.dataset.prediction_window:
            raise RuntimeError(
                f"Soft targets horizon dimension ({soft_targets.shape[2]}) does not "
                f"match the configured prediction_window ({cfg.dataset.prediction_window})."
            )

    # 6. Instantiate Student Model
    print("Instantiating Compact Transformer Student model...")
    model = M5TransformerStudent(
        training_dataset=training_data,
        d_model=cfg.student.d_model,
        nhead=cfg.student.nhead,
        num_layers=cfg.student.layers,
        dim_feedforward=cfg.student.dim_feedforward,
        dropout=cfg.student.dropout,
        lr=cfg.student.lr,
        alpha=alpha if kd_enabled else 1.0,
        lookback_window=cfg.dataset.lookback_window,
        prediction_window=cfg.dataset.prediction_window,
        soft_targets=soft_targets
    )

    # 7. Set up Logs and Outputs
    # Experiment folder: outputs_dir / student / (kd or no_kd) / exp_name /
    model_mode = "student/kd" if kd_enabled else "student/no_kd"
    exp_dir = os.path.join(resolve_path(cfg.environment.outputs_dir), model_mode, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save the fully merged configuration into experiment folder for complete reproducibility
    config_save_path = os.path.join(exp_dir, "config.yaml")
    save_config(cfg, config_save_path)
    print(f"Merged config saved to {config_save_path}")

    # Set up Logger and Callbacks
    logger = get_csv_logger(cfg.environment.outputs_dir, model_mode, args.exp_name)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        monitor="val_loss",
        filename="best_student",
        save_top_k=1,
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.student.patience,
        min_delta=1e-4,
        mode="min"
    )

    # 8. Set up Trainer
    gradient_clip_val = getattr(cfg.environment, "gradient_clip_val", 0.1)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=cfg.environment.accelerator,
        devices=cfg.environment.devices,
        precision=cfg.environment.precision,
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        enable_model_summary=True
    )

    # 9. Run Training
    print("Starting training loop...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    best_path = checkpoint_callback.best_model_path
    print(f"Training completed. Best model checkpoint saved to: {best_path}")
    
    # Save experiment metadata
    save_metadata(
        exp_dir, 
        cfg.environment.seed, 
        checkpoint_path=best_path,
        additional_fields={
            "kd_enabled": kd_enabled,
            "alpha": float(alpha)
        }
    )

if __name__ == "__main__":
    main()

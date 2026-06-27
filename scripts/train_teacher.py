import os
import sys
import argparse
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Add repository root to python path to allow importing packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, save_config, save_metadata
from utils.paths import resolve_path
from utils.seed import set_seed
from utils.logging import get_csv_logger
from data.cache import load_from_cache, load_all_from_cache
from data.dataset import build_timeseries_dataset
from models.teacher import create_tft_teacher

def main():
    parser = argparse.ArgumentParser(description="Train TFT Teacher Model on M5 Dataset")
    parser.add_argument("--env", type=str, default="local", help="Environment configuration name")
    parser.add_argument("--exp-name", type=str, default="exp_001", help="Experiment name directory")
    
    # Overrides
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--limit-train-batches", type=float, default=None, help="Limit train batches per epoch")
    parser.add_argument("--limit-val-batches", type=float, default=None, help="Limit validation batches per epoch")
    args = parser.parse_args()

    # 1. Load Configurations
    cfg = load_config(env_name=args.env)
    
    # Set seed
    set_seed(cfg.environment.seed)

    # Apply command-line overrides to config
    epochs = args.epochs if args.epochs is not None else cfg.teacher.epochs
    batch_size = args.batch_size if args.batch_size is not None else cfg.teacher.batch_size
    limit_train_batches = args.limit_train_batches if args.limit_train_batches is not None else cfg.teacher.limit_train_batches
    limit_val_batches = args.limit_val_batches if args.limit_val_batches is not None else cfg.teacher.limit_val_batches

    # 2. Load cached Parquet dataset
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

    # 5. Instantiate TFT model
    print("Instantiating Temporal Fusion Transformer model...")
    tft = create_tft_teacher(training_data, cfg)
    print(f"Number of parameters: {tft.size()/1e3:.1f}k")

    # 6. Set up Logs and Outputs
    # Experiment folder: outputs_dir / teacher / exp_name /
    exp_dir = os.path.join(resolve_path(cfg.environment.outputs_dir), "teacher", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save the fully merged configuration into experiment folder for complete reproducibility
    config_save_path = os.path.join(exp_dir, "config.yaml")
    save_config(cfg, config_save_path)
    print(f"Merged config saved to {config_save_path}")

    # Set up Logger and Callbacks
    logger = get_csv_logger(cfg.environment.outputs_dir, "teacher", args.exp_name)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        monitor="val_loss",
        filename="best_tft_teacher",
        save_top_k=1,
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.teacher.patience,
        min_delta=1e-4,
        mode="min"
    )

    # 7. Set up Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=cfg.environment.accelerator,
        devices=cfg.environment.devices,
        precision=cfg.environment.precision,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        enable_model_summary=True
    )

    # 8. Train the model
    print("Starting training loop...")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Print best checkpoint path
    best_path = checkpoint_callback.best_model_path
    print(f"Training completed. Best model checkpoint saved to: {best_path}")
    
    # Save experiment metadata
    save_metadata(exp_dir, cfg.environment.seed, checkpoint_path=best_path)

if __name__ == "__main__":
    main()

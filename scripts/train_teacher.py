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
from data.cache import resolve_stores, is_cache_valid
from data.dataset import build_timeseries_dataset
from models.teacher import create_tft_teacher
from models.wrmsse_informed import build_wrmsse_informed_coefficients
import torch
from scripts.observability import ObservabilityCallback
class EpochMetricsLoggingCallback(pl.Callback):
    """
    Minimal PyTorch Lightning callback to print epoch-level metrics to stdout.
    Required because PL does not print metrics when the progress bar is disabled.
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
            
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        metrics = trainer.callback_metrics
        
        # Format and log epoch-level metrics
        formatted = []
        for k, v in sorted(metrics.items()):
            if k.endswith("_step"):
                continue
            val = v.item() if isinstance(v, torch.Tensor) else v
            formatted.append(f"{k}: {val:.4f}")
            
        if formatted:
            print(f"Epoch {epoch}/{max_epochs} completed | " + " | ".join(formatted))

def main():
    parser = argparse.ArgumentParser(description="Train TFT Teacher Model on M5 Dataset")
    parser.add_argument("--env", type=str, default="local", help="Environment configuration name")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment configuration name")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name directory (required — e.g. exp_full_phase1)")
    
    # Resume options
    parser.add_argument("--resume", action="store_true", help="Auto-resume training from last.ckpt in experiment directory")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Explicit path to checkpoint file to resume from")
    
    # Overrides
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

    # Apply command-line overrides to config
    epochs = args.epochs if args.epochs is not None else cfg.teacher.epochs
    batch_size = args.batch_size if args.batch_size is not None else cfg.teacher.batch_size
    limit_train_batches = args.limit_train_batches if args.limit_train_batches is not None else cfg.teacher.limit_train_batches
    limit_val_batches = args.limit_val_batches if args.limit_val_batches is not None else cfg.teacher.limit_val_batches

    # 2. Verify preprocessed dataset caches exist
    from utils.paths import get_dataset_dir
    ds_dir = get_dataset_dir(cfg)
    stores = resolve_stores(cfg.environment.store_filter)
    for store in stores:
        if not is_cache_valid(ds_dir, store):
            raise FileNotFoundError(
                f"Valid cache not found for store '{store}' under '{ds_dir}'. "
                "Please run prepare_dataset.py first."
            )

    # 3. Build Datasets
    print("Building TimeSeriesDataSet objects...")
    training_data = build_timeseries_dataset(None, cfg, is_train=True)

    wrmsse_coefficients = None
    if getattr(cfg.teacher, "loss", "quantile").lower() == "wrmsse_informed":
        coefficient_bundle = build_wrmsse_informed_coefficients(cfg)
        if coefficient_bundle.audit["pathological"]:
            raise RuntimeError(
                "WRMSSE-informed coefficient distribution is pathological: "
                + "; ".join(coefficient_bundle.audit["pathological_reasons"])
            )
        if not getattr(cfg.teacher, "pretraining_audit_approved", False):
            raise RuntimeError(
                "Full WRMSSE-informed training is gated until the pre-training audit is "
                "reviewed. Run scripts/audit_wrmsse_informed.py, review its report, then "
                "set teacher.pretraining_audit_approved=true explicitly."
            )
        wrmsse_coefficients = coefficient_bundle.by_series
    
    from data.dataset import StorePartitionManager
    partition_manager = StorePartitionManager(
        training_data, cfg, series_coefficients=wrmsse_coefficients
    )

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

    # Resolve resume checkpoint path if requested
    resume_ckpt_path = None
    if args.ckpt_path:
        resume_ckpt_path = resolve_path(args.ckpt_path)
        if not os.path.exists(resume_ckpt_path):
            raise FileNotFoundError(f"Specified checkpoint for resume not found at: {resume_ckpt_path}")
        print(f"Resuming training from explicit checkpoint: {resume_ckpt_path}")
    elif args.resume:
        default_last = os.path.join(exp_dir, "last.ckpt")
        if os.path.exists(default_last):
            resume_ckpt_path = default_last
            print(f"Auto-resuming training from latest checkpoint: {resume_ckpt_path}")
        else:
            raise FileNotFoundError(f"--resume flag passed, but 'last.ckpt' was not found in: {exp_dir}")

    # Set up Logger and Callbacks
    logger = get_csv_logger(cfg.environment.outputs_dir, "teacher", args.exp_name)
    
    checkpoint_filename = getattr(cfg.teacher, "checkpoint_filename", "best_tft_teacher")
    save_top_k = getattr(cfg.teacher, "save_top_k", 1)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        monitor="val_loss",
        filename=checkpoint_filename,
        save_top_k=save_top_k,
        save_last=True,
        mode="min"
    )
    
    early_stop_patience = getattr(cfg.teacher, "early_stopping_patience", cfg.teacher.patience)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stop_patience,
        min_delta=1e-4,
        mode="min"
    )

    enable_progress_bar = True
    obs_callback = ObservabilityCallback()
    callbacks = [early_stop_callback, checkpoint_callback, obs_callback]
    if args.env == "kaggle":
        enable_progress_bar = False
        callbacks.append(EpochMetricsLoggingCallback())

    # 7. Set up Trainer
    # Experiment-specific clipping takes precedence so teacher experiments can
    # deliberately test a clipping intervention without changing the shared
    # cluster environment configuration.
    gradient_clip_val = getattr(
        cfg.teacher,
        "gradient_clip_val",
        getattr(cfg.environment, "gradient_clip_val", 0.1),
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=cfg.environment.accelerator,
        devices=cfg.environment.devices,
        precision=cfg.environment.precision,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        enable_model_summary=True,
        enable_progress_bar=enable_progress_bar
    )

    # 8. Train the model
    print("Starting training loop...")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt_path)
    
    # Print best checkpoint path
    best_path = checkpoint_callback.best_model_path
    print(f"Training completed. Best model checkpoint saved to: {best_path}")
    
    # Save experiment metadata
    import hashlib
    checkpoint_hashes = []
    best_validation_loss = None
    for f in os.listdir(exp_dir):
        if f.endswith(".ckpt"):
            ckpt_p = os.path.join(exp_dir, f)
            sz = os.path.getsize(ckpt_p)
            hasher = hashlib.sha256()
            with open(ckpt_p, 'rb') as fp:
                hasher.update(fp.read())
            sha256 = hasher.hexdigest()
            
            try:
                ckpt_data = torch.load(ckpt_p, map_location="cpu")
                epoch = ckpt_data.get("epoch")
                global_step = ckpt_data.get("global_step")
                callbacks_data = ckpt_data.get("callbacks", {})
                score = None
                for cb_type, cb_state in callbacks_data.items():
                    if "ModelCheckpoint" in cb_type and "best_model_score" in cb_state:
                        sc = cb_state.get("best_model_score")
                        if sc is not None:
                            score = sc.item() if isinstance(sc, torch.Tensor) else sc
                            break
            except Exception:
                epoch, global_step, score = None, None, None

            if os.path.normpath(ckpt_p) == os.path.normpath(best_path):
                best_validation_loss = score

            checkpoint_hashes.append({
                "experiment_identifier": args.exp_name,
                "hidden_size": cfg.teacher.hidden_size,
                "hidden_continuous_size": getattr(cfg.teacher, "hidden_continuous_size", 8),
                "epoch": epoch,
                "global_step": global_step,
                "monitored_metric": "val_loss",
                "monitored_score": score,
                "checkpoint_path": ckpt_p,
                "file_size": sz,
                "sha256_hash": sha256
            })

    best_epoch = next((ch["epoch"] for ch in checkpoint_hashes if os.path.normpath(ch["checkpoint_path"]) == os.path.normpath(best_path)), None)
    best_step = next((ch["global_step"] for ch in checkpoint_hashes if os.path.normpath(ch["checkpoint_path"]) == os.path.normpath(best_path)), None)


    additional_fields = {
        "experiment_identifier": args.exp_name,
        "gradient_clip_val": gradient_clip_val,
        "scheduler_reduction_events": obs_callback.reduction_events,
        "total_training_duration": obs_callback.total_training_duration,
        "checkpoint_hashes": checkpoint_hashes,
        "best_validation_loss": best_validation_loss,
        "best_checkpoint_path": best_path,
        "best_checkpoint_epoch": best_epoch,
        "best_checkpoint_global_step": best_step,
    }
    if torch.cuda.is_available():
        additional_fields["peak_gpu_memory_MB"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
    save_metadata(exp_dir, cfg.environment.seed, checkpoint_path=best_path, additional_fields=additional_fields)

    # Optional / Non-blocking curve plotting helper call
    try:
        from scripts.plot_training_curves import plot_curves
        plot_curves(exp_dir=exp_dir)
    except Exception as e:
        print(f"[Plotting Warning] Could not generate loss curve plot: {e}")

if __name__ == "__main__":
    main()

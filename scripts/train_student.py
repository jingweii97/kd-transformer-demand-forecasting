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
from data.cache import resolve_stores, is_cache_valid
from data.dataset import build_timeseries_dataset
from models.student import M5TransformerStudent
from models.wrmsse_informed import build_wrmsse_informed_coefficients
import torch

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
    parser = argparse.ArgumentParser(description="Train Compact Transformer Student Model on M5 Dataset")
    parser.add_argument("--env", type=str, default="local", help="Environment configuration name")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment configuration name")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name directory (required — e.g. exp_full_phase1)")
    
    # Resume options
    parser.add_argument("--resume", action="store_true", help="Auto-resume training from last.ckpt in experiment directory")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Explicit path to checkpoint file to resume from")
    
    # Overrides
    parser.add_argument("--kd", action="store_true", help="Enable teacher-student Knowledge Distillation (KD)")
    parser.add_argument("--no-kd", dest="kd", action="store_false", help="Disable teacher-student KD")
    parser.set_defaults(kd=None) # Use config setting if not specified on CLI
    
    parser.add_argument("--alpha", type=float, default=None, help="Supervised loss weight (1-alpha is distillation loss weight)")
    parser.add_argument(
        "--supervised-loss",
        choices=["huber", "wrmsse_informed"],
        default=None,
        help="Point objective for ground-truth supervision; defaults to config student.supervised_loss.",
    )
    parser.add_argument("--soft-targets-path", type=str, default=None, 
                        help="Path to the pre-computed teacher soft targets tensor (.pt file)")
    parser.add_argument(
        "--soft-targets-exp-name",
        type=str,
        default=None,
        help="Cache filename prefix for per-store soft targets; defaults to --exp-name.",
    )
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
    cfg.student.kd = kd_enabled
    
    alpha = args.alpha if args.alpha is not None else cfg.student.alpha
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    cfg.student.alpha = alpha

    supervised_loss = (
        args.supervised_loss
        if args.supervised_loss is not None
        else getattr(cfg.student, "supervised_loss", "huber")
    )
    cfg.student.supervised_loss = supervised_loss
    soft_target_exp_name = args.soft_targets_exp_name or args.exp_name
    cfg.student.soft_targets_exp_name = soft_target_exp_name
    
    epochs = args.epochs if args.epochs is not None else cfg.student.epochs
    cfg.student.epochs = epochs
    
    batch_size = args.batch_size if args.batch_size is not None else cfg.student.batch_size
    cfg.student.batch_size = batch_size
    
    limit_train_batches = args.limit_train_batches if args.limit_train_batches is not None else cfg.student.limit_train_batches
    cfg.student.limit_train_batches = limit_train_batches
    
    limit_val_batches = args.limit_val_batches if args.limit_val_batches is not None else cfg.student.limit_val_batches
    cfg.student.limit_val_batches = limit_val_batches

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

    from data.dataset import StorePartitionManager
    wrmsse_coefficients = None
    wrmsse_coefficient_audit = None
    if supervised_loss == "wrmsse_informed":
        coefficient_bundle = build_wrmsse_informed_coefficients(
            cfg, objective_config=cfg.student
        )
        if coefficient_bundle.audit["pathological"]:
            raise RuntimeError(
                "WRMSSE-informed coefficient distribution is pathological: "
                + "; ".join(coefficient_bundle.audit["pathological_reasons"])
            )
        if not getattr(cfg.student, "wrmsse_pretraining_audit_approved", False):
            raise RuntimeError(
                "WRMSSE-informed student training is gated until its shared coefficient "
                "audit is explicitly approved by setting "
                "student.wrmsse_pretraining_audit_approved=true."
            )
        wrmsse_coefficients = coefficient_bundle.by_series
        wrmsse_coefficient_audit = coefficient_bundle.audit
        print(
            "Using shared training-only WRMSSE-informed coefficients for "
            f"{len(wrmsse_coefficients):,} bottom-level series."
        )

    partition_manager = StorePartitionManager(
        training_data,
        cfg,
        exp_name=args.exp_name,
        series_coefficients=wrmsse_coefficients,
    )

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
        
        # Check if per-store target files exist instead of a single global file
        use_per_store_targets = False
        
        if soft_targets_path:
            resolved_p = resolve_path(soft_targets_path)
            if os.path.isdir(resolved_p):
                # Verify all expected per-store files (for all stores that will be streamed) exist
                stores_to_check = resolve_stores(cfg.environment.store_filter)
                missing_stores = []
                for store in stores_to_check:
                    p = os.path.join(resolved_p, f"{soft_target_exp_name}_{store}.pt")
                    if not os.path.exists(p):
                        missing_stores.append(f"{soft_target_exp_name}_{store}.pt")
                if missing_stores:
                    raise FileNotFoundError(
                        f"Soft targets directory '{resolved_p}' is missing the following expected "
                        f"per-store file(s) for cache prefix '{soft_target_exp_name}': {', '.join(missing_stores)}. "
                        f"Expected files for all streamed stores under this experiment."
                    )
                use_per_store_targets = True
                cfg.student.soft_targets_path = resolved_p
            elif os.path.isfile(resolved_p):
                # Legacy global file path mode
                pass
            else:
                raise FileNotFoundError(
                    f"Provided soft targets path '{resolved_p}' does not exist as a file or directory. "
                    f"Expected a legacy global '.pt' file or a directory containing "
                    f"'{args.exp_name}_<store>.pt' files."
                )
        else:
            # Fallback path if no path is provided via CLI: check default check directories
            from data.cache import STORES
            exp_dir = getattr(cfg.environment, "experiment_artifacts_dir", None)
            if exp_dir is not None:
                from utils.paths import get_experiment_dir
                exp_art_dir = get_experiment_dir(cfg)
                check_dirs = [os.path.join(exp_art_dir, "soft_targets"), os.path.join(exp_art_dir, "outputs", "soft_targets")]
            else:
                artifacts_dir = resolve_path(cfg.environment.artifacts_dir)
                check_dirs = [os.path.join(artifacts_dir, "soft_targets")]
            
            resolved_dir = None
            for store in STORES:
                for d in check_dirs:
                    p = os.path.join(d, f"{soft_target_exp_name}_{store}.pt")
                    if os.path.exists(p):
                        use_per_store_targets = True
                        resolved_dir = d
                        break
                if use_per_store_targets:
                    break
            if use_per_store_targets:
                cfg.student.soft_targets_path = resolved_dir

        if use_per_store_targets:
            print(f"Per-store soft targets detected for cache prefix '{soft_target_exp_name}'. Dataloader will stream them partition-by-partition.")
        else:
            # Fallback/Default path for loading legacy global soft targets tensor
            if not soft_targets_path:
                exp_dir = getattr(cfg.environment, "experiment_artifacts_dir", None)
                if exp_dir is not None:
                    from utils.paths import get_experiment_dir
                    exp_art_dir = get_experiment_dir(cfg)
                    path1 = os.path.join(exp_art_dir, "soft_targets", f"{soft_target_exp_name}.pt")
                    path2 = os.path.join(exp_art_dir, "outputs", "soft_targets", f"{soft_target_exp_name}.pt")
                    if os.path.exists(path1):
                        soft_targets_path = path1
                    elif os.path.exists(path2):
                        soft_targets_path = path2
                    else:
                        raise FileNotFoundError(
                            f"Soft targets file for '{soft_target_exp_name}' not found under configured experiment_artifacts_dir at '{exp_art_dir}'"
                        )
                else:
                    artifacts_dir = resolve_path(cfg.environment.artifacts_dir)
                    soft_targets_path = os.path.join(artifacts_dir, "soft_targets", f"{soft_target_exp_name}.pt")
            
            soft_targets_path_abs = resolve_path(soft_targets_path)
            print(f"Loading legacy global teacher forecasts from: {soft_targets_path_abs}")
            if not os.path.exists(soft_targets_path_abs):
                raise FileNotFoundError(
                    f"Soft targets file not found at {soft_targets_path_abs}. "
                    "Run generate_soft_targets.py first."
                )
            
            soft_targets = torch.load(soft_targets_path_abs, weights_only=False)
            print(f"Loaded soft targets tensor of shape: {soft_targets.shape}")
    
            # A-3: Validate tensor dimensions against the fitted dataset and config.
            expected_groups = len(training_data._categorical_encoders['id'].classes_)
            if soft_targets.shape[0] != expected_groups:
                raise RuntimeError(
                    f"Soft targets group dimension ({soft_targets.shape[0]}) does not "
                    f"match the expected number of series ({expected_groups}). "
                    "The file may have been generated for a different dataset scope. "
                    "Re-run generate_soft_targets.py."
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
        soft_targets=soft_targets,
        supervised_loss=supervised_loss,
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
    logger = get_csv_logger(cfg.environment.outputs_dir, model_mode, args.exp_name)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        monitor="val_loss",
        filename=getattr(cfg.student, "checkpoint_filename", "best_student"),
        save_top_k=int(getattr(cfg.student, "save_top_k", 1)),
        save_last=True,
        mode="min",
        auto_insert_metric_name=getattr(
            cfg.student, "checkpoint_auto_insert_metric_name", True
        ),
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.student.patience,
        min_delta=1e-4,
        mode="min"
    )

    enable_progress_bar = True
    callbacks = [early_stop_callback, checkpoint_callback]
    if args.env == "kaggle":
        enable_progress_bar = False
        callbacks.append(EpochMetricsLoggingCallback())

    # 8. Set up Trainer
    gradient_clip_val = getattr(cfg.environment, "gradient_clip_val", 0.1)
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

    # 9. Run Training
    print("Starting training loop...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt_path)
    
    best_path = checkpoint_callback.best_model_path
    print(f"Training completed. Best model checkpoint saved to: {best_path}")
    
    # Save experiment metadata
    save_metadata(
        exp_dir, 
        cfg.environment.seed, 
        checkpoint_path=best_path,
        additional_fields={
            "kd_enabled": kd_enabled,
            "alpha": float(model.alpha),
            "requested_alpha": float(alpha),
            "supervised_loss": supervised_loss,
            "checkpoint_monitor": "val_loss",
            "checkpoint_save_top_k": int(getattr(cfg.student, "save_top_k", 1)),
            "wrmsse_coefficient_audit": wrmsse_coefficient_audit,
        }
    )

    # Optional / Non-blocking curve plotting helper call
    try:
        from scripts.plot_training_curves import plot_curves
        plot_curves(exp_dir=exp_dir)
    except Exception as e:
        print(f"[Plotting Warning] Could not generate loss curve plot: {e}")

if __name__ == "__main__":
    main()

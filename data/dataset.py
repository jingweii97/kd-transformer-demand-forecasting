from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
import psutil, os
def build_timeseries_dataset(df, cfg, is_train=True, training_dataset=None, max_idx=None):
    """
    Constructs PyTorch Forecasting TimeSeriesDataSet using configuration settings.
    If is_train=True: builds the base dataset for training.
    If is_train=False: builds evaluation datasets with predict=True (exactly 1 prediction sequence per group).
    """
    max_encoder_length = cfg.dataset.lookback_window
    max_prediction_length = cfg.dataset.prediction_window
    
    static_categoricals = cfg.dataset.features.static_categoricals
    time_varying_known_categoricals = cfg.dataset.features.time_varying_known_categoricals
    time_varying_known_reals = cfg.dataset.features.time_varying_known_reals
    time_varying_unknown_reals = cfg.dataset.features.time_varying_unknown_reals
    
    if is_train:
        # Slices training data strictly up to the end of the Training window
        train_end = cfg.dataset.splits.train.end
        print(df.dtypes)
        print(df.memory_usage(deep=True).sort_values(ascending=False).head(20))
        process = psutil.Process(os.getpid())
        print(f"Original df: {df.memory_usage(deep=True).sum()/1024**3:.2f} GB")
        print(f"RSS: {process.memory_info().rss/1024**3:.2f} GB")
        df_train = df[df['time_idx'] <= train_end].copy()
        print(f"After train slice RSS: {process.memory_info().rss/1024**3:.2f} GB")

        
        print("Constructing TimeSeriesDataSet...")

        training_dataset = TimeSeriesDataSet(
            df_train,
            time_idx="time_idx",
            target=cfg.dataset.target,
            group_ids=cfg.dataset.group_ids,
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=cfg.dataset.group_ids, transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        print(f"After TimeSeriesDataSet RSS: {process.memory_info().rss/1024**3:.2f} GB")
        return training_dataset

    else:
        assert training_dataset is not None, "training_dataset must be provided to inherit parameters."
        assert max_idx is not None, "max_idx (end day of the evaluation split) must be provided."
        
        # Slices data up to the end of the target evaluation window
        df_eval = df[df['time_idx'] <= max_idx].copy()
        
        return TimeSeriesDataSet.from_dataset(
            training_dataset,
            df_eval,
            predict=True,
            stop_randomization=True
        )

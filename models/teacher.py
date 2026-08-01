from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss

def create_tft_teacher(training_dataset, cfg):
    """
    Instantiates a TemporalFusionTransformer teacher model from a training TimeSeriesDataSet
    and configuration settings.
    """
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=cfg.teacher.lr,
        hidden_size=cfg.teacher.hidden_size,
        hidden_continuous_size=getattr(cfg.teacher, "hidden_continuous_size", 8),
        lstm_layers=getattr(cfg.teacher, "lstm_layers", 1),
        attention_head_size=cfg.teacher.attention_heads,
        dropout=cfg.teacher.dropout,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=getattr(cfg.teacher, "scheduler_patience", getattr(cfg.teacher, "patience", 2)),
        mask_bias=-1e4
    )

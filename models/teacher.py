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
        attention_head_size=cfg.teacher.attention_heads,
        dropout=cfg.teacher.dropout,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=cfg.teacher.patience,
        mask_bias=float("-inf")
    )

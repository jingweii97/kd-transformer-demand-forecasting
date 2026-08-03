from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from models.losses import HuberLossMetric

def create_tft_teacher(training_dataset, cfg):
    """
    Instantiates a TemporalFusionTransformer teacher model from a training TimeSeriesDataSet
    and configuration settings.
    """
    loss_type = getattr(cfg.teacher, "loss", "quantile")
    
    if loss_type.lower() == "huber":
        huber_delta = getattr(cfg.teacher, "huber_delta", 1.0)
        loss = HuberLossMetric(delta=huber_delta)
        output_size = 1
    else:
        loss = QuantileLoss()
        output_size = len(loss.quantiles)
        
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=cfg.teacher.lr,
        hidden_size=cfg.teacher.hidden_size,
        hidden_continuous_size=getattr(cfg.teacher, "hidden_continuous_size", 8),
        lstm_layers=getattr(cfg.teacher, "lstm_layers", 1),
        attention_head_size=cfg.teacher.attention_heads,
        dropout=cfg.teacher.dropout,
        loss=loss,
        output_size=output_size,
        reduce_on_plateau_patience=getattr(cfg.teacher, "scheduler_patience", getattr(cfg.teacher, "patience", 2)),
        mask_bias=-1e4
    )

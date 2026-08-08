from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from models.losses import HuberLossMetric, MSELossMetric, WRMSSEInformedLossMetric

def create_tft_teacher(training_dataset, cfg):
    """
    Instantiates a TemporalFusionTransformer teacher model from a training TimeSeriesDataSet
    and configuration settings.
    """
    loss_type = getattr(cfg.teacher, "loss", "quantile").lower()
    
    if loss_type == "quantile":
        loss = QuantileLoss()
        output_size = len(loss.quantiles)
    elif loss_type == "huber":
        huber_delta = getattr(cfg.teacher, "huber_delta", 1.0)
        loss = HuberLossMetric(delta=huber_delta)
        output_size = 1
    elif loss_type == "mse":
        loss = MSELossMetric()
        output_size = 1
    elif loss_type == "wrmsse_informed":
        loss = WRMSSEInformedLossMetric()
        output_size = 1
    else:
        raise ValueError(
            "Unsupported teacher loss. Expected one of: quantile, huber, mse, "
            f"wrmsse_informed; got {loss_type!r}"
        )
        
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=cfg.teacher.lr,
        hidden_size=cfg.teacher.hidden_size,
        hidden_continuous_size=getattr(cfg.teacher, "hidden_continuous_size", 8),
        lstm_layers=getattr(cfg.teacher, "lstm_layers", 1),
        attention_head_size=(
            getattr(cfg.teacher, "attention_head_size", None)
            or cfg.teacher.attention_heads
        ),
        dropout=cfg.teacher.dropout,
        loss=loss,
        output_size=output_size,
        reduce_on_plateau_patience=getattr(cfg.teacher, "scheduler_patience", getattr(cfg.teacher, "patience", 2)),
        mask_bias=-1e4
    )

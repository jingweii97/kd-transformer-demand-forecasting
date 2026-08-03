import torch
import torch.nn.functional as F
from pytorch_forecasting.metrics import MultiHorizonMetric

class HuberLossMetric(MultiHorizonMetric):
    """
    Huber loss implemented for PyTorch Forecasting's MultiHorizonMetric interface.
    This calculates the Huber loss strictly without configuring output sizes natively,
    meaning output_size=1 must be explicitly set in the model configuration.
    """
    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def loss(self, y_pred, target):
        # y_pred is typically [batch, horizon, output_size=1] or [batch, horizon]
        # target is typically [batch, horizon]
        if y_pred.ndim > target.ndim:
            y_pred = y_pred.squeeze(-1)
        
        # Calculate element-wise huber loss
        loss_val = F.huber_loss(y_pred, target, reduction='none', delta=self.delta)
        return loss_val

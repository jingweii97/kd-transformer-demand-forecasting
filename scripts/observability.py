import time
import torch
import lightning.pytorch as pl

class ObservabilityCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.reduction_events = 0
        self.last_lr = None
        self.epoch_start_time = None
        self.training_start_time = None
        self.total_training_duration = 0.0

    def on_train_start(self, trainer, pl_module):
        self.training_start_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        # Track LR reductions
        opt = trainer.optimizers[0]
        current_lr = opt.param_groups[0]['lr']
        if self.last_lr is not None and current_lr < self.last_lr:
            self.reduction_events += 1
        self.last_lr = current_lr

    def on_train_epoch_end(self, trainer, pl_module):
        if self.last_lr is not None:
            pl_module.log("learning_rate", self.last_lr, on_epoch=True, prog_bar=False)
        if self.epoch_start_time:
            epoch_duration = time.time() - self.epoch_start_time
            pl_module.log("epoch_duration", epoch_duration, on_epoch=True, prog_bar=False)


    def on_train_end(self, trainer, pl_module):
        if self.training_start_time:
            self.total_training_duration = time.time() - self.training_start_time

    def on_after_backward(self, trainer, pl_module):
        # Gradient norm tracking at reasonable interval (e.g., every 100 steps)
        if trainer.global_step % 100 == 0:
            total_norm = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            pl_module.log("grad_norm_pre_clip", total_norm, on_step=True, on_epoch=False)

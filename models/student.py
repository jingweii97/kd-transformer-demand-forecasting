import torch
import torch.nn as nn
import lightning.pytorch as pl

class M5TransformerStudent(pl.LightningModule):
    def __init__(self, training_dataset, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, 
                 dropout=0.1, lr=1e-3, alpha=1.0, lookback_window=90, prediction_window=28, 
                 soft_targets=None):
        """
        training_dataset: TimeSeriesDataSet used for shape configurations and encoders
        d_model: Transformer hidden dimension size
        nhead: Number of attention heads
        num_layers: Number of Transformer encoder layers
        dim_feedforward: Feed-forward network dimension in Transformer layers
        dropout: Dropout rate
        lr: Learning rate
        alpha: Distillation weight (1.0 = purely supervised, 0.0 = purely distillation)
        lookback_window: L (number of lookback days)
        prediction_window: H (number of forecast days)
        soft_targets: Pre-computed teacher forecasts tensor of shape (num_groups, 1942, 28)
        """
        super().__init__()
        self.save_hyperparameters(ignore=['training_dataset', 'soft_targets'])
        self.training_dataset = training_dataset
        self.alpha = alpha
        self.lookback_window = lookback_window
        self.prediction_window = prediction_window
        
        # Store soft targets lookup tensor as a plain attribute to avoid saving it in checkpoints (6.6 GB)
        self.soft_targets = soft_targets

        # Categorical columns in the exact order PyTorch Forecasting stacks them
        self.cat_cols = training_dataset.categoricals
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=len(training_dataset._categorical_encoders[col].classes_) + 1,
                embedding_dim=8
            ) for col in self.cat_cols
        ])
        
        self.total_cat_dim = len(self.cat_cols) * 8
        self.num_reals = len(training_dataset.reals)
        
        # Project concatenated embeddings + reals to d_model
        self.input_projector = nn.Linear(self.total_cat_dim + self.num_reals, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Direct linear projection head (flattens lookback of d_model to prediction window)
        self.output_layer = nn.Linear(self.lookback_window * d_model, self.prediction_window)
        
        # Huber Loss (Smooth L1) for robustness
        self.loss_fn = nn.HuberLoss()

    def forward(self, x):
        # x is batch[0] dict from PyTorch Forecasting dataloader
        batch_size = x['encoder_cat'].shape[0]
        
        # Embed categoricals
        embedded = []
        for i, embed_layer in enumerate(self.embeddings):
            cat_tensor = x['encoder_cat'][:, :, i].long()
            # Clamp class values to prevent out-of-bounds index errors
            cat_tensor = torch.clamp(cat_tensor, 0, embed_layer.num_embeddings - 1)
            embedded.append(embed_layer(cat_tensor))
            
        # Concatenate categoricals and continuous variables
        embedded_tensor = torch.cat(embedded, dim=-1)
        x_full = torch.cat([embedded_tensor, x['encoder_cont']], dim=-1)
        
        # Project to Transformer hidden dimension
        x_proj = self.input_projector(x_full)
        
        # Pass through Transformer encoder
        enc_out = self.transformer_encoder(x_proj)
        
        # Flatten time and feature dimensions, project directly to forecast horizon
        enc_flat = enc_out.reshape(batch_size, -1)
        preds = self.output_layer(enc_flat)
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(y, (tuple, list)):
            y = y[0]
        preds = self(x)
        
        # y shape: (batch_size, prediction_window)
        if self.alpha < 1.0 and self.soft_targets is not None:
            # Distillation mode: extract group and time indices to get teacher forecasts
            group_ids = x['groups'][:, 0].long()
            start_times = x['decoder_time_idx'][:, 0].long()
            
            # Lookup teacher soft targets (move dynamically to device if needed)
            if self.soft_targets.device != self.device:
                self.soft_targets = self.soft_targets.to(self.device)
            teacher_preds = self.soft_targets[group_ids, start_times]
            
            # Compute losses
            loss_sup = self.loss_fn(preds, y)
            loss_dist = self.loss_fn(preds, teacher_preds)
            
            loss = self.alpha * loss_sup + (1.0 - self.alpha) * loss_dist
            self.log("train_loss_sup", loss_sup, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_loss_dist", loss_dist, on_step=False, on_epoch=True, prog_bar=True)
        else:
            # Supervised mode (Ablation Student without KD)
            loss = self.loss_fn(preds, y)
            
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(y, (tuple, list)):
            y = y[0]
        preds = self(x)
        
        # Validation is evaluated purely on ground-truth target
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

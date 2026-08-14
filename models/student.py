import torch
import torch.nn as nn
import lightning.pytorch as pl

class M5TransformerStudent(pl.LightningModule):
    def __init__(self, training_dataset, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, 
                 dropout=0.1, lr=1e-3, alpha=1.0, lookback_window=90, prediction_window=28, 
                 soft_targets=None, embedding_dim=8, output_head="flat_decoder_mlp",
                 output_head_hidden_dim=48, supervised_loss="huber"):
        """
        training_dataset: TimeSeriesDataSet used for shape configurations and encoders
        d_model: Transformer hidden dimension size
        nhead: Number of attention heads
        num_layers: Number of Transformer encoder layers
        dim_feedforward: Feed-forward network dimension in Transformer layers
        dropout: Dropout rate
        lr: Learning rate
        alpha: Supervised-loss weight (1.0 = purely supervised, 0.0 = purely distillation)
        lookback_window: L (number of lookback days)
        prediction_window: H (number of forecast days)
        soft_targets: Pre-computed teacher forecasts tensor of shape (num_groups, 1942, 28)
        embedding_dim: Embedding dimension size for categoricals
        output_head: Type of output projection head ('flat_decoder_mlp', 'flat_decoder', or 'step_wise')
        output_head_hidden_dim: Hidden dimension size for the 'flat_decoder_mlp' head
        """
        super().__init__()
        self.save_hyperparameters(ignore=['training_dataset', 'soft_targets'])
        self.training_dataset = training_dataset
        self.alpha = alpha
        self.lookback_window = lookback_window
        self.prediction_window = prediction_window
        if supervised_loss not in {"huber", "wrmsse_informed"}:
            raise ValueError(
                "supervised_loss must be 'huber' or 'wrmsse_informed', got "
                f"{supervised_loss!r}"
            )
        self.supervised_loss = supervised_loss
        
        # Store soft targets lookup tensor as a plain attribute to avoid saving it in checkpoints (6.6 GB)
        self.soft_targets = soft_targets

        # Categorical columns in the exact order PyTorch Forecasting stacks them
        self.cat_cols = training_dataset.categoricals
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=len(training_dataset._categorical_encoders[col].classes_) + 1,
                embedding_dim=embedding_dim
            ) for col in self.cat_cols
        ])
        
        self.total_cat_dim = len(self.cat_cols) * embedding_dim
        self.num_reals = len(training_dataset.reals)
        
        # Dynamically identify known continuous feature indices to avoid future leakage
        self.known_real_indices = [
            i for i, name in enumerate(training_dataset.reals)
            if name not in training_dataset.time_varying_unknown_reals
        ]
        self.num_known_reals = len(self.known_real_indices)
        
        self.enc_cont_norm = nn.LayerNorm(self.num_reals)
        self.dec_cont_norm = nn.LayerNorm(self.num_known_reals)
        
        # Project concatenated embeddings + continuous variables to d_model
        # Encoder uses all continuous features (reals); Decoder uses only future-known continuous features
        self.encoder_projector = nn.Sequential(
            nn.Linear(self.total_cat_dim + self.num_reals, d_model),
            nn.LayerNorm(d_model)
        )
        self.decoder_projector = nn.Sequential(
            nn.Linear(self.total_cat_dim + self.num_known_reals, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Configure output prediction head type
        self.output_head = output_head
        if output_head == "flat_decoder_mlp":
            self.output_layer = nn.Sequential(
                nn.Linear(self.prediction_window * d_model, output_head_hidden_dim),
                nn.ReLU(),
                nn.Linear(output_head_hidden_dim, self.prediction_window)
            )
        elif output_head == "flat_decoder":
            self.output_layer = nn.Linear(self.prediction_window * d_model, self.prediction_window)
        elif output_head == "step_wise":
            self.output_layer = nn.Linear(d_model, 1)
        else:
            raise ValueError(f"Unknown output head type: {output_head}")
            
        # Huber Loss (Smooth L1) for robustness
        self.loss_fn = nn.HuberLoss()

    def _point_loss(self, predictions, targets, coefficients=None, decoder_lengths=None):
        """Return the selected point objective reduced over batch and horizon.

        For ``wrmsse_informed`` the fixed per-series coefficient is broadcast
        over the 28 decoder steps.  This matches the teacher's
        ``WRMSSEInformedLossMetric`` semantics: coefficient times squared error,
        with no per-batch coefficient renormalization.  When decoder lengths
        are supplied, its numerator and denominator follow
        ``MultiHorizonMetric``: only valid decoder positions contribute.
        """
        if self.supervised_loss == "huber":
            return self.loss_fn(predictions, targets)

        if coefficients is None:
            raise RuntimeError(
                "WRMSSE-informed student loss requires per-series coefficients in every batch"
            )
        if coefficients.ndim != 1 or coefficients.shape[0] != predictions.shape[0]:
            raise ValueError(
                "WRMSSE-informed coefficient shape must be [batch]; got "
                f"{tuple(coefficients.shape)} for predictions {tuple(predictions.shape)}"
            )
        if not torch.isfinite(coefficients).all() or (coefficients < 0).any():
            raise ValueError("WRMSSE-informed coefficients must be finite and non-negative")
        weighted_error = torch.square(predictions - targets) * coefficients.unsqueeze(1)
        if decoder_lengths is None:
            return weighted_error.mean()
        if decoder_lengths.ndim != 1 or decoder_lengths.shape[0] != predictions.shape[0]:
            raise ValueError(
                "decoder_lengths must have shape [batch] when supplied; got "
                f"{tuple(decoder_lengths.shape)}"
            )
        if (decoder_lengths < 1).any() or (decoder_lengths > predictions.shape[1]).any():
            raise ValueError("decoder_lengths contains an invalid decoder horizon")
        valid = torch.arange(
            predictions.shape[1], device=predictions.device
        ).unsqueeze(0) < decoder_lengths.unsqueeze(1)
        return (weighted_error * valid).sum() / valid.sum()

    def forward(self, x):
        # x is batch[0] dict from PyTorch Forecasting dataloader
        batch_size = x['encoder_cat'].shape[0]
        
        # 1. Embed and project historical lookback inputs (encoder)
        enc_embedded = []
        for i, embed_layer in enumerate(self.embeddings):
            cat_tensor = x['encoder_cat'][:, :, i].long()
            # Clamp class values to prevent out-of-bounds index errors
            cat_tensor = torch.clamp(cat_tensor, 0, embed_layer.num_embeddings - 1)
            enc_embedded.append(embed_layer(cat_tensor))
        enc_embedded_tensor = torch.cat(enc_embedded, dim=-1)
        enc_cont_normed = self.enc_cont_norm(x['encoder_cont'])
        enc_full = torch.cat([enc_embedded_tensor, enc_cont_normed], dim=-1)
        enc_proj = self.encoder_projector(enc_full) # Shape: (batch_size, L, d_model)
        
        # 2. Embed and project future prediction window inputs (decoder)
        dec_embedded = []
        for i, embed_layer in enumerate(self.embeddings):
            cat_tensor = x['decoder_cat'][:, :, i].long()
            cat_tensor = torch.clamp(cat_tensor, 0, embed_layer.num_embeddings - 1)
            dec_embedded.append(embed_layer(cat_tensor))
        dec_embedded_tensor = torch.cat(dec_embedded, dim=-1)
        
        # Filter decoder continuous inputs to exclude unknown features (avoid future leakage)
        dec_cont_known = x['decoder_cont'][:, :, self.known_real_indices]
        dec_cont_normed = self.dec_cont_norm(dec_cont_known)
        dec_full = torch.cat([dec_embedded_tensor, dec_cont_normed], dim=-1)
        dec_proj = self.decoder_projector(dec_full) # Shape: (batch_size, H, d_model)
        
        # 3. Concatenate lookback and future windows along the time/sequence dimension
        x_seq = torch.cat([enc_proj, dec_proj], dim=1) # Shape: (batch_size, L + H, d_model)
        
        # 4. Pass through Transformer encoder (full self-attention across lookback & future)
        enc_out = self.transformer_encoder(x_seq) # Shape: (batch_size, L + H, d_model)
        
        # 5. Extract prediction window states and pass through the output prediction head
        dec_out = enc_out[:, -self.prediction_window:, :] # Shape: (batch_size, H, d_model)
        
        if self.output_head in ("flat_decoder_mlp", "flat_decoder"):
            dec_flat = dec_out.reshape(batch_size, -1)
            preds = self.output_layer(dec_flat)
        elif self.output_head == "step_wise":
            preds = self.output_layer(dec_out).squeeze(-1) # Shape: (batch_size, H)
            
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(y, (tuple, list)):
            y = y[0]
        preds = self(x)
        coefficients = x.get("wrmsse_informed_coefficient")
        decoder_lengths = x.get("decoder_lengths")
        
        # y shape: (batch_size, prediction_window)
        teacher_preds = x.get('soft_targets', None)
        if self.alpha < 1.0 and (self.soft_targets is not None or teacher_preds is not None):
            if teacher_preds is None:
                # Distillation mode: extract group and time indices to get teacher forecasts
                group_ids = x['groups'][:, 0].long()
                start_times = x['decoder_time_idx'][:, 0].long()
                
                # Lookup teacher soft targets (move dynamically to device if needed)
                if self.soft_targets.device != self.device:
                    self.soft_targets = self.soft_targets.to(self.device)
                teacher_preds = self.soft_targets[group_ids, start_times]
            
            # Fail loudly if any soft target is NaN / missing
            if not torch.isfinite(teacher_preds).all():
                raise RuntimeError("Missing or NaN teacher targets encountered in KD training batch. All eligible KD training samples must have complete teacher predictions.")

            # Compute losses
            loss_sup = self._point_loss(preds, y, coefficients, decoder_lengths)
            loss_dist = self._point_loss(preds, teacher_preds, coefficients, decoder_lengths)
            
            loss = self.alpha * loss_sup + (1.0 - self.alpha) * loss_dist
            self.log("train_loss_sup", loss_sup, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_loss_dist", loss_dist, on_step=False, on_epoch=True, prog_bar=True)
        else:
            # Supervised mode (Ablation Student without KD)
            loss = self._point_loss(preds, y, coefficients, decoder_lengths)
            
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(y, (tuple, list)):
            y = y[0]
        preds = self(x)
        coefficients = x.get("wrmsse_informed_coefficient")
        decoder_lengths = x.get("decoder_lengths")
        
        # Validation is evaluated purely on ground-truth target
        loss = self._point_loss(preds, y, coefficients, decoder_lengths)
        # The phase-v2 loader emits d1520..d1526. Batch-size weighting makes
        # the epoch metric their ground-truth objective mean. KD soft targets
        # are unavailable in validation and are intentionally never consulted.
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
            batch_size=preds.shape[0],
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

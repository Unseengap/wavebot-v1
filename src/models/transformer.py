"""Transformer sequence model for directional prediction.

Predicts directional bias (long/short/flat) for multi-horizon forecasting.
Also includes label generation and dataset utilities.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence models."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════
#  TRANSFORMER MODEL
# ═══════════════════════════════════════════════════════════════

class ForexTransformer(nn.Module):
    """Transformer encoder for forex directional prediction.

    Input:  [batch, seq_len (128), input_dim (~100 features)]
    Output: [batch, num_classes (3)] per horizon
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 3,
        num_horizons: int = 3,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_horizons = num_horizons

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Output heads — one per horizon
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes),
            )
            for _ in range(num_horizons)
        ])

    def forward(self, x: torch.Tensor, horizon: int = 0) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, input_dim]
            horizon: which horizon head to use (0, 1, or 2)

        Returns:
            logits: [batch, num_classes]
        """
        batch_size = x.size(0)

        # Project input to d_model dimensions
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Extract CLS token output (first position)
        cls_output = x[:, 0, :]

        # Apply output head for specified horizon
        logits = self.output_heads[horizon](cls_output)
        return logits

    def predict_all_horizons(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Predict for all horizons at once."""
        batch_size = x.size(0)

        x = self.input_proj(x)
        x = self.input_norm(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        cls_output = x[:, 0, :]

        return [head(cls_output) for head in self.output_heads]

    def predict_probs(self, x: torch.Tensor, horizon: int = 2) -> dict:
        """Get class probabilities for a single prediction.

        Returns dict like {"long": 0.72, "short": 0.15, "flat": 0.13}
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, horizon=horizon)
            probs = torch.softmax(logits, dim=-1)

        labels = ["long", "short", "flat"]
        if probs.dim() == 2:
            probs = probs[0]
        return {label: float(probs[i]) for i, label in enumerate(labels)}


# ═══════════════════════════════════════════════════════════════
#  LABEL GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_labels(
    df: pd.DataFrame,
    horizons: list[int] = None,
    threshold_atr_multiplier: float = 0.5,
) -> pd.DataFrame:
    """Generate directional labels for supervised training.

    Long:  future high exceeds threshold before future low hits stop
    Short: future low drops below threshold before future high hits stop
    Flat:  neither condition met within horizon

    Args:
        df: DataFrame with 'close', 'high', 'low', 'atr_14' columns.
        horizons: List of horizon bar counts (default [1, 3, 5]).
        threshold_atr_multiplier: Fraction of ATR used as threshold.

    Returns:
        DataFrame with label columns added.
    """
    if horizons is None:
        horizons = [1, 3, 5]

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr_14"].values

    n = len(df)

    for h in horizons:
        labels = np.full(n, "flat", dtype=object)

        for i in range(n - h):
            threshold = atr[i] * threshold_atr_multiplier
            if np.isnan(threshold) or threshold <= 0:
                continue

            entry = close[i]
            up_target = entry + threshold
            down_target = entry - threshold

            # Check future bars
            for j in range(i + 1, min(i + h + 1, n)):
                hit_up = high[j] >= up_target
                hit_down = low[j] <= down_target

                if hit_up and not hit_down:
                    labels[i] = "long"
                    break
                elif hit_down and not hit_up:
                    labels[i] = "short"
                    break
                elif hit_up and hit_down:
                    # Both hit on same bar — check which was closer to entry
                    up_dist = high[j] - entry
                    down_dist = entry - low[j]
                    labels[i] = "long" if up_dist >= down_dist else "short"
                    break

        df[f"label_h{h}"] = labels

    return df


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════

class ForexSequenceDataset(Dataset):
    """Sliding window dataset for Transformer training."""

    LABEL_MAP = {"long": 0, "short": 1, "flat": 2}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "label_h5",
        lookback: int = 128,
    ):
        self.features = df[feature_cols].values.astype(np.float32)
        raw_labels = df[label_col].values
        self.encoded_labels = np.array([self.LABEL_MAP.get(l, 2) for l in raw_labels])
        self.lookback = lookback

    def __len__(self):
        return max(0, len(self.features) - self.lookback)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx: idx + self.lookback])
        y = torch.tensor(self.encoded_labels[idx + self.lookback], dtype=torch.long)
        return x, y

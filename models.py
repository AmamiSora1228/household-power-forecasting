"""
Neural network models for multi-horizon time series forecasting.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMForecast(nn.Module):
    """Multi-layer LSTM encoder + fully-connected decoder."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, horizon: int = 90, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        # Use last layer hidden state
        out = h_n[-1]  # (batch, hidden_size)
        out = self.fc(out)  # (batch, horizon)
        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerForecast(nn.Module):
    """Encoder-only Transformer for time series forecasting."""

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 horizon: int = 90, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        # Aggregate sequence by mean pooling
        x = x.mean(dim=1)  # (batch, d_model)
        out = self.decoder(x)  # (batch, horizon)
        return out


class LagLinearForecast(nn.Module):
    """
    Lag Linear model -- the proposed improved model.

    Instead of flattening the entire history or using recurrent/attention
    structures, this model explicitly extracts a small set of interpretable
    lag features from the target series and concatenates them with the most
    recent covariates. A single linear layer maps this compact feature vector
    directly to the forecast horizon.

    Why it can win:
    - Recent daily lags capture short-term persistence.
    - Weekly lags directly encode the dominant periodicity of household power.
    - The linear mapping is the simplest possible predictor and cannot
      overfit the small training set the way larger networks can.
    - It uses all covariates at the forecast origin (calendar, weather) but
      avoids the redundant noise of the full 90-step flattened vector.
    """

    def __init__(self, input_size: int, horizon: int = 90,
                 seq_len: int = 90, dropout: float = 0.1,
                 target_idx: int = 0):
        super().__init__()
        self.target_idx = target_idx
        # Recent daily lags (1-14) + weekly lags up to ~12 weeks
        self.lags = list(range(1, 15)) + [21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
        n_lags = len(self.lags)
        self.dropout = nn.Dropout(dropout)
        # Linear from lags + current covariates -> horizon
        self.linear = nn.Linear(n_lags + input_size, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # Lag features of the target
        lag_feats = []
        for lag in self.lags:
            lag_feats.append(x[:, -lag, self.target_idx])
        lag_feats = torch.stack(lag_feats, dim=1)  # (batch, n_lags)
        # Most recent covariates
        curr = x[:, -1, :]  # (batch, input_size)
        feats = torch.cat([lag_feats, curr], dim=1)
        feats = self.dropout(feats)
        return self.linear(feats)

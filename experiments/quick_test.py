"""
Quick test: Bottleneck Temporal Encoder — simple, innovative.
Compress 90 days → small latent → expand to horizon.
The bottleneck forces the model to extract only essential patterns.
"""
import sys
sys.path.insert(0, '/storage/lcm_lab/sora/lab/work/code')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import *
from dataset import TimeSeriesDataset
from train import set_seed, train_model, evaluate_rolling


class BottleneckTemporalEncoder(nn.Module):
    """
    Bottleneck Temporal Encoder — innovative, not from any paper.

    Core idea: compress 90 days of multivariate data into a small set of
    latent vectors (e.g., 4 or 8), then expand back to the forecast horizon.
    The information bottleneck forces the model to discard noise and keep
    only the essential temporal patterns.

    Unlike Transformer which attends over all pairs of days (O(S²)),
    this is O(S) in complexity. Unlike LSTM which processes sequentially,
    this processes the whole sequence in parallel.

    Architecture:
    1. Day encoder: each day → hidden dim (shared across days)
    2. Learnable "anchor" queries attend over all days → K latent vectors
    3. Latent vectors expanded → horizon via MLP
    4. Residual: last day's value added back
    """
    def __init__(self, input_size, horizon=90, seq_len=90,
                 num_latent=8, hidden=128, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden = hidden
        self.num_latent = num_latent

        # Shared day encoder
        self.day_enc = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learnable anchor queries — what patterns to look for?
        self.anchors = nn.Parameter(torch.randn(1, num_latent, hidden) * 0.02)

        # Decoder: latent vectors → horizon
        self.decoder = nn.Sequential(
            nn.Linear(num_latent * hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

        # Residual: last observed target value → added to all predictions
        self.residual_scale = nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        B, S, D = x.shape

        # Encode each day
        h = self.day_enc(x)  # (B, S, hidden)

        # Anchors attend over days: (B, K, S) @ (B, S, hidden) → (B, K, hidden)
        anchors = self.anchors.expand(B, -1, -1)  # (B, K, hidden)
        scores = torch.bmm(anchors, h.transpose(1, 2))  # (B, K, S)
        scores = scores / (self.hidden ** 0.5)
        attn = F.softmax(scores, dim=-1)  # (B, K, S)
        latent = torch.bmm(attn, h)  # (B, K, hidden)

        # Decode
        out = self.decoder(latent.reshape(B, -1))  # (B, horizon)

        # Residual: add back last observed target value
        out = out + self.residual_scale * x[:, -1, 0:1]  # feature 0 = target

        return out


def quick_test():
    train_df = pd.read_csv(TRAIN_CSV, parse_dates=['date'], index_col='date')
    test_df = pd.read_csv(TEST_CSV, parse_dates=['date'], index_col='date')
    combined_df = pd.concat([train_df, test_df]).sort_index()
    input_features = list(train_df.columns)

    n_val = 270
    train_sub = train_df.iloc[:-n_val].copy()
    val_sub = train_df.iloc[-n_val:].copy()

    target_mean = float(train_df[TARGET_COL].mean())
    target_std = float(train_df[TARGET_COL].std())
    feature_mean = train_df[input_features].mean().values.copy()
    feature_std = train_df[input_features].std().values.copy()
    feature_std[feature_std == 0] = 1.0

    def normalize(df):
        df = df.copy()
        for i, col in enumerate(input_features):
            df[col] = (df[col].to_numpy() - feature_mean[i]) / feature_std[i]
        return df

    train_sub_n = normalize(train_sub)
    val_sub_n = normalize(val_sub)
    combined_df_n = normalize(combined_df)

    for horizon, label in [(HORIZON_SHORT, 'short'), (HORIZON_LONG, 'long')]:
        print(f'\n{"="*50}')
        print(f'{label}-term (horizon={horizon})')
        print(f'{"="*50}')

        if horizon == HORIZON_SHORT:
            origins = list(pd.date_range(start=SHORT_TEST_ORIGIN_START,
                                          end=SHORT_TEST_ORIGIN_END, freq='D'))
        else:
            origins = list(pd.date_range(start=LONG_TEST_ORIGIN_START,
                                          end=LONG_TEST_ORIGIN_END, freq='D'))

        val_horizon = min(horizon, 90)
        results = []

        for seed in SEEDS:
            set_seed(seed)
            train_ds = TimeSeriesDataset(
                train_sub_n, INPUT_LEN, horizon, input_features, TARGET_COL,
                target_mean, target_std)
            val_ds = TimeSeriesDataset(
                val_sub_n, INPUT_LEN, val_horizon, input_features, TARGET_COL,
                target_mean, target_std)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

            model = BottleneckTemporalEncoder(
                input_size=len(input_features), horizon=horizon,
                num_latent=8, hidden=128, dropout=DROPOUT).to(DEVICE)

            model, history, best_val = train_model(
                model, train_loader, val_loader, DEVICE,
                val_output_slice=val_horizon)

            eval_res = evaluate_rolling(
                model, combined_df_n, origins, INPUT_LEN, horizon,
                input_features, TARGET_COL, target_mean, target_std, DEVICE)

            results.append(eval_res)
            print(f'  Seed {seed}: MSE={eval_res["mse_mean"]:.2f} MAE={eval_res["mae_mean"]:.2f}')

        mses = [r['mse_mean'] for r in results]
        maes = [r['mae_mean'] for r in results]
        print(f'  Avg: MSE={np.mean(mses):.2f} ± {np.std(mses):.2f}, '
              f'MAE={np.mean(maes):.2f} ± {np.std(maes):.2f}')

        if label == 'short':
            print(f'  Target: Transformer MSE=494 ± 180, MAE=18.1 ± 3.8')
        else:
            print(f'  Target: Transformer MSE=814 ± 531, MAE=22.8 ± 9.5')


if __name__ == '__main__':
    quick_test()

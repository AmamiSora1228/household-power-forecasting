"""
Training and evaluation utilities.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

from config import (
    DEVICE, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
    HORIZON_SHORT, TARGET_COL, INPUT_LEN
)
from dataset import build_loaders, make_rolling_predictions
from models import LSTMForecast, TransformerForecast, LagLinearForecast


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, input_size: int, horizon: int, input_features: list = None):
    """Factory for models."""
    if model_name == 'LSTM':
        return LSTMForecast(
            input_size=input_size, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, horizon=horizon, dropout=DROPOUT
        )
    elif model_name == 'Transformer':
        return TransformerForecast(
            input_size=input_size, d_model=HIDDEN_SIZE, nhead=4,
            num_layers=NUM_LAYERS, dim_feedforward=HIDDEN_SIZE * 2,
            horizon=horizon, dropout=DROPOUT
        )
    elif model_name == 'LagLinear':
        # Linear model on daily/weekly lags + current covariates
        return LagLinearForecast(
            input_size=input_size, horizon=horizon, dropout=DROPOUT
        )
    else:
        raise ValueError(f'Unknown model: {model_name}')


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device: str):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model, loader: DataLoader, criterion, device: str, output_slice: int = None):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if output_slice is not None:
            out = out[:, :output_slice]
            y = y[:, :output_slice]
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, device: str,
                num_epochs: int = NUM_EPOCHS, patience: int = PATIENCE,
                lr: float = LEARNING_RATE, weight_decay: float = WEIGHT_DECAY,
                val_output_slice: int = None):
    """Train with early stopping and return best model + history."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=max(1, patience // 2)
    )

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device, output_slice=val_output_slice)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_loss


def evaluate_rolling(model, df, origins, input_len, horizon,
                     input_features, target_col, target_mean, target_std, device):
    """Evaluate model on rolling origins and return MSE/MAE per origin + aggregated."""
    preds_dict = make_rolling_predictions(
        model, df, origins, input_len, horizon,
        input_features, target_col, target_mean, target_std, device
    )
    mses, maes = [], []
    for origin, res in preds_dict.items():
        mse = mean_squared_error(res['target'], res['pred'])
        mae = mean_absolute_error(res['target'], res['pred'])
        mses.append(mse)
        maes.append(mae)
    return {
        'mse_per_origin': mses,
        'mae_per_origin': maes,
        'mse_mean': float(np.mean(mses)),
        'mae_mean': float(np.mean(maes)),
        'mse_std': float(np.std(mses)),
        'mae_std': float(np.std(maes)),
        'predictions': preds_dict
    }


def run_single_experiment(model_name: str, horizon: int, seed: int,
                          train_df: pd.DataFrame, val_df: pd.DataFrame,
                          test_df: pd.DataFrame, input_features: list,
                          target_mean: float, target_std: float,
                          origins: list, device: str = DEVICE):
    """Run one training/evaluation seed."""
    set_seed(seed)
    model = build_model(model_name, len(input_features), horizon, input_features)
    # Validation always uses horizon=90 to keep enough validation samples for the long task
    val_horizon = min(horizon, 90)
    train_loader, val_loader = build_loaders(
        train_df, val_df, input_len=INPUT_LEN,
        train_horizon=horizon, val_horizon=val_horizon,
        input_features=input_features, target_mean=target_mean, target_std=target_std,
        batch_size=BATCH_SIZE, target_col=TARGET_COL
    )

    val_output_slice = val_horizon
    model, history, best_val_loss = train_model(
        model, train_loader, val_loader, device=device,
        val_output_slice=val_output_slice
    )

    eval_res = evaluate_rolling(
        model, test_df, origins, INPUT_LEN, horizon,
        input_features, TARGET_COL, target_mean, target_std, device
    )
    eval_res['val_loss'] = best_val_loss
    eval_res['history'] = history
    return model, eval_res

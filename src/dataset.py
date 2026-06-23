"""
PyTorch dataset and dataloader builders for time series forecasting.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import TARGET_COL


class TimeSeriesDataset(Dataset):
    """Sliding-window time series dataset."""

    def __init__(self, df: pd.DataFrame, input_len: int, horizon: int,
                 input_features: list, target_col: str = TARGET_COL,
                 target_mean: float = 0.0, target_std: float = 1.0):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.input_features = input_features
        self.target_col = target_col

        self.inputs = df[input_features].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        self.target_mean = target_mean
        self.target_std = target_std

        self.n_samples = len(df) - input_len - horizon + 1
        if self.n_samples <= 0:
            raise ValueError(
                f'Not enough data for input_len={input_len}, horizon={horizon} (len={len(df)})'
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.inputs[idx:idx + self.input_len]
        y = self.targets[idx + self.input_len:idx + self.input_len + self.horizon]
        # The dataframe passed in is already normalized by run_experiments,
        # so we keep the target in that single-normalized scale for training.
        return torch.from_numpy(x), torch.from_numpy(y)


def build_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame,
                  input_len: int, train_horizon: int, val_horizon: int,
                  input_features: list, target_mean: float, target_std: float,
                  batch_size: int, target_col: str = TARGET_COL):
    """Create training and validation dataloaders with possibly different horizons."""
    train_ds = TimeSeriesDataset(
        train_df, input_len, train_horizon, input_features, target_col,
        target_mean, target_std
    )
    val_ds = TimeSeriesDataset(
        val_df, input_len, val_horizon, input_features, target_col,
        target_mean, target_std
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def make_rolling_predictions(model, df: pd.DataFrame, origins: list,
                             input_len: int, horizon: int,
                             input_features: list, target_col: str,
                             target_mean: float, target_std: float,
                             device: str):
    """
    Generate rolling-origin predictions on a continuous dataframe.
    Returns a dict mapping origin date -> (preds, targets).
    """
    model.eval()
    values = df[input_features].values.astype(np.float32)
    target_values = df[target_col].values.astype(np.float32)
    date_index = df.index
    results = {}

    with torch.no_grad():
        for origin in pd.to_datetime(origins):
            start_idx = df.index.get_loc(origin)
            input_idx = start_idx - input_len
            if input_idx < 0:
                raise ValueError(f'Origin {origin.date()} requires input before data start.')
            x = values[input_idx:start_idx]
            x = torch.from_numpy(x).unsqueeze(0).to(device)
            pred = model(x).cpu().numpy().squeeze(0)  # (horizon,) in normalized scale
            pred = pred * target_std + target_mean    # denormalize to original scale
            tgt = target_values[start_idx:start_idx + horizon]
            tgt = tgt * target_std + target_mean      # denormalize target to original scale
            pred_dates = date_index[start_idx:start_idx + horizon]
            results[origin] = {
                'pred': pred,
                'target': tgt,
                'dates': pred_dates
            }
    return results

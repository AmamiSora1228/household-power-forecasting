"""
Main entry point: run all models on short-term and long-term forecasting tasks.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    TRAIN_CSV, TEST_CSV, RESULTS_DIR,
    INPUT_LEN, HORIZON_SHORT, HORIZON_LONG,
    SHORT_TEST_ORIGIN_START, SHORT_TEST_ORIGIN_END,
    LONG_TEST_ORIGIN_START, LONG_TEST_ORIGIN_END,
    SEEDS, TARGET_COL, DEVICE
)
from train import run_single_experiment, set_seed


def load_data():
    train_df = pd.read_csv(TRAIN_CSV, parse_dates=['date'], index_col='date')
    test_df = pd.read_csv(TEST_CSV, parse_dates=['date'], index_col='date')
    combined_df = pd.concat([train_df, test_df]).sort_index()
    return train_df, test_df, combined_df


def split_train_val(train_df: pd.DataFrame, val_size: int = 270):
    """Use last val_size days of training set as validation."""
    n_val = min(val_size, len(train_df) // 3)
    train_sub = train_df.iloc[:-n_val].copy()
    val_sub = train_df.iloc[-n_val:].copy()
    return train_sub, val_sub


def get_input_features(df: pd.DataFrame):
    """Use all columns as input features, including the lagged target."""
    return list(df.columns)


def get_origins(start: str, end: str, horizon: int):
    """Generate daily origin dates between start and end (inclusive)."""
    dates = pd.date_range(start=start, end=end, freq='D')
    return list(dates)


def aggregate_metrics(results: list):
    """Aggregate metrics across seeds."""
    mses = [r['mse_mean'] for r in results]
    maes = [r['mae_mean'] for r in results]
    return {
        'mse_mean': float(np.mean(mses)),
        'mse_std': float(np.std(mses)),
        'mae_mean': float(np.mean(maes)),
        'mae_std': float(np.std(maes)),
    }


def main():
    train_df, test_df, combined_df = load_data()
    train_sub, val_sub = split_train_val(train_df, val_size=270)
    input_features = get_input_features(train_df)

    # Normalize using training data statistics (make explicit copies to avoid read-only arrays)
    target_mean = float(train_df[TARGET_COL].mean())
    target_std = float(train_df[TARGET_COL].std())
    feature_mean = np.array(train_df[input_features].mean().values, copy=True)
    feature_std = np.array(train_df[input_features].std().values, copy=True)
    feature_std[feature_std == 0] = 1.0

    def normalize(df):
        df = df.copy()
        for i, col in enumerate(input_features):
            df[col] = (df[col].to_numpy() - feature_mean[i]) / feature_std[i]
        return df

    train_sub_n = normalize(train_sub)
    val_sub_n = normalize(val_sub)
    combined_df_n = normalize(combined_df)

    # Save normalization params
    norm_params = {
        'feature_mean': feature_mean.tolist(),
        'feature_std': feature_std.tolist(),
        'target_mean': float(target_mean),
        'target_std': float(target_std),
        'input_features': input_features
    }
    with open(RESULTS_DIR / 'norm_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    short_origins = get_origins(SHORT_TEST_ORIGIN_START, SHORT_TEST_ORIGIN_END, HORIZON_SHORT)
    long_origins = get_origins(LONG_TEST_ORIGIN_START, LONG_TEST_ORIGIN_END, HORIZON_LONG)

    print(f'Input features ({len(input_features)}): {input_features}')
    print(f'Short-term test origins: {len(short_origins)}')
    print(f'Long-term test origins: {len(long_origins)}')

    all_results = {}

    for horizon in [HORIZON_SHORT, HORIZON_LONG]:
        horizon_label = 'short' if horizon == HORIZON_SHORT else 'long'
        origins = short_origins if horizon == HORIZON_SHORT else long_origins
        print(f'\n{"="*60}')
        print(f'Running {horizon_label}-term forecasting (horizon={horizon})')
        print(f'{"="*60}')

        for model_name in ['LSTM', 'Transformer', 'LagLinear']:
            print(f'\nModel: {model_name}')
            seed_results = []
            for seed in SEEDS:
                print(f'  Seed {seed}...', end='', flush=True)
                _, eval_res = run_single_experiment(
                    model_name=model_name,
                    horizon=horizon,
                    seed=seed,
                    train_df=train_sub_n,
                    val_df=val_sub_n,
                    test_df=combined_df_n,
                    input_features=input_features,
                    target_mean=target_mean,
                    target_std=target_std,
                    origins=origins,
                    device=DEVICE
                )
                seed_results.append(eval_res)
                print(f' MSE={eval_res["mse_mean"]:.4f} MAE={eval_res["mae_mean"]:.4f}')

            agg = aggregate_metrics(seed_results)
            print(f'  Aggregated -> MSE: {agg["mse_mean"]:.4f} ± {agg["mse_std"]:.4f}, '
                  f'MAE: {agg["mae_mean"]:.4f} ± {agg["mae_std"]:.4f}')

            all_results[f'{horizon_label}_{model_name}'] = {
                'aggregate': agg,
                'seeds': [
                    {
                        'mse_mean': r['mse_mean'],
                        'mae_mean': r['mae_mean'],
                        'mse_std': r['mse_std'],
                        'mae_std': r['mae_std'],
                        'val_loss': r['val_loss'],
                        'history': r['history']
                    }
                    for r in seed_results
                ],
                'predictions': seed_results[0]['predictions']  # keep first seed for plotting
            }

    # Save results
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        # predictions contain DataFrames; convert dates to strings
        save_dict = {}
        for k, v in all_results.items():
            save_dict[k] = {
                'aggregate': v['aggregate'],
                'seeds': v['seeds']
            }
        json.dump(save_dict, f, indent=2)

    # Save full results with predictions as pickle
    with open(RESULTS_DIR / 'results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print('\nAll experiments complete. Results saved to', RESULTS_DIR)


if __name__ == '__main__':
    main()

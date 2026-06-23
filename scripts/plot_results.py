"""
Generate figures from experimental results.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_DIR, HORIZON_SHORT, HORIZON_LONG

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 9


def load_results():
    with open(RESULTS_DIR / 'results.pkl', 'rb') as f:
        return pickle.load(f)


def plot_training_curves(results, save_dir: Path):
    """Plot training/validation loss curves for the first seed of each model."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for horizon_label in ['short', 'long']:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharey=True)
        for ax, model_name in zip(axes, ['LSTM', 'Transformer', 'LagLinear']):
            key = f'{horizon_label}_{model_name}'
            # History was not saved in aggregated results; skip if absent
            if key not in results or 'history' not in results[key].get('seeds', [{}])[0]:
                ax.set_title(f'{model_name}\n(no history)')
                ax.axis('off')
                continue
            history = results[key]['seeds'][0]['history']
            ax.plot(history['train_loss'], label='Train')
            ax.plot(history['val_loss'], label='Val')
            ax.set_title(model_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.legend()
        fig.suptitle(f'{horizon_label.capitalize()}-term Training Curves')
        plt.tight_layout()
        fig.savefig(save_dir / f'training_curves_{horizon_label}.png', bbox_inches='tight')
        plt.close(fig)


def plot_prediction_examples(results, save_dir: Path, n_examples: int = 3):
    """Plot predicted vs ground-truth curves for representative origins."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for horizon_label in ['short', 'long']:
        for model_name in ['LSTM', 'Transformer', 'LagLinear']:
            key = f'{horizon_label}_{model_name}'
            preds = results[key]['predictions']
            origins = list(preds.keys())
            step = max(1, len(origins) // n_examples)
            chosen = [origins[i * step] for i in range(n_examples)]

            fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 3), sharey=True)
            if n_examples == 1:
                axes = [axes]
            for ax, origin in zip(axes, chosen):
                res = preds[origin]
                dates = res['dates']
                ax.plot(dates, res['target'], label='Ground Truth', color='black', linewidth=1.5)
                ax.plot(dates, res['pred'], label='Prediction', color='tab:red', linewidth=1.5)
                ax.set_title(f'Origin {origin.strftime("%Y-%m-%d")}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Global Active Power (kW)')
                ax.tick_params(axis='x', rotation=30)
            axes[0].legend()
            fig.suptitle(f'{model_name} — {horizon_label.capitalize()}-term Predictions')
            plt.tight_layout()
            fig.savefig(save_dir / f'pred_{horizon_label}_{model_name}.png', bbox_inches='tight')
            plt.close(fig)


def plot_metric_comparison(results, save_dir: Path):
    """Bar chart comparing MSE and MAE across models."""
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for horizon_label in ['short', 'long']:
        for model_name in ['LSTM', 'Transformer', 'LagLinear']:
            key = f'{horizon_label}_{model_name}'
            agg = results[key]['aggregate']
            rows.append({
                'Horizon': 'Short-term (90d)' if horizon_label == 'short' else 'Long-term (365d)',
                'Model': model_name,
                'MSE': agg['mse_mean'],
                'MSE_std': agg['mse_std'],
                'MAE': agg['mae_mean'],
                'MAE_std': agg['mae_std']
            })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ['MSE', 'MAE']):
        pivot = df.pivot(index='Horizon', columns='Model', values=metric)
        err = df.pivot(index='Horizon', columns='Model', values=f'{metric}_std')
        pivot.plot(kind='bar', ax=ax, yerr=err, capsize=3, rot=0)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.legend(title='Model')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_dir / 'metric_comparison.png', bbox_inches='tight')
    plt.close(fig)

    # Save numeric table
    df.to_csv(save_dir / 'metrics_table.csv', index=False)
    print('Metrics table saved to', save_dir / 'metrics_table.csv')


def main():
    results = load_results()
    plot_dir = RESULTS_DIR / 'figures'
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_comparison(results, plot_dir)
    plot_prediction_examples(results, plot_dir, n_examples=3)
    # Training curves require history; they are generated only if history is stored
    try:
        plot_training_curves(results, plot_dir)
    except Exception as e:
        print('Skipping training curves:', e)

    print('Figures saved to', plot_dir)


if __name__ == '__main__':
    main()

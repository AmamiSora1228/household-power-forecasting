"""
Configuration for household power consumption time series forecasting.
"""
import os
from pathlib import Path

# Paths
ROOT = Path('/storage/lcm_lab/sora/lab/work')
DATA_DIR = ROOT / 'data'
CODE_DIR = ROOT / 'code'
RESULTS_DIR = ROOT / 'results'
OVERLEAF_DIR = ROOT / 'overleaf'

RAW_POWER_FILE = DATA_DIR / 'household_power_consumption.txt'
RAW_WEATHER_FILE = DATA_DIR / 'MENSQ_92_previous-1950-2024.csv.gz'
TRAIN_CSV = RESULTS_DIR / 'train.csv'
TEST_CSV = RESULTS_DIR / 'test.csv'

# Date ranges
train_start = '2007-01-01'
train_end = '2009-05-31'   # inclusive
test_start = '2009-06-01'
test_end = '2010-11-26'    # inclusive (last available date)

# Forecasting horizons
HORIZON_SHORT = 90
HORIZON_LONG = 365
INPUT_LEN = 90

# Features
SUM_COLS = ['global_active_power', 'global_reactive_power', 'sub_metering_1', 'sub_metering_2']
MEAN_COLS = ['voltage', 'global_intensity']
WEATHER_COLS = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
TARGET_COL = 'global_active_power'

# Model common hyperparameters
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 150
PATIENCE = 15

# Seeds for repeated experiments
SEEDS = [42, 2024, 123, 7, 99]

# Test origin windows (inclusive)
# Short-term: input fully inside test, output inside test
SHORT_TEST_ORIGIN_START = '2009-08-30'
SHORT_TEST_ORIGIN_END = '2010-08-28'

# Long-term: output fully inside test; input may overlap train for early origins
LONG_TEST_ORIGIN_START = '2009-06-01'
LONG_TEST_ORIGIN_END = '2009-11-26'

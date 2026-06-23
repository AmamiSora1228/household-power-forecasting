"""
Preprocess raw household power consumption data into daily train/test CSVs.

The power data was collected from a household in Sceaux, France. Sceaux is
located in the Hauts-de-Seine department, whose official French department
number is 92. Therefore we use the MENSQ_92 weather file to obtain monthly
meteorological features that are spatially consistent with the household.
"""
import gzip
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pandas as pd
import numpy as np

from config import (
    RAW_POWER_FILE, RAW_WEATHER_FILE, TRAIN_CSV, TEST_CSV,
    train_start, train_end, test_start, test_end,
    SUM_COLS, MEAN_COLS, WEATHER_COLS
)

warnings.filterwarnings('ignore')


def load_power_data(path: str) -> pd.DataFrame:
    """Load minute-level household power data."""
    df = pd.read_csv(path, sep=';', na_values=['?'], low_memory=False)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.drop(columns=['Date', 'Time'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def load_weather_data(path: str) -> pd.DataFrame:
    """Load monthly weather data and keep needed columns.

    The household is in Sceaux, Hauts-de-Seine (French department no. 92).
    Station IDs in this department start with '92', so we filter by that prefix
    and use the first available station.
    """
    with gzip.open(path, 'rt') as f:
        weather = pd.read_csv(f, sep=';', low_memory=False)
    # Keep stations in department 92 (Sceaux / Hauts-de-Seine)
    weather = weather[weather['NUM_POSTE'].astype(str).str.startswith('92')].copy()
    # Use the first station in the filtered list
    weather = weather[weather['NUM_POSTE'] == weather['NUM_POSTE'].iloc[0]].copy()
    weather['year'] = weather['AAAAMM'] // 100
    weather['month'] = weather['AAAAMM'] % 100
    # RR is in 0.1 mm; convert to mm
    weather['RR'] = weather['RR'] / 10.0
    keep = ['year', 'month'] + WEATHER_COLS
    weather = weather[keep].copy()
    # Fill missing weather values with 0 (missing is normal)
    weather[WEATHER_COLS] = weather[WEATHER_COLS].fillna(0)
    return weather


def aggregate_daily(power: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute-level power data to daily level."""
    power = power.copy()
    power['date'] = power['datetime'].dt.date
    power['date'] = pd.to_datetime(power['date'])

    # Columns to lowercase for consistency with config
    rename_map = {
        'Global_active_power': 'global_active_power',
        'Global_reactive_power': 'global_reactive_power',
        'Voltage': 'voltage',
        'Global_intensity': 'global_intensity',
        'Sub_metering_1': 'sub_metering_1',
        'Sub_metering_2': 'sub_metering_2',
        'Sub_metering_3': 'sub_metering_3',
    }
    power = power.rename(columns=rename_map)

    # Forward-fill small gaps in numeric columns before aggregation
    numeric_cols = SUM_COLS + MEAN_COLS + ['sub_metering_3']
    power[numeric_cols] = power[numeric_cols].ffill(limit=60)

    daily = []
    for dt, group in power.groupby('date'):
        row = {'date': dt}
        for col in SUM_COLS:
            row[col] = group[col].sum(skipna=True)
        for col in MEAN_COLS:
            row[col] = group[col].mean(skipna=True)
        row['sub_metering_3'] = group['sub_metering_3'].sum(skipna=True)
        daily.append(row)

    daily = pd.DataFrame(daily)
    daily = daily.set_index('date').asfreq('D')
    daily.index.name = 'date'

    # Interpolate any fully missing days (linear) and fill remaining edges
    daily = daily.interpolate(method='linear', limit_direction='both')
    daily = daily.ffill().bfill()

    # Compute remainder energy (Wh) per day
    daily['sub_metering_remainder'] = (
        daily['global_active_power'] * 1000.0 / 60.0
        - (daily['sub_metering_1'] + daily['sub_metering_2'] + daily['sub_metering_3'])
    )
    return daily


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add date-based calendar features."""
    daily = daily.copy()
    daily['year'] = daily.index.year
    daily['month'] = daily.index.month
    daily['day'] = daily.index.day
    daily['dayofweek'] = daily.index.dayofweek
    daily['dayofyear'] = daily.index.dayofyear
    daily['is_weekend'] = (daily['dayofweek'] >= 5).astype(int)
    return daily


def merge_weather(daily: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge monthly weather features into daily data."""
    daily = daily.reset_index()
    tmp = pd.DataFrame({
        'merge_year': daily['date'].dt.year,
        'merge_month': daily['date'].dt.month
    })
    daily = pd.concat([daily, tmp], axis=1)
    weather = weather.rename(columns={'year': 'merge_year', 'month': 'merge_month'})
    daily = daily.merge(weather, on=['merge_year', 'merge_month'], how='left')
    daily[WEATHER_COLS] = daily[WEATHER_COLS].fillna(0)
    daily = daily.drop(columns=['merge_year', 'merge_month'])
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.set_index('date')
    return daily


def split_train_test(daily: pd.DataFrame):
    """Split daily data into train and test by date."""
    train = daily.loc[train_start:train_end].copy()
    test = daily.loc[test_start:test_end].copy()
    return train, test


def main():
    print('Loading raw power data...')
    power = load_power_data(RAW_POWER_FILE)
    print(f'Loaded {len(power)} minute records.')

    print('Loading weather data...')
    weather = load_weather_data(RAW_WEATHER_FILE)
    print(f'Loaded {len(weather)} monthly weather records.')

    print('Aggregating to daily...')
    daily = aggregate_daily(power)
    daily = add_calendar_features(daily)
    daily = merge_weather(daily, weather)

    print('Splitting train/test...')
    train, test = split_train_test(daily)

    # Save
    train.to_csv(TRAIN_CSV)
    test.to_csv(TEST_CSV)
    print(f'Train shape: {train.shape}, saved to {TRAIN_CSV}')
    print(f'Test shape: {test.shape}, saved to {TEST_CSV}')
    print('Columns:', list(train.columns))
    print('Date range train:', train.index.min().date(), '->', train.index.max().date())
    print('Date range test:', test.index.min().date(), '->', test.index.max().date())


if __name__ == '__main__':
    main()

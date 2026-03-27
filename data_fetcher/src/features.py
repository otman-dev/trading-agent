import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()
    # Use oil returns as target
    df['oil_return'] = df['oil'].pct_change()
    df['target'] = df['oil_return'].shift(-1)

    # Create returns for other features
    for col in ['oil', 'dxy', 'sp500', 'fed_funds']:
        if col in df.columns:
            df[f'{col}_return'] = df[col].pct_change()
            df[f'{col}_return_lag1'] = df[f'{col}_return'].shift(1)
            df[f'{col}_return_lag2'] = df[f'{col}_return'].shift(2)

    # Volatility
    for col in ['oil', 'dxy']:
        if col in df.columns:
            df[f'{col}_vol_5d'] = df[col].pct_change().rolling(5).std()
            df[f'{col}_vol_20d'] = df[col].pct_change().rolling(20).std()

    # Ratios involving oil
    if 'dxy' in df.columns and 'oil' in df.columns:
        df['oil_dxy_ratio'] = df['oil'] / df['dxy']

    # SMA and ratio
    for col in ['oil', 'dxy']:
        if col in df.columns:
            df[f'{col}_sma_20'] = df[col].rolling(20).mean()
            df[f'{col}_ratio_sma'] = df[col] / df[f'{col}_sma_20']

    df.dropna(inplace=True)
    return df

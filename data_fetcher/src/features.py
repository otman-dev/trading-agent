import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()
    df['gold_return'] = df['gold'].pct_change()
    df['target'] = df['gold_return'].shift(-1)

    for col in ['oil', 'dxy', 'sp500', 'fed_funds']:
        if col in df.columns:
            df[f'{col}_return'] = df[col].pct_change()
            df[f'{col}_return_lag1'] = df[f'{col}_return'].shift(1)
            df[f'{col}_return_lag2'] = df[f'{col}_return'].shift(2)

    for col in ['gold', 'oil', 'dxy']:
        if col in df.columns:
            df[f'{col}_vol_5d'] = df[col].pct_change().rolling(5).std()
            df[f'{col}_vol_20d'] = df[col].pct_change().rolling(20).std()

    if 'oil' in df.columns and 'gold' in df.columns:
        df['gold_oil_ratio'] = df['gold'] / df['oil']
    if 'dxy' in df.columns and 'gold' in df.columns:
        df['gold_dxy_ratio'] = df['gold'] / df['dxy']

    for col in ['gold', 'oil', 'dxy']:
        if col in df.columns:
            df[f'{col}_sma_20'] = df[col].rolling(20).mean()
            df[f'{col}_ratio_sma'] = df[col] / df[f'{col}_sma_20']

    df.dropna(inplace=True)
    return df

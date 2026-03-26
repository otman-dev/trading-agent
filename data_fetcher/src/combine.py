import pandas as pd
from .fred import fetch_fred_long_term
from .alpaca import fetch_alpaca_recent

def create_unified_dataset(df_fred, df_alpaca):
    mapping = {'GC=F': 'gold', 'CL=F': 'oil', 'DX-Y.NYB': 'dxy', 'SPY': 'sp500'}
    df_alpaca = df_alpaca.rename(columns=mapping)
    # Use the last common date
    last_common = min(df_fred.index[-1], df_alpaca.index[0])
    df_fred_hist = df_fred[df_fred.index <= last_common]
    df_alpaca_recent = df_alpaca[df_alpaca.index > last_common]
    df_unified = pd.concat([df_fred_hist, df_alpaca_recent])
    return df_unified

def get_recent_data(days_back=90):
    df_fred = fetch_fred_long_term(start_date=(pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d'))
    df_alpaca = fetch_alpaca_recent(['GC=F','CL=F','DX-Y.NYB','SPY'], days_back=days_back)
    if df_fred.empty or df_alpaca.empty:
        return pd.DataFrame()
    return create_unified_dataset(df_fred, df_alpaca)

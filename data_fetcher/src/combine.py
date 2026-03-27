import pandas as pd
from .fred import fetch_fred_long_term
from .alpaca import fetch_alpaca_recent

def create_unified_dataset(df_fred, df_alpaca):
    # Alpaca columns: we assume 'CL=F' maps to 'oil'
    mapping = {'CL=F': 'oil'}
    if not df_alpaca.empty:
        df_alpaca = df_alpaca.rename(columns=mapping)
        # Find common date range
        last_common = min(df_fred.index[-1], df_alpaca.index[0])
        df_fred_hist = df_fred[df_fred.index <= last_common]
        df_alpaca_recent = df_alpaca[df_alpaca.index > last_common]
        return pd.concat([df_fred_hist, df_alpaca_recent])
    else:
        return df_fred

def get_recent_data(days_back=90):
    # FRED data
    df_fred = fetch_fred_long_term(
        start_date=(pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')
    )
    # Try Alpaca for recent oil
    try:
        df_alpaca = fetch_alpaca_recent(['CL=F'], days_back=days_back)
        if not df_alpaca.empty:
            return create_unified_dataset(df_fred, df_alpaca)
    except Exception as e:
        print(f"Alpaca error: {e}")
    # Fallback: use FRED only
    print("Using FRED only data")
    return df_fred

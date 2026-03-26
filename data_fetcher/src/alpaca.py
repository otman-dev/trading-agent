import os
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

api = tradeapi.REST(
    os.environ.get('ALPACA_API_KEY_DATA', ''),
    os.environ.get('ALPACA_SECRET_KEY_DATA', ''),
    base_url='https://paper-api.alpaca.markets'
)

def fetch_alpaca_recent(symbols, days_back=30):
    end = datetime.now()
    start = end - timedelta(days=days_back)
    data = {}
    for sym in symbols:
        try:
            bars = api.get_bars(sym, tradeapi.TimeFrame.Day, start=start.isoformat(), end=end.isoformat()).df
            data[sym] = bars['close']
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
    df = pd.DataFrame(data)
    return df

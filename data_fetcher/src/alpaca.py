import os
import pandas as pd
from datetime import datetime, timedelta

def fetch_alpaca_recent(symbols=None, days_back=30):
    if symbols is None:
        symbols = ['CL=F']
    api_key = os.environ.get('ALPACA_API_KEY_DATA', '')
    secret_key = os.environ.get('ALPACA_SECRET_KEY_DATA', '')
    if not api_key or api_key == 'your_data_api_key':
        print("Alpaca data keys missing or placeholder – skipping data fetch.")
        return pd.DataFrame()
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(api_key, secret_key, base_url='https://paper-api.alpaca.markets')
        end = datetime.now()
        start = end - timedelta(days=days_back)
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        data = {}
        for sym in symbols:
            try:
                bars = api.get_bars(sym, tradeapi.TimeFrame.Day, start=start_str, end=end_str).df
                data[sym] = bars['close']
                print(f"✓ Fetched {sym} from Alpaca")
            except Exception as e:
                print(f"Error fetching {sym}: {e}")
        df = pd.DataFrame(data)
        return df
    except ImportError:
        print("alpaca-trade-api not installed.")
        return pd.DataFrame()

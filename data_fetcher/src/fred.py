import os
import pandas as pd
from fredapi import Fred
from datetime import datetime

fred = Fred(api_key=os.environ.get('FRED_API_KEY', ''))

def fetch_fred_long_term(start_date='1990-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    series = {
        'gold': 'GOLDPMGBD228NLBM',
        'oil': 'DCOILWTICO',
        'dxy': 'DTWEXBGS',
        'fed_funds': 'FEDFUNDS',
        't10y_real': 'T10YIE',
        'sp500': 'SP500',
        'cpi': 'CPIAUCSL',
        'm2': 'M2SL',
        'recession_prob': 'RECPROUSM156N'
    }
    data = {}
    for name, sid in series.items():
        try:
            s = fred.get_series(sid, start=start_date, end=end_date)
            data[name] = s
            print(f"✓ Fetched {name}")
        except Exception as e:
            print(f"✗ Failed {name}: {e}")
    df = pd.DataFrame(data).resample('D').ffill().dropna(how='all')
    return df

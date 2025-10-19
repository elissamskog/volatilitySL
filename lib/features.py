import numpy as np, pandas as pd

def minute_returns(price: pd.Series) -> pd.Series:
    return np.log(price).diff()

def realized_vol_from_returns(ret: pd.Series, H: int) -> pd.Series:
    return ret.pow(2).rolling(H, min_periods=H).sum().pow(0.5)

def make_minute_bars(trade: pd.DataFrame) -> pd.DataFrame:
    df = trade.copy()
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], unit='ns', errors='coerce')
        if ts.isna().all():
            ts = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df['minute'] = ts.dt.floor('T')
    elif 'seconds_in_bucket' in df.columns:
        base = pd.Timestamp("2000-01-01")
        # Ensure day and ses are Series so downstream astype() works consistently.
        if 'date_id' in df:
            day = df['date_id'].rank(method='dense').fillna(1) - 1
        else:
            day = pd.Series(0, index=df.index)
        if 'time_id' in df:
            ses = df['time_id'].rank(method='dense').fillna(1) - 1
        else:
            ses = pd.Series(0, index=df.index)

        idx = (day.astype(int) * 10000 + ses.astype(int) * 1000 + (df['seconds_in_bucket'] // 60))
        df['minute'] = base + pd.to_timedelta(idx, unit='m')
    else:
        raise ValueError("Need 'timestamp' or 'seconds_in_bucket'")

    price_col = 'price' if 'price' in df else 'wap'
    size_col  = 'size'  if 'size'  in df else ('volume' if 'volume' in df else None)
    if size_col is None:
        df['__ones'] = 1.0; size_col = '__ones'

    bars = (df.groupby('minute')
              .agg(price=(price_col,'last'),
                   vol=(size_col,'sum'),
                   trades=(price_col,'count'))
              .reset_index())
    return bars.sort_values('minute').reset_index(drop=True)

def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df['ret'] = minute_returns(df['price'])
    for H in [1,5,10,30,60]:
        df[f'rv_{H}'] = realized_vol_from_returns(df['ret'], H)
    for L in [1,5,10,30,60]:
        df[f'lag_rv_{L}'] = df[f'rv_{L}'].shift(1)
    df['log_vol'] = np.log1p(df['vol'].fillna(0))
    df['trades'] = df['trades'].fillna(0)
    df['minute_of_day'] = df['minute'].dt.hour*60 + df['minute'].dt.minute
    df['dow'] = df['minute'].dt.dayofweek
    return df
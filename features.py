import pandas as pd
from data_pull import pull_data, read_csv_file

def add_label(df):
    df = df.copy()
    df["label"] = df["Close"].shift(-1) > df["Close"]
    df["label"] = df["label"].astype(int)
    df = df.iloc[:-1]
    return df

def add_features(df):
    df = df.copy()
    df["return_1_day"] = df["Close"].pct_change(1)
    df["return_5_days"] = df["Close"].pct_change(5)
    df["ma5"] = df["Close"].rolling(window=5).mean()
    df["ma20"] = df["Close"].rolling(window=20).mean()
    df["ma_ratio"] = df["ma5"]/df["ma20"]
    df["volatility_10d"] = df["return_1_day"].rolling(window=10).std()
    delta = df["Close"].diff()
    df["gains"] = delta.clip(lower=0)
    df["losses"] = delta.clip(upper=0) * (-1)
    df["avg_gain"] = df["gains"].rolling(window=14).mean()
    df["avg_loss"] = df["losses"].rolling(window=14).mean()
    df["RS"] = df["avg_gain"]/df["avg_loss"]
    df["RSI"] = 100 - (100 / (1 + df["RS"]))
    df["volume_change"] = df["Volume"].pct_change(1)
    return df

def build_feature_set(df):
    featured = add_label(add_features(df))
    featured = featured[["label", "return_1_day", 
                         "return_5_days", "ma_ratio", 
                         "volatility_10d", "RSI", 
                         "volume_change"]].dropna()
    return featured

df = read_csv_file("AAPL", "5y", "1d")
result = build_feature_set(df)
print(result.head(10))
print(result.shape)
print(result.isna().sum()) 

from features import build_feature_set
from data_pull import read_csv_file
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def chronological_split(df, test_frac=0.2):
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

def scale_for_quantum(train, test, feature_cols):
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    return X_train, X_test, scaler

if __name__ == "__main__":
    df = read_csv_file("AAPL", "5y", "1d")
    result = build_feature_set(df)
    train, test = chronological_split(result)
    feature_cols = [col for col in result.columns if col != "label"]
    X_train, X_test, scaler = scale_for_quantum(train, test, feature_cols)
    print(X_train.min(), X_train.max())
    print(X_test.min(), X_test.max())

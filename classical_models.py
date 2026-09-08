from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from data_pull import read_csv_file, pull_data
from features import build_feature_set
from preprocess import scale_for_quantum, chronological_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from evaluate import evaluate
import pandas as pd
import numpy as np


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def run_pipeline(ticker):
    try:
        df = read_csv_file(ticker, "5y", "1d")
    except FileNotFoundError:
        df = pull_data(ticker, "5y", "1d")

    result = build_feature_set(df)
    train, test = chronological_split(result)
    feature_cols = [col for col in result.columns if col != "label"]
    y_train = train["label"].values
    y_test = test["label"].values
    X_train, X_test, scaler = scale_for_quantum(train, test, feature_cols)

    logistic_regression = train_logistic_regression(X_train, y_train)
    svm = train_svm(X_train, y_train)
    random_forest = train_random_forest(X_train, y_train)

    models = {
        "Logistic Regression": logistic_regression,
        "SVM": svm,
        "Random Forest": random_forest,
    }

    results = {}
    for name, model in models.items():
        results[name] = evaluate(model, X_test, y_test)

    return results


if __name__ == "__main__":
    tickers = ["AAPL", "TSLA", "JNJ", "SPY"]
    rows = []

    for ticker in tickers:
        results = run_pipeline(ticker)
        for model_name, metrics in results.items():
            row = {"ticker": ticker, "model": model_name, **metrics}
            rows.append(row)

    comparison = pd.DataFrame(rows)
    print(comparison)

    
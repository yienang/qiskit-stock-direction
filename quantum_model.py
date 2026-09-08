from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from data_pull import read_csv_file, pull_data
from features import build_feature_set
from preprocess import scale_for_quantum, chronological_split
from evaluate import evaluate
import pandas as pd
import numpy as np
import time

def build_feature_map(num_features):
    return zz_feature_map(feature_dimension=num_features, reps=2)

def build_ansatz(num_features):
    return real_amplitudes(num_qubits = num_features, reps=2) 

def train_vqc(X_train, y_train, num_features, maxiter=30):
    feature_map = build_feature_map(num_features)
    ansatz = build_ansatz(num_features)
    optimizer = COBYLA(maxiter=maxiter)
    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
    vqc.fit(X_train, y_train)
    return vqc

# Used for multi-ticker comparisons (see README for results)
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

    start = time.time()
    model = train_vqc(X_train, y_train, num_features=6, maxiter=30)
    elapsed = time.time() - start
    print(f"Training took {elapsed:.1f} seconds")

    metrics = evaluate(model, X_test, y_test)
    metrics["train_time_sec"] = elapsed
    return metrics


if __name__ == "__main__":
    df = read_csv_file("AAPL", "5y", "1d")
    result = build_feature_set(df)
    train, test = chronological_split(result)
    feature_cols = [col for col in result.columns if col != "label"]
    y_train = train["label"].values
    X_train, X_test, scaler = scale_for_quantum(train, test, feature_cols)

    model = train_vqc(X_train, y_train, num_features=6, maxiter=30)
    print(model.weights)
    np.save("vqc_weights.npy", model.weights)
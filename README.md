# VQC Stock Direction Classifier

A Quantum Neural Network (VQC) built with Qiskit to predict next-day stock
price direction (up/down), benchmarked against classical ML baselines
(Logistic Regression, SVM, Random Forest) across multiple stocks, with a
final evaluation on real IBM quantum hardware.

## Summary

This project builds a complete, working pipeline — from raw price data to
a trained quantum classifier — and answers an honest empirical question:
**does a variational quantum classifier find real predictive signal in
short-term stock direction, and does it perform any differently than
classical models on the same task?**

The result: **no** — across 4 stocks (AAPL, TSLA, JNJ, SPY) and 4 models
(VQC, Logistic Regression, SVM, Random Forest), every model's ROC AUC
clustered tightly around 0.5 (random-guessing territory), with no
consistent gap between the quantum model and its classical counterparts.
This is consistent with the well-documented difficulty of predicting
short-term price direction, and suggests the limiting factor here is the
problem/feature set rather than any specific model architecture.

## Pipeline

1. **Data pull** (`data_pull.py`) — daily OHLCV via `yfinance`
2. **Feature engineering** (`features.py`) — 6 features: 1-day and 5-day
   returns, moving-average ratio, 10-day volatility, RSI, volume change —
   plus a binary label (next-day up/down)
3. **Preprocessing** (`preprocess.py`) — chronological (non-shuffled)
   train/test split, MinMaxScaler scaling to [0, π] for quantum angle
   encoding
4. **Classical baselines** (`classical_models.py`) — Logistic Regression,
   SVM (RBF kernel), Random Forest
5. **Quantum model** (`quantum_model.py`) — `ZZFeatureMap` (reps=2) +
   `RealAmplitudes` ansatz (reps=2), trained via `VQC` with COBYLA
   (maxiter=30)
6. **Real hardware evaluation** (`hardware_eval.py`) — the trained model's
   parameters, run on `ibm_marrakesh` via Qiskit Runtime, compared against
   a simulator run of the same circuits

## Results

### Single-ticker comparison (AAPL)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| VQC (quantum) | 0.52–0.56 | 0.43–0.57 | 0.47–0.56 | ~0.51 |
| Logistic Regression | 0.557 | 0.724 | 0.630 | 0.515 |
| SVM (RBF) | 0.550 | 0.948 | 0.696 | 0.481 |
| Random Forest | 0.563 | 0.604 | 0.583 | 0.537 |

*(VQC metrics vary somewhat between runs due to random ansatz
initialization and COBYLA's optimization path — reported as a range
across multiple runs.)*

### Multi-ticker comparison (AAPL, TSLA, JNJ, SPY)

| Ticker | VQC AUC | LogReg AUC | SVM AUC | RF AUC |
|---|---|---|---|---|
| AAPL | 0.469 | 0.515 | 0.481 | 0.524 |
| TSLA | 0.485 | 0.513 | 0.498 | 0.485 |
| JNJ | 0.521 | 0.523 | 0.463 | 0.458 |
| SPY | 0.519 | 0.419 | 0.586 | 0.432 |

Every value across every ticker and model sits roughly between 0.42 and
0.59 — no model, quantum or classical, shows meaningful separation
between "up" and "down" days on this feature set.

### Real hardware vs. simulator

The trained VQC's ansatz parameters were used to build fixed circuits for
10 test rows, submitted to `ibm_marrakesh` (real IBM hardware) and
compared against an Aer simulator run of the same circuits.

Both distributions were broadly spread across most of the 64 possible
6-bit measurement outcomes (63/64 on real hardware, 44/64 on simulator),
with no single outcome dominating either — consistent with a model that
has not converged to a confident decision boundary at this level of
training (`maxiter=30`), rather than a clean signature of hardware noise
specifically. Disentangling "hardware noise" from "an unconverged model"
would need a model trained with more optimizer iterations, run as a
follow-up.

## Key takeaways

- The full pipeline works correctly end to end, including real quantum
  hardware — data → features → quantum encoding → training → evaluation
  → hardware validation.
- On this feature set, **no tested model finds real signal** in next-day
  stock direction — a legitimate, well-supported negative result, not a
  failure of the implementation.
- Classical and quantum models perform comparably (all near AUC 0.5),
  suggesting the constraint is the prediction task itself, not the choice
  of classical vs. quantum architecture.

## Possible future work

- Train the VQC for longer (`maxiter` > 30) to see whether AUC improves
  meaningfully, or whether it plateaus near random regardless
- Try a richer or different feature set (e.g. longer-horizon labels,
  fundamental data) to see if *any* model can find signal
- Repeat the hardware-vs-simulator comparison using a model trained to
  convergence, to more cleanly isolate hardware noise from an
  under-trained model
- Apply the error-mitigation techniques explored in QGSS26 to the
  real-hardware evaluation

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run stages in order: `data_pull.py` → `features.py` → `preprocess.py` →
`classical_models.py` → `quantum_model.py` → `hardware_eval.py`
(requires an IBM Quantum account for the last stage).
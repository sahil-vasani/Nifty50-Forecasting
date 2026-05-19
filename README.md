# NIFTY50 Forecasting & Sectoral Contagion Analysis

> **Course:** Applied Forecasting Methods (IT402) — MSc. Data Science  
> **Institution:** Dhirubhai Ambani University (Formerly DA-IICT)  
> **Supervisor:** Dr. Pritam Anand

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Practical Applications](#practical-applications)
- [Limitations & Future Work](#limitations--future-work)
- [Authors](#authors)

---

## Overview

This project builds a **multi-horizon forecasting system** for the NIFTY 50 index — India's benchmark stock market index comprising the 50 largest companies listed on the NSE. The system predicts daily returns with calibrated uncertainty estimates and includes a **Sectoral Contagion Analysis** framework that models how volatility shocks propagate across sectors (Banking, IT, Pharma, Auto, FMCG, Metal, Energy) to the broader market.

The study benchmarks the **full spectrum of forecasting approaches** — from classical statistical models to state-of-the-art deep learning architectures — providing a rigorous, reproducible comparison on real Indian market data.

---

## Problem Statement

Build a multi-horizon forecasting model for NIFTY 50 that:
1. Predicts next-day and multi-day returns (point forecasts)
2. Provides reliable uncertainty estimates (probabilistic / interval forecasts)
3. Quantifies sector-to-market contagion effects

---

## Dataset

**Source:** NIFTY 50 daily OHLCV data + sectoral index returns

### Raw Features

| Feature | Description |
|---|---|
| `date` | Trading day timestamp |
| `open` | Price at market open |
| `high` | Highest intraday price |
| `low` | Lowest intraday price |
| `close` | Final closing price |
| `volume` | Total shares traded |
| `bank_ret` | Banking sector index return |
| `it_ret` | IT sector index return |
| `pharma_ret` | Pharmaceutical sector return |
| `auto_ret` | Automobile sector return |
| `fmcg_ret` | FMCG sector return |
| `metal_ret` | Metal sector return |
| `energy_ret` | Energy sector return |

### Engineered Features

| Feature | Rationale |
|---|---|
| `ret_1` | Latest return — strongest short-term directional signal |
| `vol_5` | Recent uncertainty; volatility regime indicator |
| `trend_strength` | Short-term vs long-term MA difference; bullish/bearish signal |
| `momentum` | Recent price acceleration |
| `bank_ret_lag1` (etc.) | Lagged sector returns for detecting spillover effects |
| `high_vol` | Binary flag: stable vs turbulent market phase |
| `corr_nifty_bank_ret` | NIFTY–Banking sector rolling correlation |
| `ret_5`, `ret_15`, `ma5`, `ma20` | Additional momentum and trend indicators |
| `target` | Next-day log return `r_{t+1}` — prediction target |

Return series: `r_t = ln(P_t / P_{t-1})`

---

## Models Implemented

### Classical Statistical Models

| Model | Purpose |
|---|---|
| AR(1) | Autoregressive — past values only |
| MA(1) | Moving Average — past errors only |
| ARMA(1,1) | Combined AR + MA |
| ARIMA(1,0,1) | Integrated ARMA (d=0, series already stationary) |
| ARIMAX(1,0,1) | ARIMA with exogenous sector variables |
| SARIMA | Seasonal ARIMA for periodic patterns |
| SARIMAX | Seasonal ARIMA with exogenous variables |

### Volatility Model

| Model | Purpose |
|---|---|
| GARCH(1,1) | Conditional variance estimation, VaR construction |

### Deep Learning Models

| Model | Approaches |
|---|---|
| RNN | Point forecasting, multi-horizon, quantile (two-model & single) |
| LSTM | Point forecasting, multi-horizon, quantile (two-model & single) |
| GRU | Seq2Seq, Seq2Seq + Attention |
| Transformer | Encoder-decoder with positional encoding; point + interval |

### Hybrid

- **LSTM + GARCH**: Combines deep learning return predictions with statistically rigorous volatility-based intervals.

---

## Results Summary

### Point Forecasting

| Model | RMSE | R² | Tier |
|---|---|---|---|
| AR / MA / ARMA / ARIMA | 0.00827 | −0.0025 | Classical |
| ARIMAX(1,0,1) | 0.00847 | −0.0509 | Classical |
| SARIMA / SARIMAX | ≈0.0083 | < 0 | Classical |
| RNN (Multi-step) | — | ≈ 0.28 | Deep Learning |
| LSTM (Multi-step) | 0.00369 | 0.367 | Deep Learning |
| GRU (Seq2Seq) | 0.01024 | 0.6597 | Deep Learning |
| **Transformer** | **0.00883** | **0.7479** | **Best** |

### Probabilistic Forecasting (95% CI)

| Model | PICP | MPIW | Winkler Score |
|---|---|---|---|
| AR–ARIMA (all) | 0.9838 | 0.04490 | — |
| GARCH(1,1) | 0.9877 | 0.05852 | 0.06097 |
| GRU (Seq2Seq + Attention) | 0.9665 | 0.04814 | 0.05347 |
| **Transformer** | **0.9733** | **0.02887** | **0.03149** |

> The **Transformer** achieves the best R² (0.7479), narrowest prediction intervals (MPIW = 0.02887), and best Winkler Score (0.03149) across all models.

---

## Project Structure

```
Nifty50-Forecasting/
│
├── data/
│   ├── raw/                    # Raw OHLCV + sectoral index data
│   └── processed/              # Feature-engineered dataset
│
├── models/
│   ├── classical/
│   │   ├── arima_family.py     # AR, MA, ARMA, ARIMA, ARIMAX
│   │   └── sarima.py           # SARIMA, SARIMAX
│   ├── volatility/
│   │   └── garch.py            # GARCH(1,1) implementation
│   └── deep_learning/
│       ├── rnn.py              # RNN — 4 experimental approaches
│       ├── lstm.py             # LSTM — 4 experimental approaches
│       ├── gru.py              # GRU — Seq2Seq & Seq2Seq+Attention
│       └── transformer.py      # Encoder-Decoder Transformer
│
├── requirements.txt
└── README.md
```

---

## Installation & Requirements

### Prerequisites

- Python 3.9+
- pip or conda

### Install Dependencies

```bash
git clone https://github.com/<your-repo>/Nifty50-Forecasting.git
cd Nifty50-Forecasting
pip install -r requirements.txt
```

### Core Dependencies

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels          # ARIMA, SARIMA, GARCH
arch                 # GARCH(1,1)
torch                # RNN, LSTM, GRU, Transformer
pytorch-forecasting  # Optional: high-level forecasting API
```

---

## Usage

### 1. Data Preparation

```python
# Load and engineer features
from data.feature_engineering import prepare_dataset

df = prepare_dataset("data/raw/nifty50.csv")
```

### 2. Run Classical Models

```python
from models.classical.arima_family import run_arima_pipeline

results = run_arima_pipeline(df, order=(1, 0, 1))
```

### 3. Run GARCH

```python
from models.volatility.garch import fit_garch

vol_model = fit_garch(df['ret_1'], p=1, q=1)
```

### 4. Run Deep Learning Models

```python
from models.deep_learning.transformer import TransformerForecaster

model = TransformerForecaster(input_dim=20, d_model=64, nhead=4, num_layers=2)
model.fit(train_loader, epochs=30)
predictions = model.predict(test_loader)
```

### 5. Evaluate

```python
from utils.metrics import evaluate_point, evaluate_interval

point_metrics   = evaluate_point(y_true, y_pred)      # RMSE, MAE, R²
interval_metrics = evaluate_interval(y_true, lower, upper)  # PICP, MPIW, Winkler
```

---

## Key Findings

1. **Classical models fail on returns.** AR through SARIMAX all produce negative R² — the return series is near white noise, consistent with the weak-form Efficient Market Hypothesis.

2. **GARCH is essential for volatility.** GARCH(1,1) achieves PICP = 98.77% and is the gold standard for VaR and dynamic interval construction, but cannot predict return direction.

3. **Deep learning unlocks predictive power.** RNN (R² ≈ 0.28) → LSTM (R² ≈ 0.37) → GRU (R² = 0.66) → Transformer (R² = 0.75), with each architectural advance providing measurable gains.

4. **Transformer achieves the best results overall** — highest R² (0.7479), lowest RMSE (0.00883), narrowest intervals (MPIW = 0.02887), and best Winkler Score.

5. **Multi-horizon > single-step.** All deep learning models show dramatically better R² under multi-horizon training. Single-step next-day prediction is dominated by noise.

6. **Quantile models enable risk management.** Single-model quantile approaches achieve strong calibration (80% CI coverage ≈ 84%). The LSTM + GARCH hybrid further refines intervals by combining neural predictions with statistical volatility dynamics.

7. **Attention does not always help.** On this dataset, GRU Seq2Seq (R² = 0.66) outperforms Seq2Seq + Attention (R² = 0.46). High-noise financial data and limited sample size prevent attention from learning reliable alignment weights.

---

## Practical Applications

| Use Case | Model Recommended | Benefit |
|---|---|---|
| **Portfolio Management** | Transformer (point forecasts) | Directional signal for tactical allocation |
| **Algorithmic Trading** | LSTM / GRU (multi-horizon) | 3–5 day momentum/mean-reversion strategies |
| **Risk Management (VaR)** | GARCH / LSTM+GARCH | Dynamic daily VaR at 5th percentile |
| **Sectoral Contagion** | ARIMAX / LSTM with lagged sector features | Early-warning for cross-sector shock propagation |
| **Regulatory Stress Testing** | Transformer (interval forecasts) | Calibrated PICP ≥ 95% reportable to regulators |

---

## Stationarity Test Results

The ADF test confirms the NIFTY 50 return series is stationary (d = 0):

| Metric | Value |
|---|---|
| ADF Statistic | −11.45 |
| p-value | ≈ 0.000 |
| 1% Critical Value | −3.43 |
| Conclusion | Series is stationary, d = 0 |

---

## Limitations & Future Work

- **Intraday data**: Extending to tick-level data could enable high-frequency applications.
- **Alternative architectures**: Temporal Fusion Transformer (TFT) and N-BEATS may provide further gains.
- **Regime detection**: Incorporating Hidden Markov Models to explicitly detect bull/bear/crisis regimes.
- **Crisis robustness**: Stress-test all models on COVID-19 crash period (March 2020) separately.
- **Ensemble methods**: A Transformer + LSTM + GARCH ensemble could combine the strengths of all three model families.
- **Extended sectoral coverage**: Include global indices (S&P 500, Hang Seng) as additional exogenous signals.

---

## Authors

| Name | Student ID |
|---|---|
| Gaurang Jadav | 202518012 |
| Angel Manoj | 202518033 |
| Pronnati Tapaswi | 202518052 |
| Sahil Vasani | 202518059 |

**Program:** MSc. Data Science  
**Course:** Applied Forecasting Methods (IT402)  
**Supervisor:** Dr. Pritam Anand  
**Institution:** Dhirubhai Ambani University (Formerly DA-IICT)

---
 
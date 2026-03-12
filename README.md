# Kalshi Prediction Market Analysis
# Prediction Market Price Forecasting - Kalshi Binary Options

> Forecasting short-term price movements on Kalshi prediction markets using time series analysis and machine learning, across macroeconomic and sports contracts.

**Team:** Adi Singh, Alberto Avila, Junny Choi, Manuel Arce  

---

## Overview

[Kalshi](https://kalshi.com) is a regulated prediction market platform where contracts pay $1 if an event occurs and $0 if it does not. Contract prices reflect the market's implied probability of the event happening.

This project builds and benchmarks a suite of forecasting models to predict short-term price movements across 5 high-volume Kalshi markets, covering macroeconomic events (Fed rate decisions, GDP, CPI, recession probability) and a sports outcome (Cricket T20 World Cup).

The core challenge: prediction market prices are bounded between 0 and 1, exhibit near-white-noise return dynamics, and contain irregular trade intervals — making standard time series approaches non-trivial.

---

## Markets Analyzed

| Ticker | Description | Bars | Price Range |
|---|---|---|---|
| FED-25MAY-T4.25 | Will the federal funds rate be above 4.25% in May 2025? | 4,069 | [0.23, 0.98] |
| RECSSNBER-25 | Will the US enter a recession in 2025? | 13,056 | [0.01, 0.73] |
| KXGDP-25APR30-T0.0 | Will GDP grow by April 2025? | 1,460 | [0.20, 0.83] |
| KXCPI-25JUL-T0.2 | Will CPI inflation be above 0.2% by July 2025? | 953 | [0.15, 0.95] |
| KXT20WORLDCUP-26-IND | Will India win the Cricket World Cup Final vs New Zealand? | 1,196 | [0.15, 0.55] |

---

## Project Pipeline

```
Raw API Data → Cleaning & Normalization → OHLCV Aggregation → Feature Engineering → Modeling → Evaluation
```

### 1. Data Collection
Raw trade-level data retrieved from the Kalshi public API, including trade price, volume, taker side, and timestamps across all 5 markets.

### 2. Data Cleaning
- Converted `yes_price` from cents to probabilities (0.01–0.99 scale)
- Parsed timestamps to standardized UTC
- Renamed `count` → `volume`; derived buy/sell indicators from `taker_side`
- Calculated days to expiration from settlement time
- All preprocessing handled in a single reusable pipeline: `normalise_api_trades()`

### 3. Time Series Aggregation
Individual trades aggregated into **hourly OHLCV bars**:
- Open, High, Low, Close prices
- Total volume, buy volume, sell volume
- VWAP and order flow (buy volume − sell volume)

### 4. Feature Engineering
40+ features engineered across five categories:

| Category | Features |
|---|---|
| Price & Volume | OHLCV, VWAP, buy/sell volume, order flow |
| Returns | 1/5/10/20-period returns, log returns |
| Technical Indicators | SMA, EMA, MACD, RSI, Bollinger Bands |
| Volatility & Momentum | Rolling volatility, momentum indicators |
| Target | Next-period price, return, direction |

### 5. Modeling
70/30 chronological train/test split. Six models benchmarked:

- **Deep Learning:** LSTM, GRU, Transformer
- **Statistical:** ARIMA, BSTS
- **Gradient Boosting:** XGBoost

### 6. Evaluation
Walk-forward backtesting using MAPE, RMSE, MAE, and Directional Accuracy.

---

## Results

| Market | Best Model | MAPE | RMSE | MAE |
|---|---|---|---|---|
| Fed funds rate (May 2025) | XGBoost | 3.78% | 0.0505 | 0.0334 |
| Recession in 2025 | BSTS | 138.44% | 0.0330 | 0.0311 |
| GDP growth (Apr 2025) | Transformer | 3.10% | 0.0341 | 0.0217 |
| CPI inflation (Jul 2025) | Transformer | 9.61% | 0.0544 | 0.0439 |
| Cricket T20 World Cup | XGBoost | 4.60% | 0.0231 | 0.0176 |

**Key findings:**
- XGBoost and Transformer consistently outperformed simpler baselines by better capturing short-term fluctuations
- ARIMA and BSTS produced flatter predictions, struggling with discrete price changes
- ADF stationarity tests confirmed most markets are non-stationary (p > 0.05), with the exception of the Cricket market (p = 0.0003)
- ACF/PACF analysis revealed near-white-noise return dynamics across all series, confirming that feature engineering beyond raw price is essential

---

## Key Insights

- **Bounded prices require special handling:** sigmoid activations in deep learning models and logit transforms for statistical models
- **Time-to-expiration matters:** volatility increases and prices converge to 0 or 1 near settlement
- **Microstructure features add signal:** order flow and VWAP provide predictive value beyond price alone
- **Walk-forward backtesting is essential** to prevent look-ahead bias in time series evaluation

---

## Setup

```bash
git clone https://github.com/asingh49-cmd/kalshi-forecasting
cd kalshi-forecasting
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- pandas, numpy, scikit-learn
- PyTorch (LSTM, GRU, Transformer)
- xgboost, statsmodels
- matplotlib, seaborn

---

## Repo Structure

```
├── data/                  # Raw and processed market data
├── notebooks/             # EDA, decomposition, ACF/PACF analysis
├── src/
│   ├── preprocessing.py   # normalise_api_trades() pipeline
│   ├── features.py        # Feature engineering
│   ├── models/            # LSTM, GRU, Transformer, XGBoost, ARIMA, BSTS
│   └── evaluate.py        # Walk-forward backtesting, metrics
├── results/               # Model outputs and comparison plots
└── README.md
```

---

## References
- Kalshi Public API: https://kalshi.com/docs

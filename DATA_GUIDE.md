# Data Guide — Pulling & Cleaning Kalshi Prediction Market Data

This guide walks through how we fetch raw trade data from the Kalshi API, clean and normalize it, aggregate it into OHLCV bars, and explore it in a Jupyter notebook.

---

## 1. Prerequisites

```bash
pip install httpx pandas numpy matplotlib seaborn scikit-learn
```

You need a **Kalshi API key**. Get one from your Kalshi account settings.

---

## 2. How the Kalshi API Works

Kalshi is a regulated prediction market (CFTC-regulated exchange). Every contract is a binary option priced between **$0.01 and $0.99**, representing the market's implied probability of an event.

The API lives at:
```
https://api.elections.kalshi.com/trade-api/v2
```

### Key Endpoints

| Endpoint | What It Returns |
|----------|----------------|
| `GET /markets` | List of all markets (with volume, ticker, title, expiration) |
| `GET /markets?series_ticker=KXFED` | Markets filtered by series (e.g., all Fed rate markets) |
| `GET /markets/trades?ticker=FED-25MAY-T4.25` | Individual trades for a specific market |

### Authentication

All requests use a Bearer token in the header:
```
Authorization: Bearer YOUR_API_KEY
```

### Rate Limits

Kalshi allows ~16 requests/second. Our client enforces a 60ms minimum gap between requests and handles `429 Too Many Requests` with exponential backoff.

### Pagination

All list endpoints use **cursor-based pagination**. Each response includes a `cursor` field — pass it back to get the next page. Page size is capped at 200 items.

---

## 3. Pulling the Data

### Step 1: Connect to the API

```python
from src.api_client import KalshiClient

API_KEY = "your-api-key-here"
client = KalshiClient(api_key=API_KEY)
```

### Step 2: Browse Available Markets

```python
# Get all markets in a series
fed_markets = client.get_markets(series_ticker="KXFED")

# Sort by volume to find the most active
fed_markets.sort(key=lambda m: m.get("volume", 0), reverse=True)

for m in fed_markets[:5]:
    print(f"{m['ticker']:40s}  vol={m['volume']:>10,}  {m['title'][:50]}")
```

Output:
```
FED-25MAY-T4.25                           vol= 3,745,708  Will the upper bound of the federal funds rate be...
FED-25MAY-T4.00                           vol= 2,691,205  Will the upper bound of the federal funds rate be...
FED-25SEP-T4.25                           vol= 2,182,903  Will the upper bound of the federal funds rate be...
```

### Step 3: Fetch Trades

```python
# Fetch ALL trades for a market (automatically paginates)
trades = client.get_trades("FED-25MAY-T4.25")
print(f"Got {len(trades)} trades")
```

Each trade looks like:
```python
{
    "trade_id": "abc123",
    "ticker": "FED-25MAY-T4.25",
    "created_time": "2025-01-15T14:32:05.123Z",
    "yes_price": 65,          # in cents (1-99)
    "no_price": 35,
    "count": 50,              # number of contracts
    "taker_side": "yes",      # who initiated the trade
}
```

### Step 4: Save to DataFrame

```python
import pandas as pd

df = pd.DataFrame(trades)
df.to_csv("data/raw/kalshi_trades.csv", index=False)
print(f"Saved {len(df)} trades")
```

### Our Approach: Top 10 Diverse Markets

Instead of picking markets manually, our `data_collector.py` automates this:

1. Scans 15 different series (Fed, CPI, GDP, payrolls, unemployment, recession, cricket, golf, weather, etc.)
2. Picks the **single highest-volume market** from each series
3. Ranks across series and takes the **top 10**
4. Fetches **all** trades for each (no cap)

This ensures diversity — markets from different categories have genuinely different dynamics, which makes cross-market analysis meaningful.

```python
from src.data_collector import fetch_api_data

# This does everything: scan series, rank, fetch trades, save CSV
df = fetch_api_data(top_n=10)
```

---

## 4. Understanding the Raw Data

```python
df = pd.read_csv("data/raw/kalshi_trades.csv")
df.info()
```

### Raw Columns

| Column | Type | Description |
|--------|------|-------------|
| `created_time` | string (ISO 8601) | When the trade happened |
| `yes_price` | int (1-99) | Price in cents for "yes" outcome |
| `no_price` | int (1-99) | Price in cents for "no" outcome (= 100 - yes_price) |
| `count` | int | Number of contracts traded |
| `taker_side` | string | `"yes"` or `"no"` — who initiated the trade |
| `market_ticker` | string | Unique market identifier |
| `series_ticker` | string | Series the market belongs to |
| `market_title` | string | Human-readable description |
| `expiration_time` | string (ISO 8601) | When the market expires/settles |

### Quick Look

```python
print(f"Total trades: {len(df):,}")
print(f"Markets: {df['market_ticker'].nunique()}")
print(f"Date range: {df['created_time'].min()} to {df['created_time'].max()}")

# Trades per market
df.groupby('market_ticker').size().sort_values(ascending=False)
```

---

## 5. Cleaning & Normalizing

The raw API data needs several transformations before analysis.

### Step 1: Normalize Prices

`yes_price` is in **cents** (1-99). We convert to a probability scale [0.01, 0.99]:

```python
df["price"] = df["yes_price"] / 100.0
```

### Step 2: Parse Timestamps

```python
df["timestamp"] = pd.to_datetime(df["created_time"], utc=True)
```

### Step 3: Rename Volume

```python
df["volume"] = df["count"]  # 'count' = number of contracts
```

### Step 4: Compute Buy/Sell Indicator

```python
df["is_buy"] = (df["taker_side"] == "yes").astype(int)
```

### Step 5: Compute Days to Expiration

```python
exp = pd.to_datetime(df["expiration_time"], utc=True)
df["days_to_expiration"] = ((exp - df["timestamp"]).dt.total_seconds() / 86400).clip(lower=0)
```

### All at Once

Our `preprocessor.py` does all of this in one call:

```python
from src.preprocessor import normalise_api_trades

df_clean = normalise_api_trades(df)
```

---

## 6. Aggregating to OHLCV Bars

Raw trades are irregular — sometimes 100 trades/minute, sometimes none for hours. We aggregate into regular **hourly OHLCV bars**:

```python
from src.preprocessor import trades_to_ohlcv

# For a single market
market_trades = df_clean[df_clean["market_id"] == "FED-25MAY-T4.25"]
ohlcv = trades_to_ohlcv(market_trades, freq="1h")
ohlcv.head()
```

### What the Aggregation Produces

| Column | How It's Computed |
|--------|-------------------|
| `open` | First trade price in the hour |
| `high` | Max trade price in the hour |
| `low` | Min trade price in the hour |
| `close` | Last trade price in the hour |
| `volume` | Sum of contracts traded |
| `vwap` | Volume-Weighted Average Price |
| `buy_volume` | Contracts traded by "yes" takers |
| `sell_volume` | Contracts traded by "no" takers |
| `order_flow` | `buy_volume - sell_volume` (net buying pressure) |

Gaps (hours with no trades) are **forward-filled** — the last known price carries forward.

### All Markets at Once

```python
from src.preprocessor import prepare_dataset

datasets = prepare_dataset(df, freq="1h")
# Returns: {"FED-25MAY-T4.25": DataFrame, "KXCPI-25JUL-T0.2": DataFrame, ...}

for name, market_df in datasets.items():
    print(f"{name:45s} {len(market_df):>5} bars  "
          f"price=[{market_df['close'].min():.3f}, {market_df['close'].max():.3f}]")
```

---

## 7. Exploring in a Notebook

### Full Example Notebook

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load & clean ──────────────────────────────────────────
from src.data_collector import load_data
from src.preprocessor import prepare_dataset

raw = load_data(try_api=False)  # loads cached CSV
datasets = prepare_dataset(raw, freq="1h")

print(f"Markets loaded: {len(datasets)}")
```

### Plot All Markets

```python
fig, ax = plt.subplots(figsize=(14, 6))
for name, df in datasets.items():
    ax.plot(df.index, df["close"], label=name, alpha=0.8)
ax.set_ylabel("Price (probability)")
ax.set_title("Kalshi Market Prices")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
```

### Pick the Best Market for Analysis

Not all markets are equal. Some have settled (stuck at $0.01 or $0.99) and are useless for forecasting. We score by `bars x price_std x test_std`:

```python
import numpy as np

scores = {}
for name, df in datasets.items():
    n = len(df)
    std = df["close"].std()
    test_std = df["close"].iloc[int(0.85 * n):].std()
    scores[name] = n * std * (test_std + 0.001)

best = max(scores, key=scores.get)
print(f"Best market: {best} (score={scores[best]:.1f})")
df_primary = datasets[best]
```

### Price Distribution

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(df_primary["close"], bins=50, edgecolor="black")
axes[0].set_title("Price Distribution")
axes[0].set_xlabel("Price")

returns = df_primary["close"].pct_change().dropna()
axes[1].hist(returns, bins=50, edgecolor="black")
axes[1].set_title("Return Distribution")
axes[1].set_xlabel("Hourly Return")

axes[2].plot(df_primary["volume"].rolling(24).mean())
axes[2].set_title("24h Rolling Volume")
axes[2].set_ylabel("Contracts")

plt.tight_layout()
plt.show()
```

### Volume & Order Flow

```python
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(df_primary.index, df_primary["close"], color="black")
axes[0].set_ylabel("Price")
axes[0].set_title(f"{best} — Price & Order Flow")

if "order_flow" in df_primary.columns:
    of = df_primary["order_flow"].rolling(12).mean()
    axes[1].bar(df_primary.index, of, color=["green" if x > 0 else "red" for x in of], alpha=0.6)
    axes[1].set_ylabel("Net Order Flow (12h MA)")

plt.tight_layout()
plt.show()
```

### Train/Val/Test Split Visualization

```python
from src.preprocessor import train_val_test_split

train, val, test = train_val_test_split(df_primary)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train.index, train["close"], label=f"Train ({len(train)} bars)", color="blue")
ax.plot(val.index, val["close"], label=f"Val ({len(val)} bars)", color="orange")
ax.plot(test.index, test["close"], label=f"Test ({len(test)} bars)", color="red")
ax.legend()
ax.set_title("Chronological Train / Validation / Test Split (70/15/15)")
ax.set_ylabel("Price")
plt.tight_layout()
plt.show()
```

---

## 8. Summary of the Pipeline

```
Kalshi API
    │
    ▼
┌─────────────────────────────┐
│  Raw Trades (115K rows)     │
│  created_time, yes_price,   │
│  count, taker_side, ...     │
└─────────────┬───────────────┘
              │ normalise_api_trades()
              ▼
┌─────────────────────────────┐
│  Cleaned Trades             │
│  timestamp, price [0-1],    │
│  volume, is_buy,            │
│  days_to_expiration         │
└─────────────┬───────────────┘
              │ trades_to_ohlcv(freq="1h")
              ▼
┌─────────────────────────────┐
│  OHLCV Bars (per market)    │
│  open, high, low, close,    │
│  volume, vwap, buy_volume,  │
│  sell_volume, order_flow    │
└─────────────┬───────────────┘
              │ train_val_test_split()
              ▼
┌─────────────────────────────┐
│  Train (70%) / Val (15%)    │
│  / Test (15%)               │
│  Chronological, no shuffle  │
└─────────────────────────────┘
```

---

## 9. Common Gotchas

| Issue | Cause | Fix |
|-------|-------|-----|
| `yes_price` = 65 but you expected 0.65 | API returns cents (1-99) | Divide by 100 |
| All top markets are `KXMVE...` combos | Multivariate parlay bets inflate volume | Filter out tickers starting with `KXMVE` |
| Market has 20K volume but only 2 trades | Block trades (one trade = thousands of contracts) | Rank by trade count, not volume, for time series |
| Gaps in hourly bars | No trades during off-hours | Forward-fill (`ffill`) |
| `inf` values after feature engineering | `log(0)` or division by zero in ratios | Replace inf with NaN, then fill or clip |
| Settled market stuck at $0.01 | Event already resolved | Score markets by `price_std x test_std` to avoid |
| `series_ticker` is NaN | Some API responses don't populate this field | Set it manually from the fetch metadata |

"""Trade-level → OHLCV aggregation, cleaning, train/test splitting, scaling."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Normalise raw API data into a common schema
# ------------------------------------------------------------------

def normalise_api_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Kalshi API trade data to standard format.

    API columns: created_time, yes_price (cents), count (contracts),
                 taker_side, market_ticker, expiration_time, ...
    Standard:    timestamp, price (0–1), volume, market_id, ...
    """
    out = df.copy()

    # Timestamp
    if "created_time" in out.columns and "timestamp" not in out.columns:
        out["timestamp"] = pd.to_datetime(out["created_time"], utc=True)

    # Price: yes_price is in cents (1–99), normalise to [0.01, 0.99]
    if "yes_price" in out.columns and "price" not in out.columns:
        out["price"] = out["yes_price"] / 100.0
    elif "yes_price" in out.columns:
        # price column may exist but be in different scale
        if out["price"].max() <= 1.0:
            pass  # already 0-1
        else:
            out["price"] = out["yes_price"] / 100.0

    # Volume
    if "count" in out.columns and "volume" not in out.columns:
        out["volume"] = out["count"]

    # Market id
    if "market_ticker" in out.columns and "market_id" not in out.columns:
        out["market_id"] = out["market_ticker"]

    # Taker side → buy/sell volume
    if "taker_side" in out.columns:
        out["is_buy"] = (out["taker_side"] == "yes").astype(int)

    # Days to expiration
    if "expiration_time" in out.columns:
        exp = pd.to_datetime(out["expiration_time"], utc=True)
        out["days_to_expiration"] = (
            (exp - out["timestamp"]).dt.total_seconds() / 86400
        ).clip(lower=0)

    return out


# ------------------------------------------------------------------
# Aggregation helpers
# ------------------------------------------------------------------

def trades_to_ohlcv(df: pd.DataFrame, freq: str = "1h",
                    price_col: str = "price",
                    volume_col: str = "volume",
                    time_col: str = "timestamp") -> pd.DataFrame:
    """Resample tick/5-min data into OHLCV bars at *freq* (e.g. '1h', '4h', '1D')."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    ohlcv = df[price_col].resample(freq).ohlc()
    ohlcv.columns = ["open", "high", "low", "close"]

    if volume_col in df.columns:
        ohlcv["volume"] = df[volume_col].resample(freq).sum()
    else:
        ohlcv["volume"] = df[price_col].resample(freq).count()

    # VWAP
    if volume_col in df.columns:
        ohlcv["vwap"] = (
            (df[price_col] * df[volume_col]).resample(freq).sum()
            / df[volume_col].resample(freq).sum()
        )
    else:
        ohlcv["vwap"] = ohlcv["close"]

    # Buy/sell volume split
    if "is_buy" in df.columns and volume_col in df.columns:
        ohlcv["buy_volume"] = (
            (df["is_buy"] * df[volume_col]).resample(freq).sum()
        )
        ohlcv["sell_volume"] = ohlcv["volume"] - ohlcv["buy_volume"]
        ohlcv["order_flow"] = ohlcv["buy_volume"] - ohlcv["sell_volume"]

    # Forward-fill gaps (market closed / no trades)
    ohlcv = ohlcv.ffill()
    ohlcv = ohlcv.dropna(subset=["close"])
    return ohlcv


def aggregate_market(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Full pipeline for a single market DataFrame."""
    ohlcv = trades_to_ohlcv(df, freq=freq)

    # Carry forward auxiliary columns if present
    aux_cols = ["days_to_expiration", "bid", "ask", "spread",
                "bid_depth", "ask_depth", "imbalance"]
    ts_col = "timestamp"
    if ts_col in df.columns:
        df_ts = df.copy()
        df_ts[ts_col] = pd.to_datetime(df_ts[ts_col])
        df_ts = df_ts.set_index(ts_col).sort_index()
        for col in aux_cols:
            if col in df_ts.columns:
                resampled = df_ts[col].resample(freq).last().ffill()
                ohlcv[col] = resampled

    return ohlcv


# ------------------------------------------------------------------
# Splitting
# ------------------------------------------------------------------

def train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split (no shuffling)."""
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


# ------------------------------------------------------------------
# Scaling
# ------------------------------------------------------------------

class FeatureScaler:
    """Fits a StandardScaler on training data and transforms all splits."""

    def __init__(self, exclude_cols: list[str] | None = None):
        self.scaler = StandardScaler()
        self.exclude = set(exclude_cols or [])
        self.feature_cols: list[str] = []

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.feature_cols = [c for c in train.columns if c not in self.exclude]
        out = train.copy()
        out[self.feature_cols] = self.scaler.fit_transform(train[self.feature_cols])
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return out


# ------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------

def prepare_dataset(
    raw_df: pd.DataFrame,
    freq: str = "1h",
    market_col: str = "market_id",
) -> dict[str, pd.DataFrame]:
    """Aggregate each market and return dict market_id -> OHLCV DataFrame."""
    # Auto-detect API data and normalise
    if "created_time" in raw_df.columns or "market_ticker" in raw_df.columns:
        raw_df = normalise_api_trades(raw_df)

    datasets: dict[str, pd.DataFrame] = {}
    if market_col in raw_df.columns:
        for mid, grp in raw_df.groupby(market_col):
            agg = aggregate_market(grp, freq=freq)
            if len(agg) >= 10:  # skip markets with very few bars
                datasets[str(mid)] = agg
    else:
        datasets["default"] = aggregate_market(raw_df, freq=freq)
    return datasets

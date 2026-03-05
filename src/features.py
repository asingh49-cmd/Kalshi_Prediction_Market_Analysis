"""Feature engineering for prediction-market price series.

All functions accept and return DataFrames with a DatetimeIndex.
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Technical indicators
# ------------------------------------------------------------------

def add_returns(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    df["return_1"] = df[col].pct_change()
    df["log_return_1"] = np.log(df[col] / df[col].shift(1))
    for w in [5, 10, 20]:
        df[f"return_{w}"] = df[col].pct_change(w)
    return df


def add_moving_averages(df: pd.DataFrame, col: str = "close",
                        windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or [5, 10, 20, 50]
    for w in windows:
        df[f"sma_{w}"] = df[col].rolling(w).mean()
        df[f"ema_{w}"] = df[col].ewm(span=w, adjust=False).mean()
    return df


def add_volatility(df: pd.DataFrame, col: str = "close",
                   windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or [5, 10, 20, 50]
    ret = df[col].pct_change()
    for w in windows:
        df[f"volatility_{w}"] = ret.rolling(w).std()
    return df


def add_rsi(df: pd.DataFrame, col: str = "close", window: int = 14) -> pd.DataFrame:
    delta = df[col].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi_{window}"] = 100 - 100 / (1 + rs)
    return df


def add_bollinger_bands(df: pd.DataFrame, col: str = "close",
                        window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = df[col].rolling(window).mean()
    std = df[col].rolling(window).std()
    df["bb_upper"] = np.clip(mid + n_std * std, 0, 1)
    df["bb_lower"] = np.clip(mid - n_std * std, 0, 1)
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df[col] - df["bb_lower"]) / df["bb_width"].replace(0, np.nan)
    return df


def add_macd(df: pd.DataFrame, col: str = "close",
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_momentum(df: pd.DataFrame, col: str = "close",
                 windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or [5, 10, 20]
    for w in windows:
        df[f"momentum_{w}"] = df[col] - df[col].shift(w)
    return df


# ------------------------------------------------------------------
# Prediction-market-specific
# ------------------------------------------------------------------

def add_logit_features(df: pd.DataFrame, col: str = "close",
                       eps: float = 0.001) -> pd.DataFrame:
    """Logit transform: maps (0,1) → ℝ for unbounded modeling."""
    p = df[col].clip(eps, 1 - eps)
    df["logit"] = np.log(p / (1 - p))
    df["logit_return"] = df["logit"].diff()
    return df


def add_time_to_expiration(df: pd.DataFrame,
                           dte_col: str = "days_to_expiration") -> pd.DataFrame:
    if dte_col not in df.columns:
        return df
    dte = df[dte_col].clip(lower=0.01)
    df["dte_sqrt"] = np.sqrt(dte)
    df["dte_log"] = np.log(dte)
    df["dte_inv"] = 1.0 / dte
    return df


# ------------------------------------------------------------------
# Microstructure
# ------------------------------------------------------------------

def add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    if "bid" in df.columns and "ask" in df.columns:
        df["spread_mid"] = (df["ask"] - df["bid"]) / ((df["ask"] + df["bid"]) / 2)
        df["spread_momentum"] = df["spread_mid"].diff()

    if "imbalance" in df.columns:
        df["imbalance_ma5"] = df["imbalance"].rolling(5).mean()

    if "vwap" in df.columns and "close" in df.columns:
        df["vwap_deviation"] = df["close"] - df["vwap"]

    if "volume" in df.columns:
        df["volume_ma10"] = df["volume"].rolling(10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma10"].replace(0, np.nan)

    return df


# ------------------------------------------------------------------
# Cross-market
# ------------------------------------------------------------------

def add_cross_market_features(
    datasets: dict[str, pd.DataFrame],
    col: str = "close",
    window: int = 20,
) -> dict[str, pd.DataFrame]:
    """Add rolling correlations between markets."""
    tickers = list(datasets.keys())
    if len(tickers) < 2:
        return datasets

    # Build aligned price matrix
    price_matrix = pd.DataFrame(
        {t: datasets[t][col] for t in tickers if col in datasets[t].columns}
    )
    price_matrix = price_matrix.ffill().bfill()

    for t in tickers:
        if t not in price_matrix.columns:
            continue
        # Mean rolling correlation with other markets
        corrs = []
        for t2 in tickers:
            if t2 == t or t2 not in price_matrix.columns:
                continue
            corrs.append(price_matrix[t].rolling(window).corr(price_matrix[t2]))
        if corrs:
            mean_corr = pd.concat(corrs, axis=1).mean(axis=1)
            datasets[t]["cross_corr_mean"] = mean_corr.reindex(datasets[t].index)
    return datasets


# ------------------------------------------------------------------
# Target
# ------------------------------------------------------------------

def add_targets(df: pd.DataFrame, col: str = "close",
                horizons: list[int] | None = None) -> pd.DataFrame:
    horizons = horizons or [1]
    for h in horizons:
        df[f"target_price_{h}"] = df[col].shift(-h)
        df[f"target_return_{h}"] = df[col].pct_change(h).shift(-h)
        df[f"target_direction_{h}"] = (df[f"target_return_{h}"] > 0).astype(float)
    return df


# ------------------------------------------------------------------
# Master pipeline
# ------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    """Apply the full feature pipeline to a single market OHLCV DataFrame."""
    df = add_returns(df, col)
    df = add_moving_averages(df, col)
    df = add_volatility(df, col)
    df = add_rsi(df, col)
    df = add_bollinger_bands(df, col)
    df = add_macd(df, col)
    df = add_momentum(df, col)
    df = add_logit_features(df, col)
    df = add_time_to_expiration(df)
    df = add_microstructure(df)
    df = add_targets(df, col, horizons=[1])
    return df

"""Orchestrates data fetching from Kalshi API with fallback to synthetic data."""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .api_client import KalshiClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARENT_DATA = PROJECT_ROOT.parent / "data"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

KALSHI_API_KEY = "d4bd7333-f8b7-4035-9ebd-dda25d864946"

TOP_N_MARKETS = 10

# Diverse series to scan — each covers a different economic/event category
SCAN_SERIES = [
    "KXFED",            # Fed funds rate target
    "KXFEDDECISION",    # Fed rate decision (hike/cut/hold)
    "KXCPI",            # CPI inflation
    "KXGDP",            # GDP growth
    "KXPAYROLLS",       # Non-farm payrolls
    "KXTNOTE",          # Treasury note yields
    "KXU3",             # Unemployment rate
    "KXRECSSNBER",      # NBER recession call
    "KXT20WORLDCUP",    # Cricket T20 World Cup
    "KXRAINNYCM",       # NYC rain
    "KXDENSNOWMB",      # Denver snow
    "KXHMONTHRANGE",    # Temperature anomaly
    "KXGUINEAWORM",     # Guinea worm cases
    "KXBOATSTRIKERELEASE",  # Political event
    "KXPGATOUR",        # PGA golf
]


def fetch_api_data(api_key: str | None = None,
                   top_n: int = TOP_N_MARKETS,
                   max_trades_per_market: int = 500_000) -> pd.DataFrame | None:
    """Fetch trade data for the *top_n* highest-volume markets on Kalshi.

    Picks the single highest-volume market from each series to ensure
    diversity, then ranks across series and takes the top N.
    """
    api_key = api_key or KALSHI_API_KEY
    all_trades: list[pd.DataFrame] = []

    try:
        with KalshiClient(api_key=api_key) as client:
            # Step 1: For each series, find the single biggest market
            best_per_series: dict[str, dict] = {}

            for series in SCAN_SERIES:
                print(f"Scanning series {series}...")
                try:
                    markets = client.get_markets(series_ticker=series)
                except Exception as e:
                    print(f"  Skip {series}: {e}")
                    continue
                # Filter out MVE combos
                markets = [m for m in markets
                           if not m.get("ticker", "").startswith("KXMVE")]
                if not markets:
                    continue
                # Pick the single highest-volume market from this series
                best = max(markets, key=lambda m: m.get("volume", 0))
                best["_series"] = series  # tag for display
                best_per_series[series] = best
                print(f"  Best: {best.get('ticker', '?')}  "
                      f"vol={best.get('volume', 0):,}")

            # Step 2: Rank across series by volume, take top N
            ranked = sorted(best_per_series.values(),
                            key=lambda m: m.get("volume", 0), reverse=True)
            top_markets = ranked[:top_n]

            print(f"\nTop {top_n} markets (1 per series, ranked by volume):")
            for i, mkt in enumerate(top_markets, 1):
                print(f"  {i:>2}. [{mkt['_series']:20s}]  "
                      f"{mkt.get('ticker', '?'):45s}  "
                      f"vol={mkt.get('volume', 0):>12,}")

            # Step 3: Fetch trades for each
            for mkt in top_markets:
                ticker = mkt.get("ticker", "")
                volume = mkt.get("volume", 0)
                series = mkt.get("_series", "")
                print(f"\nFetching trades for {ticker} (vol={volume:,})...")
                try:
                    trades = client.get_trades(ticker, max_items=max_trades_per_market)
                except Exception as e:
                    print(f"  Could not fetch trades for {ticker}: {e}")
                    continue
                if not trades:
                    print("  No trades returned")
                    continue

                tdf = pd.DataFrame(trades)
                tdf["market_ticker"] = ticker
                tdf["series_ticker"] = series
                tdf["market_title"] = mkt.get("title", "")
                tdf["expiration_time"] = mkt.get("expiration_time", "")
                all_trades.append(tdf)
                print(f"  Got {len(trades)} trades")

        if not all_trades:
            return None
        df = pd.concat(all_trades, ignore_index=True)
        cache_path = RAW_DIR / "kalshi_trades.csv"
        df.to_csv(cache_path, index=False)
        print(f"\nSaved {len(df)} total trades from {len(all_trades)} markets "
              f"to {cache_path}")
        return df
    except Exception as exc:
        print(f"API fetch failed: {exc}")
        return None


def load_synthetic_data() -> pd.DataFrame:
    """Load pre-generated synthetic market data as fallback."""
    synth_markets = PARENT_DATA / "synthetic_markets.csv"
    if synth_markets.exists():
        df = pd.read_csv(synth_markets, parse_dates=["timestamp"])
        logger.info("Loaded synthetic markets: %d rows", len(df))
        return df

    # Generate fresh synthetic data inline
    logger.info("Generating fresh synthetic data")
    return _generate_synthetic_inline()


def _generate_synthetic_inline(n_markets: int = 10,
                               points_per_market: int = 10000,
                               seed: int = 42) -> pd.DataFrame:
    """Generate synthetic prediction-market data (self-contained)."""
    np.random.seed(seed)
    market_types = [
        "Fed Interest Rate", "Inflation CPI", "Unemployment Rate",
        "GDP Growth", "Recession Probability", "Job NFP",
    ]
    all_data = []
    for i in range(n_markets):
        mkt_seed = np.random.randint(0, 10_000)
        rng = np.random.RandomState(mkt_seed)

        n = points_per_market
        init_price = rng.uniform(0.2, 0.8)
        vol = rng.uniform(0.01, 0.04)
        mr_speed = rng.uniform(0.05, 0.2)
        tte_days = rng.uniform(10, 60)

        end_time = datetime.now()
        timestamps = pd.date_range(
            end=end_time, periods=n, freq="5min",
        )
        exp_time = end_time + timedelta(days=tte_days)
        dte = np.array([(exp_time - t).total_seconds() / 86400 for t in timestamps])

        time_vol = 1 + 5 * np.exp(-0.1 * dte)
        prices = np.empty(n)
        prices[0] = init_price
        dt = 1 / 288
        for j in range(1, n):
            v = vol * time_vol[j]
            drift = mr_speed * (0.5 - prices[j - 1])
            shock = rng.normal()
            if rng.random() < 0.05:
                shock += rng.normal(0, 0.1)
            prices[j] = np.clip(
                prices[j - 1] + drift * dt + v * shock * np.sqrt(dt), 0.01, 0.99
            )

        spread = rng.uniform(0.01, 0.05, n)
        mdf = pd.DataFrame({
            "timestamp": timestamps,
            "price": prices,
            "days_to_expiration": dte,
            "market_id": f"MARKET_{i:03d}",
            "market_name": market_types[i % len(market_types)],
            "bid": prices - spread / 2,
            "ask": prices + spread / 2,
            "spread": spread,
            "bid_depth": rng.lognormal(2, 0.5, n),
            "ask_depth": rng.lognormal(2, 0.5, n),
            "volume": rng.lognormal(1, 1, n) * 1000,
        })
        mdf["imbalance"] = (
            (mdf["bid_depth"] - mdf["ask_depth"])
            / (mdf["bid_depth"] + mdf["ask_depth"])
        )
        all_data.append(mdf)

    df = pd.concat(all_data, ignore_index=True)
    return df


def load_data(try_api: bool = True) -> pd.DataFrame:
    """Main entry point: try API first, then fall back to synthetic data."""
    # Check for cached data
    cached_csv = RAW_DIR / "kalshi_trades.csv"
    if cached_csv.exists():
        print(f"Loading cached trades from {cached_csv}")
        return pd.read_csv(cached_csv)

    if try_api:
        df = fetch_api_data()
        if df is not None and len(df) > 0:
            return df
        print("API returned no data – falling back to synthetic data")

    return load_synthetic_data()

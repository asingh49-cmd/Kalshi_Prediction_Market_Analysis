"""Lightweight httpx-based client for the Kalshi public API (no auth required)."""

import time
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
MIN_REQUEST_INTERVAL = 0.06  # ~16 req/s rate limit


class KalshiClient:
    """Kalshi API client with rate limiting and retry logic."""

    def __init__(self, api_key: str | None = None,
                 base_url: str = BASE_URL, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._last_request_time = 0.0
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=10.0,
            headers=headers,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """GET with rate limiting and retries."""
        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                resp = self._client.get(path, params=params)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate limited – waiting %ss", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        return {}

    def _paginate(self, path: str, key: str, params: Optional[dict] = None,
                  limit: int = 200, max_items: int = 50_000) -> list:
        """Cursor-based pagination – collects all pages."""
        params = dict(params or {})
        params["limit"] = min(limit, 200)
        items: list = []
        while True:
            data = self._get(path, params)
            batch = data.get(key, [])
            items.extend(batch)
            cursor = data.get("cursor")
            if not batch or not cursor or len(items) >= max_items:
                break
            params["cursor"] = cursor
        return items

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def get_series(self, series_ticker: str) -> dict:
        return self._get(f"/series/{series_ticker}")

    def get_events(self, series_ticker: Optional[str] = None,
                   status: str = "open", **kwargs) -> list:
        params: dict = {"status": status, **kwargs}
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._paginate("/events", "events", params)

    def get_markets(self, event_ticker: Optional[str] = None,
                    series_ticker: Optional[str] = None,
                    status: Optional[str] = None, **kwargs) -> list:
        params: dict = {**kwargs}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        return self._paginate("/markets", "markets", params)

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}")

    def get_trades(self, ticker: str, max_items: int = 500_000, **kwargs) -> list:
        params = {"ticker": ticker, **kwargs}
        return self._paginate("/markets/trades", "trades", params, max_items=max_items)

    def get_candlesticks(self, series_ticker: str, ticker: str,
                         period_interval: int = 60, **kwargs) -> list:
        params = {"series_ticker": series_ticker, "ticker": ticker,
                  "period_interval": period_interval, **kwargs}
        return self._paginate("/series/{}/markets/{}/candlesticks".format(
            series_ticker, ticker), "candlesticks", params)

    def get_orderbook(self, ticker: str) -> dict:
        return self._get(f"/orderbook/{ticker}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

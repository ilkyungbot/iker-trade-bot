"""
Layer 1: Data collection from Bybit API.

Fetches OHLCV candles, funding rates, open interest, and liquidation data.
Handles pagination, rate limits, and retries.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Protocol

import pandas as pd
from pybit.unified_trading import HTTP

from core.types import Candle, FundingRate, OpenInterest

logger = logging.getLogger(__name__)


class ExchangeClient(Protocol):
    """Protocol for exchange API client (allows mocking)."""

    def get_kline(self, **kwargs) -> dict: ...
    def get_funding_rate_history(self, **kwargs) -> dict: ...
    def get_open_interest(self, **kwargs) -> dict: ...
    def get_tickers(self, **kwargs) -> dict: ...


class BybitCollector:
    """Collects market data from Bybit."""

    MAX_CANDLES_PER_REQUEST = 200
    RATE_LIMIT_SLEEP = 0.1  # seconds between requests

    def __init__(self, client: ExchangeClient):
        self.client = client

    @classmethod
    def from_config(cls, api_key: str, api_secret: str, testnet: bool = False) -> "BybitCollector":
        client = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
        )
        return cls(client=client)

    def get_candles(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> list[Candle]:
        """Fetch OHLCV candles with automatic pagination."""
        all_candles: list[Candle] = []
        current_start = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000) if end_time else int(time.time() * 1000)

        while current_start < end_ms:
            try:
                response = self.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    start=current_start,
                    limit=self.MAX_CANDLES_PER_REQUEST,
                )

                result_list = response.get("result", {}).get("list", [])
                if not result_list:
                    break

                for row in result_list:
                    ts = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
                    if ts.timestamp() * 1000 > end_ms:
                        continue
                    candle = Candle(
                        timestamp=ts,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        symbol=symbol,
                        interval=interval,
                    )
                    all_candles.append(candle)

                # Bybit returns newest first, so last element has oldest timestamp
                newest_ts = int(result_list[0][0])
                if newest_ts <= current_start:
                    # No progress, we've fetched all available data
                    break
                # Advance past the newest candle we received
                current_start = newest_ts + _interval_to_ms(interval)

                if len(result_list) < self.MAX_CANDLES_PER_REQUEST:
                    break

                time.sleep(self.RATE_LIMIT_SLEEP)

            except Exception as e:
                logger.error(f"Error fetching candles for {symbol}: {e}")
                raise

        # Sort by timestamp ascending and deduplicate
        all_candles.sort(key=lambda c: c.timestamp)
        if all_candles:
            deduped = [all_candles[0]]
            for c in all_candles[1:]:
                if c.timestamp != deduped[-1].timestamp:
                    deduped.append(c)
            all_candles = deduped
        return all_candles

    def get_funding_rates(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> list[FundingRate]:
        """Fetch historical funding rates.

        Bybit v5 returns newest first. We use endTime to paginate backwards
        from now until we reach start_time.
        """
        rates: list[FundingRate] = []
        start_ms = int(start_time.timestamp() * 1000)
        cursor_end = int(end_time.timestamp() * 1000) if end_time else int(time.time() * 1000)

        while cursor_end > start_ms:
            try:
                response = self.client.get_funding_rate_history(
                    category="linear",
                    symbol=symbol,
                    endTime=cursor_end,
                    limit=200,
                )

                result_list = response.get("result", {}).get("list", [])
                if not result_list:
                    break

                for row in result_list:
                    ts_ms = int(row["fundingRateTimestamp"])
                    if ts_ms < start_ms:
                        continue
                    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    rates.append(FundingRate(
                        timestamp=ts,
                        symbol=symbol,
                        rate=float(row["fundingRate"]),
                    ))

                # Oldest entry in this batch (last element, since newest-first)
                oldest_ts = int(result_list[-1]["fundingRateTimestamp"])
                if oldest_ts <= start_ms:
                    break  # Reached our start boundary
                # Move cursor before the oldest entry
                cursor_end = oldest_ts - 1

                if len(result_list) < 200:
                    break

                time.sleep(self.RATE_LIMIT_SLEEP)

            except Exception as e:
                logger.error(f"Error fetching funding rates for {symbol}: {e}")
                raise

        rates.sort(key=lambda r: r.timestamp)
        return rates

    def get_open_interest_history(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: datetime | None = None,
    ) -> list[OpenInterest]:
        """Fetch open interest data."""
        try:
            kwargs: dict = {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": interval,
                "limit": 200,
            }
            if start_time:
                kwargs["startTime"] = int(start_time.timestamp() * 1000)

            response = self.client.get_open_interest(
                **kwargs,
            )

            result_list = response.get("result", {}).get("list", [])
            oi_list = []
            for row in result_list:
                ts = datetime.fromtimestamp(int(row["timestamp"]) / 1000, tz=timezone.utc)
                oi_list.append(OpenInterest(
                    timestamp=ts,
                    symbol=symbol,
                    value=float(row["openInterest"]),
                ))

            oi_list.sort(key=lambda o: o.timestamp)
            return oi_list

        except Exception as e:
            logger.error(f"Error fetching open interest for {symbol}: {e}")
            raise

    def get_all_usdt_perpetuals(self) -> list[dict]:
        """Fetch all USDT perpetual futures tickers with 24h volume."""
        try:
            response = self.client.get_tickers(category="linear")
            tickers = response.get("result", {}).get("list", [])
            usdt_perps = [
                {
                    "symbol": t["symbol"],
                    "volume_24h": float(t.get("turnover24h", 0)),
                    "last_price": float(t.get("lastPrice", 0)),
                    "mark_price": float(t.get("markPrice", 0)),
                    "price_24h_pct": float(t.get("price24hPcnt", 0)) * 100,
                    "high_24h": float(t.get("highPrice24h", 0)),
                    "low_24h": float(t.get("lowPrice24h", 0)),
                    "prev_price_1h": float(t.get("prevPrice1h", 0)),
                    "funding_rate": float(t.get("fundingRate", 0)),
                }
                for t in tickers
                if t["symbol"].endswith("USDT")
            ]
            return sorted(usdt_perps, key=lambda x: x["volume_24h"], reverse=True)
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            raise


def _interval_to_ms(interval: str) -> int:
    """Convert Bybit interval string to milliseconds."""
    mapping = {
        "1": 60_000,
        "3": 180_000,
        "5": 300_000,
        "15": 900_000,
        "30": 1_800_000,
        "60": 3_600_000,
        "120": 7_200_000,
        "240": 14_400_000,
        "360": 21_600_000,
        "720": 43_200_000,
        "D": 86_400_000,
        "W": 604_800_000,
    }
    return mapping.get(interval, 3_600_000)

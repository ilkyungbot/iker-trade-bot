"""Shared test fixtures for iker-trade-bot."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from core.types import ManualPosition, Side, Candle


@pytest.fixture
def make_position():
    """Factory for ManualPosition with sensible defaults."""
    _counter = [0]

    def _factory(
        symbol="BTCUSDT",
        side=Side.LONG,
        entry_price=50000.0,
        leverage=5.0,
        stop_loss=48000.0,
        take_profit=55000.0,
        margin_usdt=500.0,
        entry_reason="test",
        chat_id="123",
        is_active=True,
    ):
        _counter[0] += 1
        return ManualPosition(
            id=_counter[0],
            chat_id=chat_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            leverage=leverage,
            created_at=datetime.now(timezone.utc),
            is_active=is_active,
            stop_loss=stop_loss,
            take_profit=take_profit,
            margin_usdt=margin_usdt,
            entry_reason=entry_reason,
        )
    return _factory


@pytest.fixture
def make_df():
    """Factory for DataFrame with all indicator columns."""
    def _factory(n=100, **overrides):
        data = {
            "open": [100.0] * n,
            "high": [102.0] * n,
            "low": [98.0] * n,
            "close": [101.0] * n,
            "volume": [1000.0] * n,
            "atr": [2.0] * n,
            "adx": [25.0] * n,
            "rsi": [50.0] * n,
            "ema_20": [100.0] * n,
            "ema_50": [99.0] * n,
            "bb_lower": [95.0] * n,
            "bb_upper": [105.0] * n,
            "bb_width": [0.1] * n,
            "macd_hist": [0.0] * n,
            "volume_ratio": [1.0] * n,
            "ema_golden_cross": [False] * n,
            "ema_death_cross": [False] * n,
            "rsi_cross_up": [False] * n,
            "rsi_cross_down": [False] * n,
            "macd_hist_cross_up": [False] * n,
            "macd_hist_cross_down": [False] * n,
            "volume_anomaly": [False] * n,
            "candle_hammer": [False] * n,
            "candle_bullish_engulfing": [False] * n,
            "candle_morning_star": [False] * n,
            "candle_inverted_hammer": [False] * n,
            "candle_bearish_engulfing": [False] * n,
            "is_sideways": [False] * n,
        }
        for k, v in overrides.items():
            if isinstance(v, list):
                data[k] = v
            else:
                data[k] = [v] * n
        return pd.DataFrame(data)
    return _factory


@pytest.fixture
def make_candles():
    """Factory for list of Candle objects."""
    def _factory(symbol="BTCUSDT", n=60, interval="240", base_price=50000.0):
        candles = []
        for i in range(n):
            ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
            candles.append(Candle(
                timestamp=ts,
                open=base_price + i,
                high=base_price + i + 50,
                low=base_price + i - 50,
                close=base_price + i + 10,
                volume=1000.0,
                symbol=symbol,
                interval=interval,
            ))
        return candles
    return _factory

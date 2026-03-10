"""Tests for trend following strategy."""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from strategy.trend_following import TrendFollowingStrategy
from core.types import SignalAction


def _make_df_1h(
    n: int = 50,
    close_values: list[float] | None = None,
    adx: float = 30.0,
    atr: float = 2.0,
) -> pd.DataFrame:
    """Create 1H DataFrame with required features."""
    if close_values is None:
        close_values = [100.0 + i * 0.5 for i in range(n)]

    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]
    highs = [c + 0.5 for c in close_values]
    lows = [c - 0.5 for c in close_values]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "close": close_values,
        "high": highs,
        "low": lows,
        "open": [c - 0.1 for c in close_values],
        "volume": [1000.0] * n,
        "adx": [adx] * n,
        "atr": [atr] * n,
    })

    # Donchian channels: rolling max/min over 20 periods
    # Strategy uses prev candle's donchian, so the channel reflects prior range
    df["donchian_high"] = df["high"].rolling(20).max()
    df["donchian_low"] = df["low"].rolling(20).min()

    return df


def _make_df_4h(n: int = 60, trending_up: bool = True) -> pd.DataFrame:
    """Create 4H DataFrame with EMA."""
    if trending_up:
        ema_values = [100.0 + i * 0.3 for i in range(n)]
    else:
        ema_values = [100.0 - i * 0.3 for i in range(n)]

    return pd.DataFrame({
        "close": ema_values,
        "ema_50": ema_values,
    })


class TestTrendFollowingEntry:
    def test_long_signal_on_breakout(self):
        strategy = TrendFollowingStrategy()
        # Create uptrending data where last close breaks above donchian high
        closes = [100.0 + i * 0.2 for i in range(49)]
        closes.append(120.0)  # breakout candle

        df_1h = _make_df_1h(close_values=closes, adx=30.0, atr=2.0)
        df_4h = _make_df_4h(trending_up=True)

        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT")

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG
        assert signal.stop_loss < signal.entry_price
        assert signal.take_profit > signal.entry_price

    def test_short_signal_on_breakdown(self):
        strategy = TrendFollowingStrategy()
        # Create downtrending data where last close breaks below donchian low
        closes = [100.0 - i * 0.2 for i in range(49)]
        closes.append(70.0)  # breakdown candle

        df_1h = _make_df_1h(close_values=closes, adx=30.0, atr=2.0)
        df_4h = _make_df_4h(trending_up=False)

        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT")

        assert signal is not None
        assert signal.action == SignalAction.ENTER_SHORT
        assert signal.stop_loss > signal.entry_price
        assert signal.take_profit < signal.entry_price

    def test_no_signal_when_adx_low(self):
        strategy = TrendFollowingStrategy()
        closes = [100.0 + i * 0.2 for i in range(49)]
        closes.append(120.0)

        df_1h = _make_df_1h(close_values=closes, adx=15.0)  # ADX below threshold
        df_4h = _make_df_4h(trending_up=True)

        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT")
        assert signal is None

    def test_no_signal_when_ema_disagrees(self):
        strategy = TrendFollowingStrategy()
        closes = [100.0 + i * 0.2 for i in range(49)]
        closes.append(120.0)

        df_1h = _make_df_1h(close_values=closes, adx=30.0)
        df_4h = _make_df_4h(trending_up=False)  # EMA says down, breakout says up

        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT")
        assert signal is None

    def test_no_signal_when_already_in_position(self):
        strategy = TrendFollowingStrategy()
        closes = [100.0 + i * 0.2 for i in range(49)]
        closes.append(120.0)

        df_1h = _make_df_1h(close_values=closes, adx=30.0)
        df_4h = _make_df_4h(trending_up=True)

        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT", current_position_side="long")
        # Should not generate entry when already in position (might exit though)
        if signal is not None:
            assert signal.action == SignalAction.EXIT


class TestTrendFollowingExit:
    def test_exit_long_on_donchian_low_break(self):
        strategy = TrendFollowingStrategy()
        closes = [100.0] * 49
        closes.append(80.0)  # breaks below donchian low

        df_1h = _make_df_1h(close_values=closes, adx=30.0, atr=2.0)
        df_4h = _make_df_4h(trending_up=True)

        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT", current_position_side="long")
        assert signal is not None
        assert signal.action == SignalAction.EXIT

    def test_exit_short_on_donchian_high_break(self):
        strategy = TrendFollowingStrategy()
        closes = [100.0] * 49
        closes.append(120.0)  # breaks above donchian high

        df_1h = _make_df_1h(close_values=closes, adx=30.0, atr=2.0)
        df_4h = _make_df_4h(trending_up=False)

        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT", current_position_side="short")
        assert signal is not None
        assert signal.action == SignalAction.EXIT


class TestTrendFollowingEdgeCases:
    def test_insufficient_data(self):
        strategy = TrendFollowingStrategy()
        df_1h = _make_df_1h(n=10)
        df_4h = _make_df_4h(n=10)
        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT")
        assert signal is None

    def test_stop_loss_uses_atr(self):
        strategy = TrendFollowingStrategy()
        closes = [100.0 + i * 0.2 for i in range(49)]
        closes.append(120.0)
        atr = 3.0

        df_1h = _make_df_1h(close_values=closes, adx=30.0, atr=atr)
        df_4h = _make_df_4h(trending_up=True)

        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT")
        if signal and signal.action == SignalAction.ENTER_LONG:
            expected_sl = signal.entry_price - (atr * 1.5)
            assert abs(signal.stop_loss - expected_sl) < 0.01

    def test_confidence_scales_with_adx(self):
        strategy = TrendFollowingStrategy()
        closes = [100.0 + i * 0.2 for i in range(49)]
        closes.append(120.0)

        df_1h = _make_df_1h(close_values=closes, adx=40.0)
        df_4h = _make_df_4h(trending_up=True)

        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT")
        if signal:
            assert signal.confidence == 40.0 / 50.0

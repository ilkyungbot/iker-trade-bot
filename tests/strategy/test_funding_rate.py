"""Tests for funding rate strategy."""

import pandas as pd
from datetime import datetime, timezone, timedelta
from strategy.funding_rate import FundingRateStrategy
from core.types import SignalAction


def _make_df(n: int = 30, rsi: float = 50.0, atr: float = 2.0) -> pd.DataFrame:
    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "close": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "open": [100.0] * n,
        "volume": [1000.0] * n,
        "atr": [atr] * n,
        "rsi": [rsi] * n,
    })


class TestFundingRateEntry:
    def test_short_on_extreme_positive_funding_with_overbought_rsi(self):
        strategy = FundingRateStrategy()
        df_1h = _make_df(rsi=75.0)  # overbought
        df_4h = _make_df()

        signal = strategy.generate_signal(
            df_1h, df_4h, "BTCUSDT",
            latest_funding_rate=0.002,  # extreme positive
        )

        assert signal is not None
        assert signal.action == SignalAction.ENTER_SHORT

    def test_long_on_extreme_negative_funding_with_oversold_rsi(self):
        strategy = FundingRateStrategy()
        df_1h = _make_df(rsi=25.0)  # oversold
        df_4h = _make_df()

        signal = strategy.generate_signal(
            df_1h, df_4h, "BTCUSDT",
            latest_funding_rate=-0.002,  # extreme negative
        )

        assert signal is not None
        assert signal.action == SignalAction.ENTER_LONG

    def test_no_signal_on_normal_funding(self):
        strategy = FundingRateStrategy()
        df_1h = _make_df(rsi=75.0)
        df_4h = _make_df()

        signal = strategy.generate_signal(
            df_1h, df_4h, "BTCUSDT",
            latest_funding_rate=0.0001,  # normal
        )

        assert signal is None

    def test_no_signal_without_price_confirmation(self):
        strategy = FundingRateStrategy()
        df_1h = _make_df(rsi=50.0)  # RSI neutral — no confirmation
        df_4h = _make_df()

        signal = strategy.generate_signal(
            df_1h, df_4h, "BTCUSDT",
            latest_funding_rate=0.002,
        )

        assert signal is None

    def test_no_signal_without_funding_rate(self):
        strategy = FundingRateStrategy()
        df_1h = _make_df(rsi=75.0)
        df_4h = _make_df()

        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT")
        assert signal is None


class TestFundingRateExit:
    def test_exit_when_funding_normalizes(self):
        strategy = FundingRateStrategy()
        df_1h = _make_df()
        df_4h = _make_df()

        signal = strategy.generate_signal(
            df_1h, df_4h, "BTCUSDT",
            current_position_side="short",
            latest_funding_rate=0.0001,  # normalized
        )

        assert signal is not None
        assert signal.action == SignalAction.EXIT

    def test_hold_when_funding_still_extreme(self):
        strategy = FundingRateStrategy()
        df_1h = _make_df()
        df_4h = _make_df()

        signal = strategy.generate_signal(
            df_1h, df_4h, "BTCUSDT",
            current_position_side="short",
            latest_funding_rate=0.002,  # still extreme
        )

        assert signal is None  # hold position


class TestFundingRateStopLoss:
    def test_short_stop_above_entry(self):
        strategy = FundingRateStrategy()
        df_1h = _make_df(rsi=75.0, atr=3.0)
        df_4h = _make_df()

        signal = strategy.generate_signal(
            df_1h, df_4h, "BTCUSDT",
            latest_funding_rate=0.002,
        )

        assert signal is not None
        assert signal.stop_loss > signal.entry_price
        assert signal.stop_loss == signal.entry_price + 3.0  # ATR × 1.0

    def test_long_stop_below_entry(self):
        strategy = FundingRateStrategy()
        df_1h = _make_df(rsi=25.0, atr=3.0)
        df_4h = _make_df()

        signal = strategy.generate_signal(
            df_1h, df_4h, "BTCUSDT",
            latest_funding_rate=-0.002,
        )

        assert signal is not None
        assert signal.stop_loss < signal.entry_price
        assert signal.stop_loss == signal.entry_price - 3.0

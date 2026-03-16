"""Tests for funding rate strategy — 완화된 임계값."""

import pandas as pd
from datetime import datetime, timezone, timedelta
from strategy.funding_rate import FundingRateStrategy
from core.types import SignalAction, SignalQuality


def _make_df(n: int = 30, rsi: float = 50.0, atr: float = 2.0, is_sideways: bool = False) -> pd.DataFrame:
    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=4*i) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "close": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "open": [100.0] * n,
        "volume": [1000.0] * n,
        "atr": [atr] * n,
        "rsi": [rsi] * n,
        "is_sideways": [is_sideways] * n,
    })


class TestFundingRateEntry:
    def test_short_on_extreme_positive_funding(self):
        """양의 펀딩 + RSI 65+ → 숏 시그널."""
        strategy = FundingRateStrategy()
        df = _make_df(rsi=70.0)

        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=0.002,
        )

        assert result is not None
        assert result.signal.action == SignalAction.ENTER_SHORT
        assert len(result.explanation) >= 2

    def test_long_on_extreme_negative_funding(self):
        """음의 펀딩 + RSI 35- → 롱 시그널."""
        strategy = FundingRateStrategy()
        df = _make_df(rsi=30.0)

        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=-0.002,
        )

        assert result is not None
        assert result.signal.action == SignalAction.ENTER_LONG

    def test_threshold_lowered_to_007(self):
        """0.07% 임계값으로 완화됨."""
        strategy = FundingRateStrategy()
        df = _make_df(rsi=70.0)

        # 0.08% → 이전 0.1% 기준에서는 무시됨, 이제는 시그널
        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=0.0008,
        )
        assert result is not None

    def test_no_signal_below_threshold(self):
        strategy = FundingRateStrategy()
        df = _make_df(rsi=70.0)

        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=0.0005,  # 0.05% → 임계값 미달
        )
        assert result is None

    def test_rsi_threshold_relaxed_to_65(self):
        """RSI 65에서 시그널 생성 (기존 70에서 완화)."""
        strategy = FundingRateStrategy()
        df = _make_df(rsi=66.0)

        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=0.002,
        )
        assert result is not None

    def test_no_signal_without_rsi_confirmation(self):
        strategy = FundingRateStrategy()
        df = _make_df(rsi=50.0)

        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=0.002,
        )
        assert result is None

    def test_no_signal_without_funding_rate(self):
        strategy = FundingRateStrategy()
        df = _make_df(rsi=75.0)
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result is None

    def test_sideways_blocks_signal(self):
        strategy = FundingRateStrategy()
        df = _make_df(rsi=70.0, is_sideways=True)

        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=0.002,
        )
        assert result is None


class TestFundingRateQuality:
    def test_strong_on_very_extreme_funding(self):
        """0.1%+ → STRONG."""
        strategy = FundingRateStrategy()
        df = _make_df(rsi=75.0)

        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=0.0015,
        )
        assert result is not None
        assert result.quality == SignalQuality.STRONG

    def test_moderate_on_mild_extreme(self):
        """0.07~0.1% → MODERATE."""
        strategy = FundingRateStrategy()
        df = _make_df(rsi=70.0)

        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=0.0008,
        )
        assert result is not None
        assert result.quality == SignalQuality.MODERATE


class TestFundingRateStopLoss:
    def test_short_stop_above_entry(self):
        strategy = FundingRateStrategy()
        df = _make_df(rsi=75.0, atr=3.0)

        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=0.002,
        )

        assert result is not None
        assert result.signal.stop_loss > result.signal.entry_price
        assert result.signal.stop_loss == result.signal.entry_price + 3.0

    def test_long_stop_below_entry(self):
        strategy = FundingRateStrategy()
        df = _make_df(rsi=25.0, atr=3.0)

        result = strategy.generate_signal(
            df, "BTCUSDT",
            latest_funding_rate=-0.002,
        )

        assert result is not None
        assert result.signal.stop_loss < result.signal.entry_price
        assert result.signal.stop_loss == result.signal.entry_price - 3.0

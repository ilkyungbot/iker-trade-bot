"""Tests for trend following strategy — multi-indicator scoring."""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from strategy.trend_following import TrendFollowingStrategy
from core.types import SignalAction, SignalQuality


def _make_df(
    n: int = 60,
    close_base: float = 100.0,
    adx: float = 25.0,
    atr: float = 2.0,
    rsi: float = 50.0,
    is_sideways: bool = False,
    ema_golden_cross: bool = False,
    ema_death_cross: bool = False,
    rsi_cross_up: bool = False,
    rsi_cross_down: bool = False,
    macd_hist_cross_up: bool = False,
    macd_hist_cross_down: bool = False,
    volume_anomaly: bool = False,
    volume_ratio: float = 1.0,
    candle_hammer: bool = False,
    candle_bullish_engulfing: bool = False,
    candle_bearish_engulfing: bool = False,
    candle_inverted_hammer: bool = False,
    candle_morning_star: bool = False,
) -> pd.DataFrame:
    """Create 4H DataFrame with all required features."""
    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=4*i) for i in range(n)]
    closes = [close_base + i * 0.1 for i in range(n)]
    opens = [c - 0.05 for c in closes]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000.0] * n,
        "adx": [adx] * n,
        "atr": [atr] * n,
        "rsi": [rsi] * n,
        "rsi_signal": [rsi - 1] * n,
        "is_sideways": [False] * (n - 1) + [is_sideways],
        "ema_golden_cross": [False] * (n - 1) + [ema_golden_cross],
        "ema_death_cross": [False] * (n - 1) + [ema_death_cross],
        "rsi_cross_up": [False] * (n - 1) + [rsi_cross_up],
        "rsi_cross_down": [False] * (n - 1) + [rsi_cross_down],
        "macd_hist_cross_up": [False] * (n - 1) + [macd_hist_cross_up],
        "macd_hist_cross_down": [False] * (n - 1) + [macd_hist_cross_down],
        "volume_anomaly": [False] * (n - 1) + [volume_anomaly],
        "volume_ratio": [1.0] * (n - 1) + [volume_ratio],
        "candle_hammer": [False] * (n - 1) + [candle_hammer],
        "candle_bullish_engulfing": [False] * (n - 1) + [candle_bullish_engulfing],
        "candle_bearish_engulfing": [False] * (n - 1) + [candle_bearish_engulfing],
        "candle_inverted_hammer": [False] * (n - 1) + [candle_inverted_hammer],
        "candle_morning_star": [False] * (n - 1) + [candle_morning_star],
        "candle_doji": [False] * n,
        "bb_upper": [c + 3.0 for c in closes],
        "bb_lower": [c - 3.0 for c in closes],
        "ema_20": closes,
        "ema_50": closes,
    })
    return df


class TestTrendScoring:
    def test_strong_long_signal(self):
        """3+ 지표 → STRONG 시그널."""
        strategy = TrendFollowingStrategy()
        df = _make_df(
            adx=25.0,
            ema_golden_cross=True,
            rsi_cross_up=True,
            macd_hist_cross_up=True,
        )
        result = strategy.generate_signal(df, "BTCUSDT")

        assert result is not None
        assert result.quality == SignalQuality.STRONG
        assert result.signal.action == SignalAction.ENTER_LONG
        assert len(result.explanation) >= 3

    def test_moderate_long_signal(self):
        """2개 지표 → MODERATE 시그널."""
        strategy = TrendFollowingStrategy()
        df = _make_df(
            adx=25.0,
            ema_golden_cross=True,
            # ADX > 20이 추가 1점
        )
        result = strategy.generate_signal(df, "BTCUSDT")

        assert result is not None
        assert result.quality == SignalQuality.MODERATE

    def test_no_signal_single_indicator(self):
        """1개 지표만 → 시그널 없음."""
        strategy = TrendFollowingStrategy()
        df = _make_df(adx=15.0, ema_golden_cross=True)
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result is None

    def test_short_signal(self):
        """숏 시그널 생성."""
        strategy = TrendFollowingStrategy()
        df = _make_df(
            adx=25.0,
            ema_death_cross=True,
            rsi_cross_down=True,
            macd_hist_cross_down=True,
        )
        result = strategy.generate_signal(df, "BTCUSDT")

        assert result is not None
        assert result.signal.action == SignalAction.ENTER_SHORT
        assert result.signal.stop_loss > result.signal.entry_price

    def test_sideways_filter_blocks(self):
        """횡보장이면 시그널 없음."""
        strategy = TrendFollowingStrategy()
        df = _make_df(
            adx=25.0,
            is_sideways=True,
            ema_golden_cross=True,
            rsi_cross_up=True,
            macd_hist_cross_up=True,
        )
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result is None

    def test_volume_anomaly_adds_score(self):
        """거래량 이상치 + 상승 캔들 → 롱 점수 추가."""
        strategy = TrendFollowingStrategy()
        df = _make_df(
            adx=25.0,
            ema_golden_cross=True,
            volume_anomaly=True,
            volume_ratio=2.5,
        )
        # open < close 이므로 상승 캔들
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result is not None

    def test_sl_tp_uses_atr(self):
        """SL/TP가 ATR 기반."""
        strategy = TrendFollowingStrategy()
        df = _make_df(
            adx=25.0, atr=100.0,
            ema_golden_cross=True,
            rsi_cross_up=True,
        )
        result = strategy.generate_signal(df, "BTCUSDT")
        if result:
            s = result.signal
            assert abs(s.entry_price - s.stop_loss) == 150.0  # ATR * 1.5
            assert abs(s.take_profit - s.entry_price) == 300.0  # ATR * 3.0

    def test_risk_reward_ratio(self):
        """R:R = 1:2."""
        strategy = TrendFollowingStrategy()
        df = _make_df(
            adx=25.0,
            ema_golden_cross=True,
            rsi_cross_up=True,
        )
        result = strategy.generate_signal(df, "BTCUSDT")
        if result:
            assert result.risk_reward_ratio == 2.0

    def test_insufficient_data(self):
        strategy = TrendFollowingStrategy()
        df = _make_df(n=10)
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result is None

    def test_candle_pattern_scoring(self):
        """캔들 패턴이 롱 스코어에 기여."""
        strategy = TrendFollowingStrategy()
        df = _make_df(
            adx=25.0,
            candle_hammer=True,
        )
        result = strategy.generate_signal(df, "BTCUSDT")
        # hammer + ADX > 20 = 2점 → MODERATE
        assert result is not None
        assert result.quality == SignalQuality.MODERATE

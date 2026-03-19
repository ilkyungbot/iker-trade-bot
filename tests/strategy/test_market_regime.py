"""Tests for MarketRegimeClassifier — TDD."""

import numpy as np
import pandas as pd
import pytest

from strategy.market_regime import MarketRegimeClassifier, Regime, RegimeState


@pytest.fixture
def classifier():
    return MarketRegimeClassifier()


def _make_df_with_atr(atr_values: list[float]) -> pd.DataFrame:
    """Helper: create DataFrame with 'close' and 'atr' columns."""
    n = len(atr_values)
    return pd.DataFrame({
        "close": [100.0] * n,
        "atr": atr_values,
    })


def _make_btc_df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


class TestClassify:
    def test_high_vol_trend(self, classifier):
        """ATR ratio > 1.3 → HIGH_VOL_TREND."""
        # Last 14 values high, earlier 50 values low → ratio > 1.3
        atr_long_part = [1.0] * 50
        atr_short_part = [2.0] * 14  # mean(last14)=2.0 / mean(last50)=~1.28 → ratio ≈ 1.56
        atr_values = atr_long_part + atr_short_part
        df = _make_df_with_atr(atr_values)

        result = classifier.classify(df)

        assert result.regime == Regime.HIGH_VOL_TREND
        assert result.atr_ratio > 1.3
        assert "고변동 추세장" in result.message

    def test_low_vol_range(self, classifier):
        """ATR ratio < 0.7 → LOW_VOL_RANGE."""
        # Last 14 values low, earlier values high → ratio < 0.7
        atr_long_part = [3.0] * 50
        atr_short_part = [1.0] * 14  # mean(last14)=1.0 / mean(last50)=~1.56 → ratio ≈ 0.64
        atr_values = atr_long_part + atr_short_part
        df = _make_df_with_atr(atr_values)

        result = classifier.classify(df)

        assert result.regime == Regime.LOW_VOL_RANGE
        assert result.atr_ratio < 0.7
        assert "저변동 횡보장" in result.message

    def test_transition(self, classifier):
        """ATR ratio between 0.7 and 1.3 → TRANSITION."""
        atr_values = [1.0] * 64  # all same → ratio = 1.0
        df = _make_df_with_atr(atr_values)

        result = classifier.classify(df)

        assert result.regime == Regime.TRANSITION
        assert 0.7 <= result.atr_ratio <= 1.3
        assert "변동성 전환 구간" in result.message

    def test_btc_above_200ema(self, classifier):
        """When BTC close > 200 EMA, btc_above_200ema is True."""
        atr_values = [1.0] * 64
        df = _make_df_with_atr(atr_values)
        # Steadily rising prices → last close well above 200 EMA
        btc_closes = list(np.linspace(50, 200, 250))
        btc_df = _make_btc_df(btc_closes)

        result = classifier.classify(df, btc_df=btc_df)

        assert result.btc_above_200ema is True

    def test_btc_below_200ema_warning(self, classifier):
        """When BTC close < 200 EMA, btc_above_200ema is False."""
        atr_values = [1.0] * 64
        df = _make_df_with_atr(atr_values)
        # Steadily falling prices → last close below 200 EMA
        btc_closes = list(np.linspace(200, 50, 250))
        btc_df = _make_btc_df(btc_closes)

        result = classifier.classify(df, btc_df=btc_df)

        assert result.btc_above_200ema is False

    def test_no_btc_df_defaults_true(self, classifier):
        """If btc_df not provided, btc_above_200ema defaults to True."""
        atr_values = [1.0] * 64
        df = _make_df_with_atr(atr_values)

        result = classifier.classify(df)

        assert result.btc_above_200ema is True

    def test_insufficient_data_uses_available(self, classifier):
        """With fewer than 50 bars, use whatever is available."""
        atr_values = [1.5] * 20  # only 20 bars
        df = _make_df_with_atr(atr_values)

        result = classifier.classify(df)
        # ratio = mean(last14) / mean(last20) = 1.5/1.5 = 1.0
        assert result.regime == Regime.TRANSITION


class TestGetRegimeAdvice:
    def test_high_vol_trend_advice(self, classifier):
        state = RegimeState(
            regime=Regime.HIGH_VOL_TREND,
            atr_ratio=1.5,
            btc_above_200ema=True,
            message="고변동 추세장 — 추세추종 유리, 넓은 스탑 권장",
        )
        advice = classifier.get_regime_advice(state, side="long")

        assert advice["position_size_factor"] == 1.0
        assert advice["stop_buffer_multiplier"] > 1.0
        assert isinstance(advice["warnings"], list)

    def test_low_vol_range_advice(self, classifier):
        state = RegimeState(
            regime=Regime.LOW_VOL_RANGE,
            atr_ratio=0.5,
            btc_above_200ema=True,
            message="저변동 횡보장 — 진입 자제 권장, 좁은 스탑",
        )
        advice = classifier.get_regime_advice(state, side="long")

        assert advice["position_size_factor"] == 0.5
        assert advice["stop_buffer_multiplier"] < 1.0

    def test_transition_advice(self, classifier):
        state = RegimeState(
            regime=Regime.TRANSITION,
            atr_ratio=1.0,
            btc_above_200ema=True,
            message="변동성 전환 구간 — 포지션 축소 권장",
        )
        advice = classifier.get_regime_advice(state, side="long")

        assert advice["position_size_factor"] == 0.5

    def test_btc_below_200ema_long_warning(self, classifier):
        state = RegimeState(
            regime=Regime.HIGH_VOL_TREND,
            atr_ratio=1.5,
            btc_above_200ema=False,
            message="고변동 추세장 — 추세추종 유리, 넓은 스탑 권장",
        )
        advice = classifier.get_regime_advice(state, side="long")

        assert any("200 EMA" in w or "200EMA" in w for w in advice["warnings"])

    def test_btc_below_200ema_short_no_extra_warning(self, classifier):
        state = RegimeState(
            regime=Regime.HIGH_VOL_TREND,
            atr_ratio=1.5,
            btc_above_200ema=False,
            message="고변동 추세장 — 추세추종 유리, 넓은 스탑 권장",
        )
        advice = classifier.get_regime_advice(state, side="short")

        # No BTC EMA warning for shorts
        assert not any("200 EMA" in w or "200EMA" in w for w in advice["warnings"])

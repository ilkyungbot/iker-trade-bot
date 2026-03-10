"""Tests for position size adjustments."""

from sizing.adjustments import (
    volatility_adjustment,
    correlation_adjustment,
    drawdown_adjustment,
    consecutive_loss_adjustment,
    apply_all_adjustments,
)


class TestVolatilityAdjustment:
    def test_normal_volatility(self):
        # ATR at baseline → multiplier = 1.0
        assert volatility_adjustment(0.02, baseline_atr_pct=0.02) == 1.0

    def test_high_volatility_reduces(self):
        # ATR 2x baseline → multiplier = 0.5
        adj = volatility_adjustment(0.04, baseline_atr_pct=0.02)
        assert abs(adj - 0.5) < 0.01

    def test_low_volatility_increases(self):
        adj = volatility_adjustment(0.01, baseline_atr_pct=0.02)
        assert adj > 1.0

    def test_capped_at_1_5(self):
        adj = volatility_adjustment(0.005, baseline_atr_pct=0.02)
        assert adj == 1.5

    def test_floored_at_0_1(self):
        adj = volatility_adjustment(0.5, baseline_atr_pct=0.02)
        assert adj == 0.1

    def test_zero_atr(self):
        assert volatility_adjustment(0.0) == 1.0


class TestCorrelationAdjustment:
    def test_no_existing_positions(self):
        assert correlation_adjustment([], "long", 0.9) == 1.0

    def test_high_corr_same_direction(self):
        adj = correlation_adjustment(["long", "long"], "long", 0.9)
        assert adj < 1.0

    def test_low_corr_no_reduction(self):
        adj = correlation_adjustment(["long"], "long", 0.3)
        assert adj == 1.0

    def test_opposite_direction_no_reduction(self):
        adj = correlation_adjustment(["short"], "long", 0.9)
        assert adj == 1.0


class TestDrawdownAdjustment:
    def test_no_drawdown(self):
        assert drawdown_adjustment(0.0) == 1.0

    def test_at_threshold(self):
        adj = drawdown_adjustment(0.10)
        assert adj == 0.5  # MDD_SIZE_REDUCTION_FACTOR

    def test_above_threshold(self):
        adj = drawdown_adjustment(0.12)
        assert adj == 0.5

    def test_half_threshold(self):
        adj = drawdown_adjustment(0.05)
        assert 0.7 < adj < 0.8  # linear interpolation


class TestConsecutiveLossAdjustment:
    def test_below_threshold(self):
        assert consecutive_loss_adjustment(3) == 1.0

    def test_at_threshold(self):
        assert consecutive_loss_adjustment(5) == 0.7

    def test_above_threshold(self):
        assert consecutive_loss_adjustment(10) == 0.7


class TestApplyAllAdjustments:
    def test_all_normal(self):
        result = apply_all_adjustments(
            base_size=100.0,
            current_atr_pct=0.02,
            existing_position_sides=[],
            new_signal_side="long",
            correlation_to_existing=0.0,
            current_mdd=0.0,
            consecutive_losses=0,
        )
        assert result == 100.0

    def test_all_adverse(self):
        result = apply_all_adjustments(
            base_size=100.0,
            current_atr_pct=0.06,  # 3x baseline
            existing_position_sides=["long", "long", "long"],
            new_signal_side="long",
            correlation_to_existing=0.95,
            current_mdd=0.12,
            consecutive_losses=6,
            ml_confidence=0.5,
        )
        assert result < 10.0  # should be heavily reduced

    def test_respects_max_size(self):
        result = apply_all_adjustments(
            base_size=1000.0,
            current_atr_pct=0.01,  # low vol → increase
            existing_position_sides=[],
            new_signal_side="long",
            correlation_to_existing=0.0,
            current_mdd=0.0,
            consecutive_losses=0,
            max_size=500.0,
        )
        assert result == 500.0

    def test_ml_confidence_scales(self):
        high_conf = apply_all_adjustments(
            base_size=100.0,
            current_atr_pct=0.02,
            existing_position_sides=[],
            new_signal_side="long",
            correlation_to_existing=0.0,
            current_mdd=0.0,
            consecutive_losses=0,
            ml_confidence=1.3,
        )
        low_conf = apply_all_adjustments(
            base_size=100.0,
            current_atr_pct=0.02,
            existing_position_sides=[],
            new_signal_side="long",
            correlation_to_existing=0.0,
            current_mdd=0.0,
            consecutive_losses=0,
            ml_confidence=0.6,
        )
        assert high_conf > low_conf

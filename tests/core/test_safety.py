"""Tests for Layer 0: Safety constants and validation functions."""

from core.safety import (
    MAX_RISK_PER_TRADE,
    MAX_DAILY_LOSS,
    MAX_WEEKLY_LOSS,
    MAX_LEVERAGE,
    MAX_MDD,
    MIN_PAIR_VOLUME_24H,
    MAX_POSITION_VS_VOLUME,
    REQUIRE_EXCHANGE_STOP_LOSS,
    validate_leverage,
    validate_position_risk,
    validate_position_size,
    is_pair_eligible,
)


class TestSafetyConstants:
    def test_risk_per_trade_is_one_percent(self):
        assert MAX_RISK_PER_TRADE == 0.01

    def test_daily_loss_is_three_percent(self):
        assert MAX_DAILY_LOSS == 0.03

    def test_weekly_loss_is_five_percent(self):
        assert MAX_WEEKLY_LOSS == 0.05

    def test_max_leverage_is_ten(self):
        assert MAX_LEVERAGE == 10

    def test_max_mdd_is_fifteen_percent(self):
        assert MAX_MDD == 0.15

    def test_min_volume_is_ten_million(self):
        assert MIN_PAIR_VOLUME_24H == 10_000_000.0

    def test_max_position_vs_volume_is_two_percent(self):
        assert MAX_POSITION_VS_VOLUME == 0.02

    def test_exchange_stop_loss_required(self):
        assert REQUIRE_EXCHANGE_STOP_LOSS is True


class TestValidateLeverage:
    def test_normal_leverage_passes(self):
        assert validate_leverage(5.0) == 5.0

    def test_leverage_clamped_to_max(self):
        assert validate_leverage(20.0) == 10.0

    def test_zero_leverage_returns_one(self):
        assert validate_leverage(0.0) == 1.0

    def test_negative_leverage_returns_one(self):
        assert validate_leverage(-5.0) == 1.0

    def test_exact_max_passes(self):
        assert validate_leverage(10.0) == 10.0


class TestValidatePositionRisk:
    def test_normal_risk_passes(self):
        assert validate_position_risk(0.005) == 0.005

    def test_risk_clamped_to_max(self):
        assert validate_position_risk(0.05) == 0.01

    def test_zero_risk_returns_zero(self):
        assert validate_position_risk(0.0) == 0.0

    def test_negative_risk_returns_zero(self):
        assert validate_position_risk(-0.01) == 0.0


class TestValidatePositionSize:
    def test_normal_size_passes(self):
        # $1000 position, $100M daily volume → max allowed = $2M
        assert validate_position_size(1000.0, 100_000_000.0) == 1000.0

    def test_size_clamped_to_volume_limit(self):
        # $500K position, $10M daily volume → max allowed = $200K
        assert validate_position_size(500_000.0, 10_000_000.0) == 200_000.0

    def test_zero_size_returns_zero(self):
        assert validate_position_size(0.0, 10_000_000.0) == 0.0

    def test_negative_size_returns_zero(self):
        assert validate_position_size(-100.0, 10_000_000.0) == 0.0


class TestIsPairEligible:
    def test_eligible_pair(self):
        assert is_pair_eligible(50_000_000.0) is True

    def test_ineligible_pair(self):
        assert is_pair_eligible(5_000_000.0) is False

    def test_exact_threshold(self):
        assert is_pair_eligible(10_000_000.0) is True

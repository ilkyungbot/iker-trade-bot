"""Tests for Layer 0: Safety constants (simplified for signal bot)."""

from core.safety import (
    MAX_API_ERRORS_PER_HOUR,
    MAX_DATA_STALENESS_SECONDS,
    MIN_PAIR_VOLUME_24H,
    is_pair_eligible,
)


class TestSafetyConstants:
    def test_api_errors_limit(self):
        assert MAX_API_ERRORS_PER_HOUR == 5

    def test_staleness_limit(self):
        assert MAX_DATA_STALENESS_SECONDS == 1800

    def test_min_volume_is_ten_million(self):
        assert MIN_PAIR_VOLUME_24H == 10_000_000.0


class TestIsPairEligible:
    def test_eligible_pair(self):
        assert is_pair_eligible(50_000_000.0) is True

    def test_ineligible_pair(self):
        assert is_pair_eligible(5_000_000.0) is False

    def test_exact_threshold(self):
        assert is_pair_eligible(10_000_000.0) is True

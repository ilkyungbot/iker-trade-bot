"""Tests for EdgeDetector — funding rate extremes and OI anomaly detection."""

from datetime import datetime, timezone, timedelta

import pytest

from core.types import FundingRate, OpenInterest
from strategy.edge_detector import EdgeDetector, EdgeSignal


@pytest.fixture
def detector():
    return EdgeDetector()


def _ts(minutes_ago: int = 0) -> datetime:
    return datetime(2026, 3, 19, 12, 0, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)


def _fr(rate: float, minutes_ago: int = 0) -> FundingRate:
    return FundingRate(timestamp=_ts(minutes_ago), symbol="BTCUSDT", rate=rate)


def _oi(value: float, minutes_ago: int = 0) -> OpenInterest:
    return OpenInterest(timestamp=_ts(minutes_ago), symbol="BTCUSDT", value=value)


# ── Funding extreme ──────────────────────────────────────────────


class TestFundingExtreme:
    def test_no_signals_normal_funding(self, detector):
        """Normal funding rate (< ±0.05%) produces no signal."""
        rates = [_fr(0.0001)]  # 0.01%
        assert detector.detect_funding_extreme(rates) is None

    def test_funding_extreme_positive(self, detector):
        """Positive >= 0.05% → bearish (contrarian), moderate strength."""
        rates = [_fr(0.0005)]  # 0.05%
        sig = detector.detect_funding_extreme(rates)
        assert sig is not None
        assert sig.signal_type == "funding_extreme"
        assert sig.direction == "bearish"
        assert sig.strength == "moderate"
        assert "롱 과열" in sig.message

    def test_funding_extreme_negative(self, detector):
        """Negative <= -0.05% → bullish (contrarian), moderate strength."""
        rates = [_fr(-0.0005)]  # -0.05%
        sig = detector.detect_funding_extreme(rates)
        assert sig is not None
        assert sig.signal_type == "funding_extreme"
        assert sig.direction == "bullish"
        assert sig.strength == "moderate"
        assert "숏 과열" in sig.message

    def test_funding_extreme_strong(self, detector):
        """Rate >= ±0.1% → strong strength."""
        rates = [_fr(0.001)]  # 0.1%
        sig = detector.detect_funding_extreme(rates)
        assert sig is not None
        assert sig.strength == "strong"

        rates_neg = [_fr(-0.001)]
        sig_neg = detector.detect_funding_extreme(rates_neg)
        assert sig_neg is not None
        assert sig_neg.strength == "strong"

    def test_empty_list_returns_none(self, detector):
        assert detector.detect_funding_extreme([]) is None


# ── Funding trend ────────────────────────────────────────────────


class TestFundingTrend:
    def test_funding_trend_consecutive_up(self, detector):
        """3 consecutive increases → bearish."""
        rates = [
            _fr(0.0001, minutes_ago=30),
            _fr(0.0002, minutes_ago=20),
            _fr(0.0003, minutes_ago=10),
        ]
        sig = detector.detect_funding_trend(rates)
        assert sig is not None
        assert sig.signal_type == "funding_trend"
        assert sig.direction == "bearish"
        assert "연속 상승" in sig.message

    def test_funding_trend_consecutive_down(self, detector):
        """3 consecutive decreases → bullish."""
        rates = [
            _fr(0.0003, minutes_ago=30),
            _fr(0.0002, minutes_ago=20),
            _fr(0.0001, minutes_ago=10),
        ]
        sig = detector.detect_funding_trend(rates)
        assert sig is not None
        assert sig.signal_type == "funding_trend"
        assert sig.direction == "bullish"
        assert "연속 하락" in sig.message

    def test_funding_trend_not_enough_data(self, detector):
        """Less than 3 data points → None."""
        rates = [_fr(0.0001, 20), _fr(0.0002, 10)]
        assert detector.detect_funding_trend(rates) is None

    def test_funding_trend_mixed(self, detector):
        """Non-monotonic rates → None."""
        rates = [
            _fr(0.0001, 30),
            _fr(0.0003, 20),
            _fr(0.0002, 10),
        ]
        assert detector.detect_funding_trend(rates) is None


# ── OI anomaly ───────────────────────────────────────────────────


class TestOIAnomaly:
    def test_oi_drop_anomaly(self, detector):
        """OI drop >= 3% → liquidation cascade signal."""
        data = [_oi(100_000, 10), _oi(96_000, 0)]
        sig = detector.detect_oi_anomaly(data)
        assert sig is not None
        assert sig.signal_type == "oi_anomaly"
        assert "급감" in sig.message
        assert "청산" in sig.message

    def test_oi_spike_anomaly(self, detector):
        """OI spike >= 3% → new money entering."""
        data = [_oi(100_000, 10), _oi(104_000, 0)]
        sig = detector.detect_oi_anomaly(data)
        assert sig is not None
        assert sig.signal_type == "oi_anomaly"
        assert "급증" in sig.message
        assert "유입" in sig.message

    def test_oi_no_anomaly_small_change(self, detector):
        """OI change < threshold → None."""
        data = [_oi(100_000, 10), _oi(101_000, 0)]
        assert detector.detect_oi_anomaly(data) is None

    def test_oi_not_enough_data(self, detector):
        assert detector.detect_oi_anomaly([_oi(100_000)]) is None
        assert detector.detect_oi_anomaly([]) is None

    def test_oi_custom_threshold(self, detector):
        """Custom threshold_pct should be respected."""
        data = [_oi(100_000, 10), _oi(101_500, 0)]  # 1.5%
        assert detector.detect_oi_anomaly(data, threshold_pct=1.0) is not None
        assert detector.detect_oi_anomaly(data, threshold_pct=3.0) is None


# ── detect_all ───────────────────────────────────────────────────


class TestDetectAll:
    def test_detect_all_combines(self, detector):
        """detect_all returns signals from all sub-detectors."""
        rates = [
            _fr(0.0003, 30),
            _fr(0.0005, 20),
            _fr(0.001, 10),
        ]
        oi = [_oi(100_000, 10), _oi(94_000, 0)]

        signals = detector.detect_all(rates, oi)
        types = {s.signal_type for s in signals}
        assert "funding_extreme" in types
        assert "funding_trend" in types
        assert "oi_anomaly" in types

    def test_detect_all_empty_inputs(self, detector):
        assert detector.detect_all([], []) == []

"""Tests for data validator."""

from datetime import datetime, timezone, timedelta
from data.validator import DataValidator
from core.types import Candle


def _make_candle(
    ts: datetime,
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: float = 1000.0,
    symbol: str = "BTCUSDT",
    interval: str = "60",
) -> Candle:
    return Candle(
        timestamp=ts, open=open_, high=high, low=low,
        close=close, volume=volume, symbol=symbol, interval=interval,
    )


def _make_hourly_candles(start: datetime, count: int) -> list[Candle]:
    """Generate a sequence of valid hourly candles."""
    candles = []
    for i in range(count):
        ts = start + timedelta(hours=i)
        candles.append(_make_candle(ts=ts, close=100.0 + i * 0.5))
    return candles


class TestGapDetection:
    def test_no_gaps(self):
        v = DataValidator()
        candles = _make_hourly_candles(datetime(2024, 1, 1, tzinfo=timezone.utc), 10)
        result = v.validate_candles(candles, interval="60",
                                     now=datetime(2024, 1, 1, 10, tzinfo=timezone.utc))
        assert result.gaps == []

    def test_detects_gap(self):
        v = DataValidator()
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles = [
            _make_candle(ts=start),
            _make_candle(ts=start + timedelta(hours=1)),
            # gap: missing hour 2
            _make_candle(ts=start + timedelta(hours=3)),
        ]
        result = v.validate_candles(candles, interval="60",
                                     now=start + timedelta(hours=3, minutes=5))
        assert len(result.gaps) == 1

    def test_single_candle_no_gaps(self):
        v = DataValidator()
        candles = [_make_candle(ts=datetime(2024, 1, 1, tzinfo=timezone.utc))]
        result = v.validate_candles(candles, interval="60",
                                     now=datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc))
        assert result.gaps == []


class TestAnomalyDetection:
    def test_normal_candles_no_anomalies(self):
        v = DataValidator()
        candles = _make_hourly_candles(datetime(2024, 1, 1, tzinfo=timezone.utc), 5)
        result = v.validate_candles(candles, interval="60",
                                     now=datetime(2024, 1, 1, 5, tzinfo=timezone.utc))
        assert result.anomalies == []

    def test_detects_zero_price(self):
        v = DataValidator()
        candles = [
            _make_candle(ts=datetime(2024, 1, 1, tzinfo=timezone.utc), close=0.0),
        ]
        result = v.validate_candles(candles, interval="60",
                                     now=datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc))
        assert len(result.anomalies) >= 1
        assert "Non-positive" in result.anomalies[0][1]

    def test_detects_extreme_candle(self):
        v = DataValidator()
        candles = [
            _make_candle(
                ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open_=100.0, high=130.0, low=95.0, close=125.0,  # 25% body
            ),
        ]
        result = v.validate_candles(candles, interval="60",
                                     now=datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc))
        assert any("Extreme candle" in a[1] for a in result.anomalies)

    def test_detects_high_less_than_low(self):
        v = DataValidator()
        candles = [
            _make_candle(
                ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open_=100.0, high=95.0, low=105.0, close=100.0,
            ),
        ]
        result = v.validate_candles(candles, interval="60",
                                     now=datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc))
        assert any("High" in a[1] and "Low" in a[1] for a in result.anomalies)

    def test_detects_inter_candle_gap(self):
        v = DataValidator()
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles = [
            _make_candle(ts=start, close=100.0),
            _make_candle(ts=start + timedelta(hours=1), open_=125.0, high=130.0, low=124.0, close=126.0),
        ]
        result = v.validate_candles(candles, interval="60",
                                     now=start + timedelta(hours=1, minutes=5))
        assert any("Inter-candle gap" in a[1] for a in result.anomalies)


class TestStalenessDetection:
    def test_fresh_data_not_stale(self):
        v = DataValidator()
        now = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)
        candles = [_make_candle(ts=datetime(2024, 1, 1, 0, 45, tzinfo=timezone.utc))]
        result = v.validate_candles(candles, interval="60", now=now)
        assert result.is_stale is False

    def test_old_data_is_stale(self):
        v = DataValidator()
        now = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)  # 2 hours after last candle
        candles = [_make_candle(ts=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc))]
        result = v.validate_candles(candles, interval="60", now=now)
        assert result.is_stale is True
        assert result.stale_seconds >= 7200

    def test_empty_candles_invalid(self):
        v = DataValidator()
        result = v.validate_candles([], interval="60")
        assert result.is_valid is False
        assert result.is_stale is True


class TestOverallValidity:
    def test_valid_data(self):
        v = DataValidator()
        now = datetime(2024, 1, 1, 4, 20, tzinfo=timezone.utc)  # 20 min after last candle
        candles = _make_hourly_candles(datetime(2024, 1, 1, tzinfo=timezone.utc), 5)
        result = v.validate_candles(candles, interval="60", now=now)
        assert result.is_valid is True

    def test_gap_makes_invalid(self):
        v = DataValidator()
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles = [
            _make_candle(ts=start),
            _make_candle(ts=start + timedelta(hours=3)),
        ]
        result = v.validate_candles(candles, interval="60",
                                     now=start + timedelta(hours=3, minutes=5))
        assert result.is_valid is False

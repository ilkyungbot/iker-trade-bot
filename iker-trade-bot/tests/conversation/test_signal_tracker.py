"""Tests for signal tracker."""

import pytest
from datetime import datetime, timezone, timedelta

from conversation.signal_tracker import SignalTracker


@pytest.fixture
def tracker(tmp_path):
    db_path = str(tmp_path / "test_tracker.db")
    return SignalTracker(db_path=db_path)


class TestRecordSignal:
    def test_record_returns_id(self, tracker):
        sig_id = tracker.record_signal(
            symbol="BTCUSDT",
            direction="long",
            strategy="trend_following",
            quality="strong",
            entry_price=67450.0,
            stop_loss=66800.0,
            take_profit=68750.0,
            signal_time=datetime.now(timezone.utc),
        )
        assert sig_id > 0


class TestUpdateOutcome:
    def test_tp_hit_detection(self, tracker):
        sig_id = tracker.record_signal(
            symbol="BTCUSDT",
            direction="long",
            strategy="trend_following",
            quality="strong",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            signal_time=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        tracker.update_outcome(sig_id, price_4h=105.0, price_8h=108.0, price_24h=112.0)

        # 확인: weekly_report에서 tp_hit 반영
        report = tracker.weekly_report()
        assert report["total"] == 1
        assert report["tp_rate"] == 1.0

    def test_sl_hit_detection(self, tracker):
        sig_id = tracker.record_signal(
            symbol="BTCUSDT",
            direction="long",
            strategy="trend_following",
            quality="moderate",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            signal_time=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        tracker.update_outcome(sig_id, price_4h=94.0)

        report = tracker.weekly_report()
        assert report["total"] == 1
        assert report["sl_rate"] == 1.0

    def test_short_tp_hit(self, tracker):
        """숏 포지션 TP 도달."""
        sig_id = tracker.record_signal(
            symbol="ETHUSDT",
            direction="short",
            strategy="funding_rate",
            quality="strong",
            entry_price=3000.0,
            stop_loss=3100.0,
            take_profit=2900.0,
            signal_time=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        tracker.update_outcome(sig_id, price_24h=2850.0)

        report = tracker.weekly_report()
        assert report["tp_rate"] == 1.0


class TestWeeklyReport:
    def test_empty_report(self, tracker):
        report = tracker.weekly_report()
        assert report["total"] == 0
        assert report["tp_rate"] == 0

    def test_quality_breakdown(self, tracker):
        now = datetime.now(timezone.utc) - timedelta(hours=25)
        # Strong signal → TP hit
        id1 = tracker.record_signal("BTC", "long", "trend", "strong", 100, 95, 110, now)
        tracker.update_outcome(id1, price_24h=115.0)

        # Moderate signal → SL hit
        id2 = tracker.record_signal("ETH", "long", "trend", "moderate", 100, 95, 110, now)
        tracker.update_outcome(id2, price_24h=93.0)

        report = tracker.weekly_report()
        assert report["total"] == 2
        assert "strong" in report["by_quality"]
        assert "moderate" in report["by_quality"]
        assert report["by_quality"]["strong"]["tp"] == 1
        assert report["by_quality"]["moderate"]["sl"] == 1


class TestUncheckedSignals:
    def test_get_unchecked(self, tracker):
        old_time = datetime.now(timezone.utc) - timedelta(hours=30)
        tracker.record_signal("BTCUSDT", "long", "trend", "strong", 100, 95, 110, old_time)

        unchecked = tracker.get_unchecked_signals(hours_ago=24)
        assert len(unchecked) == 1
        assert unchecked[0]["symbol"] == "BTCUSDT"

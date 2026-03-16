"""Tests for SignalCooldown (formerly CircuitBreaker)."""

from datetime import datetime, timezone, timedelta
from execution.circuit_breaker import SignalCooldown


class TestSignalCooldown:
    def test_can_send_when_no_prior(self):
        sc = SignalCooldown(cooldown_minutes=30)
        assert sc.can_send_signal(None) is True

    def test_cooldown_blocks(self):
        sc = SignalCooldown(cooldown_minutes=30)
        recent = datetime.now(timezone.utc) - timedelta(minutes=10)
        assert sc.can_send_signal(recent) is False

    def test_cooldown_expires(self):
        sc = SignalCooldown(cooldown_minutes=30)
        old = datetime.now(timezone.utc) - timedelta(minutes=31)
        assert sc.can_send_signal(old) is True


class TestAPIErrors:
    def test_halt_after_threshold(self):
        sc = SignalCooldown()
        for _ in range(4):
            assert sc.record_api_error() is False
        assert sc.record_api_error() is True
        assert sc.is_halted

    def test_resume(self):
        sc = SignalCooldown()
        for _ in range(5):
            sc.record_api_error()
        assert sc.is_halted
        sc.resume()
        assert not sc.is_halted


class TestDataStaleness:
    def test_stale_data(self):
        sc = SignalCooldown()
        old_time = datetime.now(timezone.utc) - timedelta(minutes=35)
        assert sc.check_data_staleness(old_time) is True

    def test_fresh_data(self):
        sc = SignalCooldown()
        fresh = datetime.now(timezone.utc) - timedelta(seconds=10)
        assert sc.check_data_staleness(fresh) is False

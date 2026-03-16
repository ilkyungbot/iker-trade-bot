"""Tests for SignalCooldown."""

from datetime import datetime, timezone, timedelta
from execution.circuit_breaker import SignalCooldown


class TestCooldown:
    def test_default_allows_signal(self):
        sc = SignalCooldown()
        assert sc.can_send_signal(None) is True

    def test_blocks_within_cooldown(self):
        sc = SignalCooldown(cooldown_minutes=30)
        recent = datetime.now(timezone.utc) - timedelta(minutes=15)
        assert sc.can_send_signal(recent) is False

    def test_allows_after_cooldown(self):
        sc = SignalCooldown(cooldown_minutes=30)
        old = datetime.now(timezone.utc) - timedelta(minutes=35)
        assert sc.can_send_signal(old) is True


class TestAPIErrors:
    def test_halt_after_5_errors(self):
        sc = SignalCooldown()
        for _ in range(4):
            assert sc.record_api_error() is False
        assert sc.record_api_error() is True
        assert sc.is_halted

    def test_halted_blocks_signals(self):
        sc = SignalCooldown()
        for _ in range(5):
            sc.record_api_error()
        assert sc.can_send_signal(None) is False


class TestResume:
    def test_resume_clears_halt(self):
        sc = SignalCooldown()
        for _ in range(5):
            sc.record_api_error()
        assert sc.is_halted
        sc.resume()
        assert not sc.is_halted
        assert sc.can_send_signal(None) is True

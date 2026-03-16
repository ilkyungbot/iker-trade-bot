"""Extra tests for reporter — signal bot."""

from datetime import datetime, timezone
from review.reporter import Reporter
from core.types import Signal, SignalAction, SignalMessage, SignalQuality, StrategyName


class TestExitSignalFormat:
    def test_exit_signal_format(self):
        reporter = Reporter()
        result = reporter.format_exit_signal("BTCUSDT", "long", "목표가 도달")
        assert isinstance(result, str)
        assert "청산 시그널" in result
        assert "BTCUSDT" in result

    def test_monitoring_update_format(self):
        reporter = Reporter()
        result = reporter.format_monitoring_update(
            "BTCUSDT", "short", 67450, 67000, 68000, 66000,
        )
        assert "숏" in result

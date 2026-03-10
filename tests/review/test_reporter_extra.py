"""Extra tests for reporter — gap coverage."""

from datetime import datetime, timezone

from review.reporter import Reporter
from core.types import Signal, SignalAction, StrategyName


class TestFormatSignalAlertExit:
    def test_exit_signal_does_not_crash(self):
        """format_signal_alert with EXIT action should not crash and should
        produce a valid string (the EXIT branch skips price/SL/TP fields)."""
        reporter = Reporter()

        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            action=SignalAction.EXIT,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            confidence=0.85,
        )

        result = reporter.format_signal_alert(signal)
        assert isinstance(result, str)
        assert "EXIT" in result
        assert "BTCUSDT" in result
        assert "85%" in result
        # Should NOT contain SL/TP lines
        assert "SL:" not in result
        assert "TP:" not in result

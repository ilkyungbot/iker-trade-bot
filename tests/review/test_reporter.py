"""Tests for Telegram reporter formatting (Korean) — Signal Bot."""

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from review.reporter import Reporter, TelegramBotSender
from core.types import (
    Signal, SignalAction, SignalMessage, SignalQuality, StrategyName,
)


def _make_signal_msg(direction="long"):
    action = SignalAction.ENTER_LONG if direction == "long" else SignalAction.ENTER_SHORT
    s = Signal(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        symbol="BTCUSDT",
        action=action,
        strategy=StrategyName.TREND_FOLLOWING,
        entry_price=67450,
        stop_loss=66800 if direction == "long" else 68100,
        take_profit=68750 if direction == "long" else 66150,
        confidence=0.57,
        metadata={"score": 4},
    )
    return SignalMessage(
        signal=s,
        quality=SignalQuality.STRONG,
        explanation=["EMA 골든크로스", "MACD 양전환", "ADX 28 > 20"],
        indicators={"adx": 28.5, "rsi": 55.2},
        risk_reward_ratio=2.0,
    )


class TestSignalMessageFormat:
    def test_format_long_signal(self):
        reporter = Reporter()
        msg = reporter.format_signal_message(_make_signal_msg("long"))
        assert "롱 시그널" in msg
        assert "BTCUSDT" in msg
        assert "67,450" in msg
        assert "근거" in msg
        assert "EMA 골든크로스" in msg
        assert "1시간 내 응답" in msg

    def test_format_short_signal(self):
        reporter = Reporter()
        msg = reporter.format_signal_message(_make_signal_msg("short"))
        assert "숏 시그널" in msg

    def test_quality_shown(self):
        reporter = Reporter()
        msg = reporter.format_signal_message(_make_signal_msg())
        assert "강함" in msg
        assert "4/7" in msg


class TestMonitoringUpdate:
    def test_format_positive(self):
        reporter = Reporter()
        msg = reporter.format_monitoring_update(
            "BTCUSDT", "long", 67450, 68000, 66800, 68750,
        )
        assert "BTCUSDT" in msg
        assert "롱" in msg
        assert "홀딩" in msg

    def test_format_negative(self):
        reporter = Reporter()
        msg = reporter.format_monitoring_update(
            "BTCUSDT", "long", 67450, 67000, 66800, 68750,
        )
        assert "BTCUSDT" in msg


class TestExitSignal:
    def test_format(self):
        reporter = Reporter()
        msg = reporter.format_exit_signal("BTCUSDT", "long", "목표가 도달!")
        assert "청산 시그널" in msg
        assert "목표가 도달" in msg
        assert "응답" in msg


class TestWeeklyAccuracy:
    def test_empty_report(self):
        reporter = Reporter()
        msg = reporter.format_weekly_accuracy({"total": 0})
        assert "시그널이 없습니다" in msg

    def test_with_data(self):
        reporter = Reporter()
        report = {
            "total": 10,
            "tp_rate": 0.6,
            "sl_rate": 0.3,
            "by_quality": {"strong": {"total": 5, "tp": 4, "sl": 1}},
        }
        msg = reporter.format_weekly_accuracy(report)
        assert "10개" in msg
        assert "60%" in msg
        assert "strong" in msg


class TestSendIntegration:
    @pytest.mark.anyio
    async def test_send_signal_calls_sender(self):
        sender = AsyncMock()
        reporter = Reporter(sender=sender, chat_id="123")
        msg = _make_signal_msg()

        await reporter.send_signal(msg)
        sender.send_message.assert_awaited_once()
        text = sender.send_message.call_args[0][1]
        assert "BTCUSDT" in text

    @pytest.mark.anyio
    async def test_send_alert(self):
        sender = AsyncMock()
        reporter = Reporter(sender=sender, chat_id="123")
        await reporter.send_alert("test message")
        text = sender.send_message.call_args[0][1]
        assert "알림" in text

    @pytest.mark.anyio
    async def test_send_logs_when_no_sender(self, caplog):
        reporter = Reporter(sender=None, chat_id="")
        with caplog.at_level(logging.INFO):
            await reporter.send_alert("test")
        assert any("[NO TELEGRAM]" in r.message for r in caplog.records)


class TestTelegramBotSender:
    @patch("review.reporter.HAS_TELEGRAM", False)
    def test_raises_import_error(self):
        with pytest.raises(ImportError, match="python-telegram-bot is required"):
            TelegramBotSender(bot_token="fake")

    @patch("review.reporter.HAS_TELEGRAM", True)
    @patch("review.reporter.Bot", create=True)
    def test_creates_bot(self, mock_bot_cls):
        sender = TelegramBotSender(bot_token="tok123")
        mock_bot_cls.assert_called_once_with(token="tok123")

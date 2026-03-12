"""Tests for Telegram reporter formatting (Korean)."""

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from review.reporter import Reporter, TelegramBotSender
from review.performance import PerformanceMetrics
from core.types import (
    Trade, Side, StrategyName, PortfolioState, CircuitBreakerState,
    Signal, SignalAction,
)


def _make_trade(pnl: float = 100.0) -> Trade:
    return Trade(
        symbol="BTCUSDT", side=Side.LONG, strategy=StrategyName.TREND_FOLLOWING,
        entry_price=50000, exit_price=51000, quantity=0.1, leverage=5,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        pnl=pnl, pnl_percent=2.0, fees=5, slippage=0,
        stop_loss_hit=False, trailing_stop_hit=True,
    )


def _make_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_trades=50, winning_trades=25, losing_trades=25,
        win_rate=0.5, avg_win=200, avg_loss=100, payoff_ratio=2.0,
        total_pnl=2500, total_fees=250, net_pnl=2500,
        max_drawdown=500, sharpe_ratio=1.5, profit_factor=2.0,
        avg_trade_duration_hours=6.0,
    )


class TestTradeAlert:
    def test_format_winning_trade(self):
        reporter = Reporter()
        msg = reporter.format_trade_alert(_make_trade(pnl=100))
        assert "BTCUSDT" in msg
        assert "+100" in msg
        assert "롱" in msg
        assert "포지션 청산" in msg
        assert "추세추종" in msg

    def test_format_losing_trade(self):
        reporter = Reporter()
        msg = reporter.format_trade_alert(_make_trade(pnl=-50))
        assert "-50" in msg
        assert "손절" not in msg  # stop_loss_hit=False

    def test_format_stop_loss_hit(self):
        reporter = Reporter()
        trade = _make_trade(pnl=-50)
        trade = Trade(**{**trade.__dict__, "stop_loss_hit": True})
        msg = reporter.format_trade_alert(trade)
        assert "손절" in msg


class TestSignalAlert:
    def test_format_long_signal(self):
        reporter = Reporter()
        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT", action=SignalAction.ENTER_LONG,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=50000, stop_loss=49000, take_profit=52000,
            confidence=0.8,
        )
        msg = reporter.format_signal_alert(signal)
        assert "롱" in msg
        assert "80%" in msg
        assert "진입 시그널" in msg

    def test_format_short_signal(self):
        reporter = Reporter()
        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="ETHUSDT", action=SignalAction.ENTER_SHORT,
            strategy=StrategyName.FUNDING_RATE,
            entry_price=3000, stop_loss=3100, take_profit=2800,
            confidence=0.6,
        )
        msg = reporter.format_signal_alert(signal)
        assert "숏" in msg
        assert "펀딩레이트" in msg


class TestDailyReport:
    def test_format(self):
        reporter = Reporter()
        state = PortfolioState(
            total_capital=1050000, available_capital=1000000,
            daily_pnl=5000, current_mdd=0.02,
        )
        msg = reporter.format_daily_report(
            state, _make_metrics(), datetime(2024, 1, 1),
        )
        assert "일간 리포트" in msg
        assert "1,050,000" in msg
        assert "+5,000" in msg
        assert "승률" in msg


class TestWeeklyReport:
    def test_format_with_strategies(self):
        reporter = Reporter()
        state = PortfolioState(
            total_capital=1050000, available_capital=1000000,
            weekly_pnl=15000, current_mdd=0.03,
        )
        strategy_attrs = {
            "trend_following": _make_metrics(),
            "funding_rate": _make_metrics(),
        }
        msg = reporter.format_weekly_report(state, _make_metrics(), strategy_attrs)
        assert "주간 리포트" in msg
        assert "추세추종" in msg
        assert "펀딩레이트" in msg


# ---------------------------------------------------------------------------
# Integration tests: Reporter + TelegramSender
# ---------------------------------------------------------------------------


class TestReporterSendIntegration:
    """Test that Reporter delegates to sender.send_message correctly."""

    @pytest.mark.anyio
    async def test_send_calls_sender_when_provided(self):
        sender = AsyncMock()
        reporter = Reporter(sender=sender, chat_id="12345")

        await reporter.send_trade_alert(_make_trade())

        sender.send_message.assert_awaited_once()
        call_args = sender.send_message.call_args
        assert call_args[0][0] == "12345"          # chat_id
        assert "BTCUSDT" in call_args[0][1]        # text contains symbol
        assert call_args[1]["parse_mode"] == "HTML" # keyword arg

    @pytest.mark.anyio
    async def test_send_logs_when_sender_is_none(self, caplog):
        reporter = Reporter(sender=None, chat_id="")

        with caplog.at_level(logging.INFO):
            await reporter.send_trade_alert(_make_trade())

        assert any("[NO TELEGRAM]" in r.message for r in caplog.records)

    @pytest.mark.anyio
    async def test_send_catches_sender_exception(self, caplog):
        """_send must not propagate exceptions raised by the sender."""
        sender = AsyncMock()
        sender.send_message.side_effect = RuntimeError("network down")
        reporter = Reporter(sender=sender, chat_id="12345")

        with caplog.at_level(logging.ERROR):
            await reporter.send_trade_alert(_make_trade())  # must not raise

        assert any("Failed to send Telegram message" in r.message for r in caplog.records)

    @pytest.mark.anyio
    async def test_send_signal_alert_delegates(self):
        sender = AsyncMock()
        reporter = Reporter(sender=sender, chat_id="99")

        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="ETHUSDT", action=SignalAction.ENTER_LONG,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=3000, stop_loss=2900, take_profit=3200,
            confidence=0.75,
        )
        await reporter.send_signal_alert(signal)

        sender.send_message.assert_awaited_once()
        text = sender.send_message.call_args[0][1]
        assert "롱" in text
        assert "ETHUSDT" in text


class TestSendAlert:
    """Test the raw send_alert method formats correctly."""

    @pytest.mark.anyio
    async def test_send_alert_wraps_in_korean_prefix(self):
        sender = AsyncMock()
        reporter = Reporter(sender=sender, chat_id="1")

        await reporter.send_alert("Circuit breaker tripped")

        text = sender.send_message.call_args[0][1]
        assert "긴급 알림" in text
        assert "Circuit breaker tripped" in text

    @pytest.mark.anyio
    async def test_send_alert_logs_when_no_sender(self, caplog):
        reporter = Reporter(sender=None, chat_id="")
        with caplog.at_level(logging.INFO):
            await reporter.send_alert("test message")
        assert any("[NO TELEGRAM]" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TelegramBotSender unit tests (telegram lib mocked)
# ---------------------------------------------------------------------------


class TestTelegramBotSender:

    @patch("review.reporter.HAS_TELEGRAM", False)
    def test_raises_import_error_when_telegram_unavailable(self):
        with pytest.raises(ImportError, match="python-telegram-bot is required"):
            TelegramBotSender(bot_token="fake-token")

    @patch("review.reporter.HAS_TELEGRAM", True)
    @patch("review.reporter.Bot", create=True)
    def test_creates_bot_with_token(self, mock_bot_cls):
        sender = TelegramBotSender(bot_token="tok123")
        mock_bot_cls.assert_called_once_with(token="tok123")

    @pytest.mark.anyio
    @patch("review.reporter.HAS_TELEGRAM", True)
    @patch("review.reporter.Bot", create=True)
    async def test_send_message_delegates_to_bot(self, mock_bot_cls):
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        mock_bot_cls.return_value = mock_bot

        sender = TelegramBotSender(bot_token="tok")
        await sender.send_message("chat1", "hello", parse_mode="HTML")

        mock_bot.send_message.assert_awaited_once_with(
            chat_id="chat1", text="hello", parse_mode="HTML",
        )

    @pytest.mark.anyio
    @patch("review.reporter.HAS_TELEGRAM", True)
    @patch("review.reporter.Bot", create=True)
    async def test_send_message_logs_on_exception(self, mock_bot_cls, caplog):
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(side_effect=Exception("timeout"))
        mock_bot_cls.return_value = mock_bot

        sender = TelegramBotSender(bot_token="tok")
        with caplog.at_level(logging.ERROR):
            await sender.send_message("chat1", "hi")  # must not raise

        assert any("Telegram send error" in r.message for r in caplog.records)

"""
Layer 6: Telegram reporting.

Sends trade alerts, daily/weekly/monthly reports, and handles kill switch commands.
"""

import logging
from datetime import datetime
from typing import Protocol

from core.types import Trade, PortfolioState, Signal, SignalAction
from review.performance import PerformanceMetrics

logger = logging.getLogger(__name__)


class TelegramSender(Protocol):
    """Protocol for sending Telegram messages."""

    async def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML") -> None: ...


class Reporter:
    """Formats and sends reports via Telegram."""

    def __init__(self, sender: TelegramSender | None = None, chat_id: str = ""):
        self.sender = sender
        self.chat_id = chat_id

    def format_trade_alert(self, trade: Trade) -> str:
        """Format a single trade completion alert."""
        emoji = "+" if trade.pnl > 0 else ""
        return (
            f"<b>Trade Closed</b>\n"
            f"Pair: {trade.symbol}\n"
            f"Side: {trade.side.value.upper()}\n"
            f"Strategy: {trade.strategy.value}\n"
            f"Entry: {trade.entry_price:.4f}\n"
            f"Exit: {trade.exit_price:.4f}\n"
            f"PnL: {emoji}{trade.pnl:.2f} USDT ({emoji}{trade.pnl_percent:.1f}%)\n"
            f"{'SL Hit' if trade.stop_loss_hit else 'TP/Signal Exit'}"
        )

    def format_signal_alert(self, signal: Signal) -> str:
        """Format a new signal alert."""
        action = "LONG" if signal.action == SignalAction.ENTER_LONG else "SHORT"
        if signal.action == SignalAction.EXIT:
            action = "EXIT"
        return (
            f"<b>Signal: {action}</b>\n"
            f"Pair: {signal.symbol}\n"
            f"Strategy: {signal.strategy.value}\n"
            f"Price: {signal.entry_price:.4f}\n"
            f"SL: {signal.stop_loss:.4f}\n"
            f"TP: {signal.take_profit:.4f}\n"
            f"Confidence: {signal.confidence:.0%}"
        )

    def format_daily_report(
        self, state: PortfolioState, metrics: PerformanceMetrics, date: datetime
    ) -> str:
        """Format daily performance report."""
        pnl_emoji = "+" if state.daily_pnl >= 0 else ""
        return (
            f"<b>Daily Report - {date.strftime('%Y-%m-%d')}</b>\n\n"
            f"Capital: {state.total_capital:,.2f} USDT\n"
            f"Daily PnL: {pnl_emoji}{state.daily_pnl:,.2f} USDT\n"
            f"Open Positions: {state.position_count}\n"
            f"MDD: {state.current_mdd:.1%}\n"
            f"Circuit Breaker: {state.circuit_breaker_state.value}\n\n"
            f"<b>All-time Stats</b>\n"
            f"Total Trades: {metrics.total_trades}\n"
            f"Win Rate: {metrics.win_rate:.0%}\n"
            f"Profit Factor: {metrics.profit_factor:.2f}\n"
            f"Sharpe: {metrics.sharpe_ratio:.2f}"
        )

    def format_weekly_report(
        self,
        state: PortfolioState,
        metrics: PerformanceMetrics,
        strategy_attrs: dict[str, PerformanceMetrics],
    ) -> str:
        """Format weekly performance report."""
        pnl_emoji = "+" if state.weekly_pnl >= 0 else ""
        report = (
            f"<b>Weekly Report</b>\n\n"
            f"Capital: {state.total_capital:,.2f} USDT\n"
            f"Weekly PnL: {pnl_emoji}{state.weekly_pnl:,.2f} USDT\n"
            f"MDD: {state.current_mdd:.1%}\n\n"
        )

        for name, m in strategy_attrs.items():
            emoji = "+" if m.net_pnl >= 0 else ""
            report += (
                f"<b>{name}</b>\n"
                f"  Trades: {m.total_trades} | WR: {m.win_rate:.0%} | "
                f"PnL: {emoji}{m.net_pnl:.2f}\n"
            )

        return report

    async def send_trade_alert(self, trade: Trade) -> None:
        """Send trade completion alert."""
        msg = self.format_trade_alert(trade)
        await self._send(msg)

    async def send_signal_alert(self, signal: Signal) -> None:
        """Send new signal alert."""
        msg = self.format_signal_alert(signal)
        await self._send(msg)

    async def send_daily_report(
        self, state: PortfolioState, metrics: PerformanceMetrics, date: datetime
    ) -> None:
        msg = self.format_daily_report(state, metrics, date)
        await self._send(msg)

    async def send_weekly_report(
        self, state: PortfolioState, metrics: PerformanceMetrics,
        strategy_attrs: dict[str, PerformanceMetrics],
    ) -> None:
        msg = self.format_weekly_report(state, metrics, strategy_attrs)
        await self._send(msg)

    async def send_alert(self, message: str) -> None:
        """Send a raw alert message."""
        await self._send(f"<b>ALERT</b>\n{message}")

    async def _send(self, text: str) -> None:
        if self.sender and self.chat_id:
            try:
                await self.sender.send_message(self.chat_id, text, parse_mode="HTML")
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {e}")
        else:
            logger.info(f"[NO TELEGRAM] {text[:100]}...")

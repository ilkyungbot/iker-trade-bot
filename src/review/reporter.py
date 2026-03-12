"""
Layer 6: Telegram reporting.

Sends trade alerts, daily/weekly/monthly reports via Telegram (Korean).
"""

import logging
from datetime import datetime
from typing import Protocol

try:
    from telegram import Bot
    from telegram.error import TelegramError
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

from core.types import (
    Trade, PortfolioState, Signal, SignalAction,
    StrategyName, CircuitBreakerState, Side,
)
from review.performance import PerformanceMetrics

logger = logging.getLogger(__name__)

# --- Korean translation helpers ---

_STRATEGY_KR = {
    StrategyName.TREND_FOLLOWING.value: "추세추종",
    StrategyName.FUNDING_RATE.value: "펀딩레이트",
    "trend_following": "추세추종",
    "funding_rate": "펀딩레이트",
}

_CB_STATE_KR = {
    CircuitBreakerState.NORMAL.value: "정상",
    CircuitBreakerState.DAILY_HALT.value: "일간정지",
    CircuitBreakerState.WEEKLY_HALT.value: "주간정지",
    CircuitBreakerState.SIZE_REDUCED.value: "규모축소",
    CircuitBreakerState.FULL_STOP.value: "전면중단",
}

_SIDE_KR = {
    Side.LONG.value: "롱",
    Side.SHORT.value: "숏",
    "long": "롱",
    "short": "숏",
}


def _strategy_kr(name: str) -> str:
    return _STRATEGY_KR.get(name, name)


def _cb_state_kr(state: CircuitBreakerState) -> str:
    return _CB_STATE_KR.get(state.value, state.value)


def _side_kr(side) -> str:
    val = side.value if hasattr(side, "value") else str(side)
    return _SIDE_KR.get(val, val)


def _sign(v: float) -> str:
    return "+" if v > 0 else ""


class TelegramSender(Protocol):
    """Protocol for sending Telegram messages."""

    async def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML") -> None: ...


class TelegramBotSender:
    """Concrete TelegramSender using python-telegram-bot."""

    def __init__(self, bot_token: str) -> None:
        if not HAS_TELEGRAM:
            raise ImportError("python-telegram-bot is required: pip install python-telegram-bot")
        self._bot = Bot(token=bot_token)

    async def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML") -> None:
        try:
            await self._bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        except Exception as e:
            logger.error(f"Telegram send error: {e}")


class Reporter:
    """Formats and sends reports via Telegram (Korean)."""

    def __init__(self, sender: TelegramSender | None = None, chat_id: str = ""):
        self.sender = sender
        self.chat_id = chat_id

    def format_trade_alert(self, trade: Trade) -> str:
        """Format a single trade completion alert."""
        icon = "\u2705" if trade.pnl > 0 else "\u274c"
        reason = "손절" if trade.stop_loss_hit else "목표가 도달"
        s = _sign(trade.pnl)
        return (
            f"<b>{icon} 포지션 청산</b>\n"
            f"종목: {trade.symbol}\n"
            f"방향: {_side_kr(trade.side)}\n"
            f"전략: {_strategy_kr(trade.strategy.value)}\n"
            f"진입가: {trade.entry_price:,.4f}\n"
            f"청산가: {trade.exit_price:,.4f}\n"
            f"손익: {s}{trade.pnl:,.2f} USDT ({s}{trade.pnl_percent:.1f}%)\n"
            f"사유: {reason}"
        )

    def format_signal_alert(self, signal: Signal) -> str:
        """Format a new signal alert."""
        if signal.action == SignalAction.EXIT:
            return (
                f"<b>\U0001f6aa 청산 시그널</b>\n"
                f"종목: {signal.symbol}\n"
                f"전략: {_strategy_kr(signal.strategy.value)}\n"
                f"신뢰도: {signal.confidence:.0%}"
            )
        if signal.action == SignalAction.ENTER_LONG:
            icon = "\U0001f4c8"
            direction = "롱"
        else:
            icon = "\U0001f4c9"
            direction = "숏"
        return (
            f"<b>{icon} {direction} 진입 시그널</b>\n"
            f"종목: {signal.symbol}\n"
            f"전략: {_strategy_kr(signal.strategy.value)}\n"
            f"진입가: {signal.entry_price:,.4f}\n"
            f"손절가: {signal.stop_loss:,.4f}\n"
            f"목표가: {signal.take_profit:,.4f}\n"
            f"신뢰도: {signal.confidence:.0%}"
        )

    def format_daily_report(
        self, state: PortfolioState, metrics: PerformanceMetrics, date: datetime
    ) -> str:
        """Format daily performance report."""
        s = _sign(state.daily_pnl)
        return (
            f"<b>\U0001f4ca 일간 리포트 \u2014 {date.strftime('%Y-%m-%d')}</b>\n\n"
            f"자본금: {state.total_capital:,.2f} USDT\n"
            f"오늘 손익: {s}{state.daily_pnl:,.2f} USDT\n"
            f"보유 포지션: {state.position_count}개\n"
            f"최대낙폭: {state.current_mdd:.1%}\n"
            f"서킷브레이커: {_cb_state_kr(state.circuit_breaker_state)}\n\n"
            f"<b>\U0001f4c8 누적 성과</b>\n"
            f"총 거래: {metrics.total_trades}회\n"
            f"승률: {metrics.win_rate:.0%}\n"
            f"손익비: {metrics.profit_factor:.2f}\n"
            f"샤프지수: {metrics.sharpe_ratio:.2f}"
        )

    def format_weekly_report(
        self,
        state: PortfolioState,
        metrics: PerformanceMetrics,
        strategy_attrs: dict[str, PerformanceMetrics],
    ) -> str:
        """Format weekly performance report."""
        s = _sign(state.weekly_pnl)
        report = (
            f"<b>\U0001f4cb 주간 리포트</b>\n\n"
            f"자본금: {state.total_capital:,.2f} USDT\n"
            f"주간 손익: {s}{state.weekly_pnl:,.2f} USDT\n"
            f"최대낙폭: {state.current_mdd:.1%}\n\n"
        )

        for name, m in strategy_attrs.items():
            ps = _sign(m.net_pnl)
            report += (
                f"<b>\u25b8 {_strategy_kr(name)}</b>\n"
                f"  거래 {m.total_trades}회 | 승률 {m.win_rate:.0%} | "
                f"손익 {ps}{m.net_pnl:,.2f}\n"
            )

        return report

    async def send_trade_alert(self, trade: Trade) -> None:
        msg = self.format_trade_alert(trade)
        await self._send(msg)

    async def send_signal_alert(self, signal: Signal) -> None:
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
        await self._send(f"<b>\U0001f6a8 긴급 알림</b>\n{message}")

    async def _send(self, text: str) -> None:
        if self.sender and self.chat_id:
            try:
                await self.sender.send_message(self.chat_id, text, parse_mode="HTML")
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {e}")
        else:
            logger.info(f"[NO TELEGRAM] {text[:100]}...")

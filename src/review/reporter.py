"""
Layer 6: Telegram reporting — 시그널 봇 전용.

시그널 메시지 포맷, 모니터링 업데이트, 정확도 리포트.
"""

import logging
from datetime import datetime
from typing import Protocol

try:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

from core.types import (
    Signal, SignalMessage, SignalQuality, SignalAction,
    StrategyName, Side,
)

logger = logging.getLogger(__name__)

# --- Korean translation helpers ---

_STRATEGY_KR = {
    StrategyName.TREND_FOLLOWING.value: "추세추종",
    StrategyName.FUNDING_RATE.value: "펀딩레이트",
    "trend_following": "추세추종",
    "funding_rate": "펀딩레이트",
}

_QUALITY_ICON = {
    SignalQuality.STRONG: "\U0001f7e2 강함",
    SignalQuality.MODERATE: "\U0001f7e1 보통",
    SignalQuality.WEAK: "\U0001f534 약함",
}


def _strategy_kr(name: str) -> str:
    return _STRATEGY_KR.get(name, name)


def _sign(v: float) -> str:
    return "+" if v > 0 else ""


def _pct(entry: float, target: float) -> str:
    pct = (target - entry) / entry * 100 if entry > 0 else 0
    return f"{_sign(pct)}{pct:.2f}%"


class TelegramSender(Protocol):
    """Protocol for sending Telegram messages."""
    async def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML", reply_markup: object = None) -> None: ...


class TelegramBotSender:
    """Concrete TelegramSender using python-telegram-bot."""

    def __init__(self, bot_token: str) -> None:
        if not HAS_TELEGRAM:
            raise ImportError("python-telegram-bot is required: pip install python-telegram-bot")
        self._bot = Bot(token=bot_token)

    async def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML", reply_markup=None) -> None:
        try:
            await self._bot.send_message(
                chat_id=chat_id, text=text, parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")


class Reporter:
    """시그널 봇 전용 리포터."""

    def __init__(self, sender: TelegramSender | None = None, chat_id: str = ""):
        self.sender = sender
        self.chat_id = chat_id

    def format_signal_message(self, msg: SignalMessage) -> str:
        """시그널 메시지 포맷 (인라인 버튼과 함께 사용)."""
        s = msg.signal
        if s.action == SignalAction.ENTER_LONG:
            icon = "\U0001f4c8"
            direction = "롱"
        else:
            icon = "\U0001f4c9"
            direction = "숏"

        quality_str = _QUALITY_ICON.get(msg.quality, str(msg.quality.value))
        score = s.metadata.get("score", "?")

        lines = [
            f"<b>{icon} {direction} 시그널 | {s.symbol} (4H)</b>",
            "",
            f"진입가: {s.entry_price:,.0f}",
            f"손절가: {s.stop_loss:,.0f} ({_pct(s.entry_price, s.stop_loss)})",
            f"목표가: {s.take_profit:,.0f} ({_pct(s.entry_price, s.take_profit)})",
            f"R:R = 1:{msg.risk_reward_ratio}",
            f"품질: {quality_str} ({score}/7)",
            "",
            "<b>\U0001f4ca 근거:</b>",
        ]
        for reason in msg.explanation:
            lines.append(f"• {reason}")

        lines.append("")
        lines.append("\u23f0 1시간 내 응답")

        return "\n".join(lines)

    def format_monitoring_update(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> str:
        """모니터링 업데이트 메시지."""
        pnl_pct = (current_price - entry_price) / entry_price * 100
        if direction == "short":
            pnl_pct = -pnl_pct

        sl_dist = abs(current_price - stop_loss) / current_price * 100
        tp_dist = abs(take_profit - current_price) / current_price * 100

        icon = "\u2705" if pnl_pct > 0 else "\u26a0\ufe0f"

        return (
            f"<b>\U0001f4ca {symbol} {'롱' if direction == 'long' else '숏'} 현황</b>\n\n"
            f"현재가: {current_price:,.0f} ({_sign(pnl_pct)}{pnl_pct:.2f}%)\n"
            f"손절까지: -{sl_dist:.2f}%\n"
            f"목표까지: +{tp_dist:.2f}%\n\n"
            f"상태: 홀딩 {icon}"
        )

    def format_exit_signal(self, symbol: str, direction: str, reason: str) -> str:
        """청산 시그널 메시지."""
        return (
            f"<b>\U0001f6aa 청산 시그널 | {symbol}</b>\n\n"
            f"방향: {'롱' if direction == 'long' else '숏'}\n"
            f"사유: {reason}\n\n"
            "\u23f0 응답해주세요"
        )

    def format_weekly_accuracy(self, report: dict) -> str:
        """주간 시그널 정확도 리포트."""
        total = report.get("total", 0)
        if total == 0:
            return "<b>\U0001f4cb 주간 시그널 리포트</b>\n\n이번 주 발송된 시그널이 없습니다."

        tp_rate = report.get("tp_rate", 0) * 100
        sl_rate = report.get("sl_rate", 0) * 100

        lines = [
            "<b>\U0001f4cb 주간 시그널 리포트</b>",
            "",
            f"총 시그널: {total}개",
            f"목표가 도달: {tp_rate:.0f}%",
            f"손절 도달: {sl_rate:.0f}%",
            "",
        ]

        by_quality = report.get("by_quality", {})
        for quality, stats in by_quality.items():
            q_total = stats["total"]
            q_tp = stats["tp"] / q_total * 100 if q_total > 0 else 0
            lines.append(f"• {quality}: {q_total}개 (TP {q_tp:.0f}%)")

        return "\n".join(lines)

    def get_signal_keyboard(self):
        """시그널 응답 인라인 키보드."""
        if not HAS_TELEGRAM:
            return None
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("잡았다 \u2705", callback_data="entered"),
                InlineKeyboardButton("패스 \u274c", callback_data="pass"),
            ]
        ])

    def get_exit_keyboard(self):
        """청산 응답 인라인 키보드."""
        if not HAS_TELEGRAM:
            return None
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("팔았다 \u2705", callback_data="exited"),
                InlineKeyboardButton("홀딩 계속 \u23f3", callback_data="hold"),
            ]
        ])

    async def send_signal(self, msg: SignalMessage) -> None:
        text = self.format_signal_message(msg)
        keyboard = self.get_signal_keyboard()
        await self._send(text, reply_markup=keyboard)

    async def send_monitoring_update(self, symbol: str, direction: str, entry: float, current: float, sl: float, tp: float) -> None:
        text = self.format_monitoring_update(symbol, direction, entry, current, sl, tp)
        await self._send(text)

    async def send_exit_signal(self, symbol: str, direction: str, reason: str) -> None:
        text = self.format_exit_signal(symbol, direction, reason)
        keyboard = self.get_exit_keyboard()
        await self._send(text, reply_markup=keyboard)

    async def send_weekly_accuracy(self, report: dict) -> None:
        text = self.format_weekly_accuracy(report)
        await self._send(text)

    async def send_alert(self, message: str) -> None:
        await self._send(f"<b>\U0001f6a8 알림</b>\n{message}")

    async def _send(self, text: str, reply_markup=None) -> None:
        if self.sender and self.chat_id:
            try:
                await self.sender.send_message(self.chat_id, text, parse_mode="HTML", reply_markup=reply_markup)
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {e}")
        else:
            logger.info(f"[NO TELEGRAM] {text[:100]}...")

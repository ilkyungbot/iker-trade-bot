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

    def format_hourly_briefing(self, briefing: dict) -> str:
        """1시간 브리핑 메시지 포맷."""
        now_str = briefing.get("time", "")
        market_summary = briefing.get("market_summary", {})
        scored_coins = briefing.get("scored_coins", [])
        funding_alerts = briefing.get("funding_alerts", [])
        watched_pairs = briefing.get("watched_pairs", [])

        lines = [
            f"<b>\U0001f4e1 시장 브리핑</b>  {now_str}",
            "",
        ]

        # 주요 코인 현황
        if market_summary:
            lines.append("<b>\U0001f4b0 주요 코인</b>")
            for coin in market_summary.get("top_coins", []):
                symbol = coin["symbol"].replace("USDT", "")
                price = coin["price"]
                chg_1h = coin.get("change_1h", 0)
                chg_24h = coin.get("change_24h", 0)
                chg_icon = "\U0001f7e2" if chg_1h >= 0 else "\U0001f534"
                vol_str = f"${coin['volume_24h']/1e9:.1f}B" if coin['volume_24h'] >= 1e9 else f"${coin['volume_24h']/1e6:.0f}M"

                # 거래소와 동일한 소수점 표시
                if price >= 100:
                    price_str = f"{price:,.2f}"
                elif price >= 1:
                    price_str = f"{price:.4f}"
                else:
                    price_str = f"{price:.6f}"

                lines.append(
                    f"  {chg_icon} <b>{symbol}</b> ${price_str}"
                )
                lines.append(
                    f"      1h {chg_1h:+.2f}% | 24h {chg_24h:+.2f}% | {vol_str}"
                )
            lines.append("")

        # 스코어 감지된 코인
        if scored_coins:
            lines.append("<b>\U0001f3af 시그널 감지 코인</b>")
            for sc in scored_coins:
                symbol = sc["symbol"].replace("USDT", "")
                direction = "\U0001f4c8롱" if sc["direction"] == "long" else "\U0001f4c9숏"
                score = sc["score"]
                quality = sc["quality"]
                q_icon = "\U0001f7e2" if quality == "strong" else "\U0001f7e1" if quality == "moderate" else "\U0001f534"
                reasons_str = " / ".join(sc["reasons"][:3])
                lines.append(f"  {q_icon} <b>{symbol}</b> {direction} ({score}/7점)")
                lines.append(f"    └ {reasons_str}")
            lines.append("")
        else:
            lines.append("<b>\U0001f3af 시그널 감지 코인</b>")
            lines.append("  감지된 시그널 없음")
            lines.append("")

        # 펀딩비 이상
        if funding_alerts:
            lines.append("<b>\U0001f4b8 펀딩비 주의</b>")
            for fa in funding_alerts:
                symbol = fa["symbol"].replace("USDT", "")
                rate = fa["rate"]
                direction = "롱과열" if rate > 0 else "숏과열"
                lines.append(f"  {symbol}: {rate*100:+.4f}% ({direction})")
            lines.append("")

        # 관찰 중인 페어
        if watched_pairs:
            pair_str = ", ".join(p.replace("USDT", "") for p in watched_pairs)
            lines.append(f"<b>\U0001f440 관찰 페어</b>: {pair_str}")

        return "\n".join(lines)

    def format_coin_analysis(self, analysis: dict) -> str:
        """단일 코인 심층 분석 메시지 포맷."""
        sym = analysis["symbol"].replace("USDT", "")
        price = analysis["price"]
        ind = analysis["indicators"]

        # 가격 포맷
        if price >= 100:
            price_str = f"${price:,.2f}"
        elif price >= 1:
            price_str = f"${price:.4f}"
        else:
            price_str = f"${price:.6f}"

        lines = [
            f"<b>\U0001f50e {sym} 심층 분석</b>",
            "",
            f"<b>\U0001f4b0 현재가</b>: {price_str}",
            f"  1h {analysis['change_1h']:+.2f}% | 24h {analysis['change_24h']:+.2f}%",
            f"  24h 고가 {analysis['high_24h']:,.2f} / 저가 {analysis['low_24h']:,.2f}",
            "",
            "<b>\U0001f4ca 기술적 지표</b>",
        ]

        # RSI
        rsi = ind.get("rsi")
        if rsi is not None:
            if rsi >= 70:
                rsi_label = "과매수 \u26a0\ufe0f"
            elif rsi <= 30:
                rsi_label = "과매도 \u26a0\ufe0f"
            elif rsi >= 60:
                rsi_label = "강세"
            elif rsi <= 40:
                rsi_label = "약세"
            else:
                rsi_label = "중립"
            lines.append(f"  RSI: {rsi} ({rsi_label})")

        # ADX
        adx = ind.get("adx")
        if adx is not None:
            if adx >= 40:
                adx_label = "강한 추세"
            elif adx >= 20:
                adx_label = "추세 있음"
            else:
                adx_label = "추세 약함"
            lines.append(f"  ADX: {adx} ({adx_label})")

        # MACD
        macd = ind.get("macd_hist")
        if macd is not None:
            macd_label = "상승 모멘텀" if macd > 0 else "하락 모멘텀"
            lines.append(f"  MACD 히스토그램: {macd:+.4f} ({macd_label})")

        # EMA 배열
        lines.append(f"  EMA: {analysis['ema_position']}")

        # 볼린저
        bb = ind.get("bb_width")
        if bb is not None:
            bb_label = "변동성 확대" if bb > 0.1 else "수축 (돌파 임박)" if bb < 0.03 else "보통"
            lines.append(f"  볼린저 폭: {bb:.4f} ({bb_label})")

        # 거래량
        vol_ratio = ind.get("volume_ratio")
        if vol_ratio is not None:
            vol_label = f"평균 대비 {vol_ratio}배"
            if vol_ratio >= 2:
                vol_label += " \U0001f525"
            lines.append(f"  거래량: {vol_label}")

        # 횡보 여부
        if ind.get("is_sideways"):
            lines.append(f"  \u26a0\ufe0f 횡보장 감지")

        # 펀딩비
        fr = analysis.get("funding_rate", 0)
        if fr != 0:
            fr_label = "롱과열" if fr > 0.0005 else "숏과열" if fr < -0.0005 else "중립"
            lines.append(f"  펀딩비: {fr*100:+.4f}% ({fr_label})")

        lines.append("")

        # 스코어링
        score = analysis["score"]
        direction = analysis["direction"]
        dir_kr = "\U0001f4c8 롱" if direction == "long" else "\U0001f4c9 숏" if direction == "short" else "\u23f8 중립"

        score_bar = "\U0001f7e2" * score + "\u26aa" * (7 - score)
        lines.append(f"<b>\U0001f3af 스코어링</b>: {dir_kr} {score}/7")
        lines.append(f"  {score_bar}")

        if analysis["reasons"]:
            for r in analysis["reasons"]:
                lines.append(f"  \u2022 {r}")

        lines.append("")

        # 최종 판단
        verdict = analysis["verdict"]
        verdict_reason = analysis["verdict_reason"]

        if "적극" in verdict:
            v_icon = "\U0001f7e2\U0001f7e2"
        elif "고려" in verdict:
            v_icon = "\U0001f7e2"
        elif "조건부" in verdict:
            v_icon = "\U0001f7e1"
        else:
            v_icon = "\u26aa"

        lines.append(f"<b>{v_icon} 판단: {verdict}</b>")
        lines.append(f"  {verdict_reason}")

        # TP/SL
        sl = analysis.get("sl")
        tp = analysis.get("tp")
        if sl is not None and tp is not None:
            entry = analysis["entry"]
            lines.append("")
            lines.append("<b>\U0001f4cd 진입 시 참고</b>")

            if entry >= 100:
                lines.append(f"  진입가: {entry:,.2f}")
                lines.append(f"  손절가: {sl:,.2f} ({_pct(entry, sl)})")
                lines.append(f"  목표가: {tp:,.2f} ({_pct(entry, tp)})")
            else:
                lines.append(f"  진입가: {entry:.4f}")
                lines.append(f"  손절가: {sl:.4f} ({_pct(entry, sl)})")
                lines.append(f"  목표가: {tp:.4f} ({_pct(entry, tp)})")

        return "\n".join(lines)

    async def send_coin_analysis(self, analysis: dict) -> None:
        text = self.format_coin_analysis(analysis)
        await self._send(text)

    async def send_hourly_briefing(self, briefing: dict) -> None:
        text = self.format_hourly_briefing(briefing)
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

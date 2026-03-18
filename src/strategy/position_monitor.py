"""
포지션 모니터링 — 이벤트 감지 엔진.
활성 포지션별로 기술 지표 + 레버리지 맥락을 분석하여
6종 이벤트(상승/하락 신호, 포지션 체크, 매도/홀딩/매수 추천)를 감지.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
import pandas as pd
from core.types import ManualPosition, Side

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionEvent:
    event_type: str  # bullish_signal, bearish_signal, position_check, sell_recommendation, hold_recommendation, buy_recommendation
    position_id: int
    symbol: str
    message: str
    severity: str  # info, warning, critical
    pnl_percent: float


class PositionMonitor:
    def __init__(self):
        self._last_events: dict[int, dict[str, datetime]] = {}

    def detect_events(self, position: ManualPosition, df: pd.DataFrame, current_price: float, funding_rate: float) -> list[PositionEvent]:
        events: list[PositionEvent] = []
        if len(df) < 2:
            return events

        current = df.iloc[-1]
        pnl_pct = self._calc_pnl_pct(position, current_price)
        liq_distance = self._calc_liquidation_distance(position, current_price)

        # 1. 포지션 체크 (청산가 접근)
        if liq_distance < 30:
            severity = "critical" if liq_distance < 15 else "warning"
            events.append(PositionEvent(
                event_type="position_check", position_id=position.id, symbol=position.symbol,
                message=f"청산가 접근 주의! 잔여 마진 약 {liq_distance:.1f}% (PnL: {pnl_pct:+.1f}%)",
                severity=severity, pnl_percent=pnl_pct,
            ))

        # 2. 상승 신호
        bullish_reasons = self._check_bullish(current, df)
        if len(bullish_reasons) >= 2:
            events.append(PositionEvent(
                event_type="bullish_signal", position_id=position.id, symbol=position.symbol,
                message="상승 신호: " + " / ".join(bullish_reasons),
                severity="info", pnl_percent=pnl_pct,
            ))

        # 3. 하락 신호
        bearish_reasons = self._check_bearish(current, df)
        if len(bearish_reasons) >= 2:
            events.append(PositionEvent(
                event_type="bearish_signal", position_id=position.id, symbol=position.symbol,
                message="하락 신호: " + " / ".join(bearish_reasons),
                severity="warning" if position.side == Side.LONG else "info",
                pnl_percent=pnl_pct,
            ))

        # 4. 매도 추천
        sell_reasons = self._check_sell(position, current, pnl_pct, funding_rate)
        if sell_reasons:
            events.append(PositionEvent(
                event_type="sell_recommendation", position_id=position.id, symbol=position.symbol,
                message="매도 추천: " + " / ".join(sell_reasons),
                severity="warning", pnl_percent=pnl_pct,
            ))

        # 5. 매수 추천 (물타기)
        buy_reasons = self._check_buy(position, current, df, pnl_pct)
        if buy_reasons:
            events.append(PositionEvent(
                event_type="buy_recommendation", position_id=position.id, symbol=position.symbol,
                message="매수(물타기) 추천: " + " / ".join(buy_reasons),
                severity="info", pnl_percent=pnl_pct,
            ))

        # 6. 홀딩 추천
        if not sell_reasons and not buy_reasons and abs(pnl_pct) < 30:
            hold_reasons = self._check_hold(position, current, pnl_pct)
            if hold_reasons:
                events.append(PositionEvent(
                    event_type="hold_recommendation", position_id=position.id, symbol=position.symbol,
                    message="홀딩 추천: " + " / ".join(hold_reasons),
                    severity="info", pnl_percent=pnl_pct,
                ))

        events = self._filter_cooldown(position.id, events)
        return events

    def _filter_cooldown(self, position_id: int, events: list[PositionEvent], cooldown_minutes: int = 30) -> list[PositionEvent]:
        """동일 이벤트 타입에 대해 쿨다운 적용."""
        now = datetime.now(timezone.utc)
        if position_id not in self._last_events:
            self._last_events[position_id] = {}

        filtered = []
        for event in events:
            last_time = self._last_events[position_id].get(event.event_type)
            # critical은 쿨다운 10분, 나머지 30분
            cd = 10 if event.severity == "critical" else cooldown_minutes
            if last_time is None or (now - last_time).total_seconds() >= cd * 60:
                filtered.append(event)
                self._last_events[position_id][event.event_type] = now

        return filtered

    def clear_position(self, position_id: int) -> None:
        """포지션 청산 시 쿨다운 기록 제거."""
        self._last_events.pop(position_id, None)

    def _calc_pnl_pct(self, pos: ManualPosition, current_price: float) -> float:
        if pos.entry_price == 0:
            return 0.0
        price_change_pct = (current_price - pos.entry_price) / pos.entry_price * 100
        if pos.side == Side.SHORT:
            price_change_pct = -price_change_pct
        return price_change_pct * pos.leverage

    def _calc_liquidation_distance(self, pos: ManualPosition, current_price: float) -> float:
        if pos.leverage == 0 or pos.entry_price == 0:
            return 0.0
        if pos.side == Side.LONG:
            liq_price = pos.entry_price * (1 - 1 / pos.leverage)
            if current_price <= liq_price:
                return 0
            return (current_price - liq_price) / (pos.entry_price - liq_price) * 100
        else:
            liq_price = pos.entry_price * (1 + 1 / pos.leverage)
            if current_price >= liq_price:
                return 0
            return (liq_price - current_price) / (liq_price - pos.entry_price) * 100

    def _check_bullish(self, current, df) -> list[str]:
        reasons = []
        if current.get("ema_golden_cross", False):
            reasons.append("EMA 골든크로스")
        if current.get("rsi_cross_up", False):
            reasons.append("RSI 상향돌파")
        if current.get("macd_hist_cross_up", False):
            reasons.append("MACD 양전환")
        vol_ratio = current.get("volume_ratio", 1.0)
        if vol_ratio >= 2.0 and current.get("close", 0) > current.get("open", 0):
            reasons.append(f"거래량 급증 {vol_ratio:.1f}배")
        rsi = current.get("rsi", 50)
        if rsi <= 30:
            reasons.append(f"RSI 과매도 ({rsi:.0f})")
        return reasons

    def _check_bearish(self, current, df) -> list[str]:
        reasons = []
        if current.get("ema_death_cross", False):
            reasons.append("EMA 데드크로스")
        if current.get("rsi_cross_down", False):
            reasons.append("RSI 하향돌파")
        if current.get("macd_hist_cross_down", False):
            reasons.append("MACD 음전환")
        vol_ratio = current.get("volume_ratio", 1.0)
        if vol_ratio >= 2.0 and current.get("close", 0) < current.get("open", 0):
            reasons.append(f"거래량 급증 {vol_ratio:.1f}배 (하락)")
        rsi = current.get("rsi", 50)
        if rsi >= 70:
            reasons.append(f"RSI 과매수 ({rsi:.0f})")
        return reasons

    def _check_sell(self, pos, current, pnl_pct, funding_rate) -> list[str]:
        reasons = []
        if pnl_pct >= 30:
            reasons.append(f"PnL {pnl_pct:+.1f}% 도달")
        if pos.side == Side.LONG and current.get("rsi", 50) >= 72 and pnl_pct > 10:
            reasons.append(f"RSI 과매수({current.get('rsi', 0):.0f}) + 수익 구간")
        if pos.side == Side.SHORT and current.get("rsi", 50) <= 28 and pnl_pct > 10:
            reasons.append(f"RSI 과매도({current.get('rsi', 0):.0f}) + 수익 구간")
        if pos.side == Side.LONG and funding_rate > 0.001:
            reasons.append(f"높은 펀딩비 ({funding_rate*100:.3f}%)")
        if pos.side == Side.SHORT and funding_rate < -0.001:
            reasons.append(f"높은 역펀딩비 ({funding_rate*100:.3f}%)")
        return reasons

    def _check_buy(self, pos, current, df, pnl_pct) -> list[str]:
        reasons = []
        if pos.side == Side.LONG and pnl_pct < -5:
            if current.get("rsi", 50) <= 30:
                reasons.append(f"RSI 과매도({current.get('rsi', 0):.0f})")
            if current.get("macd_hist_cross_up", False):
                reasons.append("MACD 반등 신호")
        elif pos.side == Side.SHORT and pnl_pct < -5:
            if current.get("rsi", 50) >= 70:
                reasons.append(f"RSI 과매수({current.get('rsi', 0):.0f})")
            if current.get("macd_hist_cross_down", False):
                reasons.append("MACD 하락 신호")
        return reasons

    def _check_hold(self, pos, current, pnl_pct) -> list[str]:
        reasons = []
        adx = current.get("adx", 0)
        ema_20 = current.get("ema_20", 0)
        ema_50 = current.get("ema_50", 0)
        close = current.get("close", 0)
        if pos.side == Side.LONG:
            if close > ema_20 > ema_50 and adx > 20:
                reasons.append("강세 정배열 유지")
            if 0 < pnl_pct < 20 and adx > 25:
                reasons.append(f"추세 강도 양호 (ADX {adx:.0f})")
        else:
            if close < ema_20 < ema_50 and adx > 20:
                reasons.append("약세 역배열 유지")
            if 0 < pnl_pct < 20 and adx > 25:
                reasons.append(f"추세 강도 양호 (ADX {adx:.0f})")
        return reasons

"""
포지션 모니터링 v2 — 새 프레임워크 기반.
ExitManager, EdgeDetector, MarketRegimeClassifier를 통합하여
7종 이벤트를 감지.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from core.types import ManualPosition, Side, FundingRate, OpenInterest
from execution.exit_manager import ExitManager, ExitSignal
from strategy.edge_detector import EdgeDetector, EdgeSignal
from strategy.market_regime import MarketRegimeClassifier, RegimeState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionEvent:
    event_type: str
    position_id: int
    symbol: str
    message: str
    severity: str
    pnl_percent: float


class PositionMonitorV2:
    """통합 포지션 모니터. ExitManager + EdgeDetector + MarketRegime."""

    def __init__(self):
        self.exit_manager = ExitManager()
        self.edge_detector = EdgeDetector()
        self.regime_classifier = MarketRegimeClassifier()
        self._last_events: dict[int, dict[str, datetime]] = {}

    def check_position(
        self,
        position: ManualPosition,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        funding_rates: list[FundingRate] | None = None,
        oi_data: list[OpenInterest] | None = None,
        btc_df: pd.DataFrame | None = None,
    ) -> dict:
        """
        Comprehensive position check. Returns:
        {
            "exit_signals": list[ExitSignal],
            "edge_signals": list[EdgeSignal],
            "regime": RegimeState | None,
            "pnl_pct": float,
        }
        """
        result = {
            "exit_signals": [],
            "edge_signals": [],
            "regime": None,
            "pnl_pct": 0.0,
        }

        # PnL
        if position.entry_price > 0:
            pnl = (current_price - position.entry_price) / position.entry_price * 100
            if position.side == Side.SHORT:
                pnl = -pnl
            result["pnl_pct"] = pnl * position.leverage

        # Exit signals
        result["exit_signals"] = self.exit_manager.check_exits(position, current_price, atr)

        # Edge signals
        if funding_rates or oi_data:
            result["edge_signals"] = self.edge_detector.detect_all(
                funding_rates or [], oi_data or []
            )

        # Market regime
        if len(df) >= 50:
            result["regime"] = self.regime_classifier.classify(df, btc_df)

        # 쿨다운 필터: 동일 알림 반복 방지
        result["exit_signals"] = self._filter_exit_cooldown(position.id, result["exit_signals"])
        result["edge_signals"] = self._filter_edge_cooldown(position.id, result["edge_signals"])

        return result

    def _filter_exit_cooldown(self, position_id: int, signals: list[ExitSignal]) -> list[ExitSignal]:
        """ExitSignal 쿨다운. critical=10분, 나머지=30분."""
        now = datetime.now(timezone.utc)
        if position_id not in self._last_events:
            self._last_events[position_id] = {}

        filtered = []
        for sig in signals:
            key = f"exit:{sig.signal_type}"
            last = self._last_events[position_id].get(key)
            cd_minutes = 10 if sig.severity == "critical" else 30
            if last is None or (now - last).total_seconds() >= cd_minutes * 60:
                filtered.append(sig)
                self._last_events[position_id][key] = now
        return filtered

    def _filter_edge_cooldown(self, position_id: int, signals: list[EdgeSignal]) -> list[EdgeSignal]:
        """EdgeSignal 쿨다운. 동일 signal_type 30분."""
        now = datetime.now(timezone.utc)
        if position_id not in self._last_events:
            self._last_events[position_id] = {}

        filtered = []
        for sig in signals:
            key = f"edge:{sig.signal_type}"
            last = self._last_events[position_id].get(key)
            if last is None or (now - last).total_seconds() >= 30 * 60:
                filtered.append(sig)
                self._last_events[position_id][key] = now
        return filtered

    def clear_position(self, position_id: int) -> None:
        self.exit_manager.clear_position(position_id)
        self._last_events.pop(position_id, None)


# ---------------------------------------------------------------------------
# Legacy class — backward compatibility with existing tests
# ---------------------------------------------------------------------------


class PositionMonitorLegacy:
    """기존 PositionMonitor (레거시). 테스트 호환용으로 유지."""

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


# Backward-compatible alias
PositionMonitor = PositionMonitorLegacy

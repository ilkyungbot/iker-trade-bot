"""
Exit management: partial take-profit, trailing stop, time stop, SL warning.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from core.types import ManualPosition, Side


@dataclass(frozen=True)
class ExitSignal:
    signal_type: str  # "partial_tp", "trailing_stop", "time_stop", "sl_warning"
    position_id: int
    symbol: str
    message: str  # Korean description
    severity: str  # "info", "warning", "critical"
    pnl_percent: float
    suggested_action: str  # Korean action suggestion


class ExitManager:
    """출구 관리 엔진.

    Note: _partial_tp_triggered와 _trailing_highs는 메모리 기반.
    봇 재시작 시 리셋됨 — partial_tp가 중복 발생할 수 있음.
    TODO: DB 기반 영속화 필요.
    """

    def __init__(self) -> None:
        self._partial_tp_triggered: set[int] = set()
        self._trailing_highs: dict[int, float] = {}

    def check_exits(
        self, position: ManualPosition, current_price: float, atr: float
    ) -> list[ExitSignal]:
        """Check all exit conditions for a position."""
        signals: list[ExitSignal] = []

        if position.stop_loss is None or position.take_profit is None:
            return signals

        pnl_pct = self._calc_pnl_pct(position, current_price)

        # 1. SL warning (approaching stop loss)
        sl_distance_pct = self._calc_sl_distance_pct(position, current_price)
        if sl_distance_pct == 0:
            severity = "critical"
            signals.append(
                ExitSignal(
                    signal_type="sl_warning",
                    position_id=position.id,
                    symbol=position.symbol,
                    message="손절가 도달/돌파! 즉시 청산하세요.",
                    severity=severity,
                    pnl_percent=pnl_pct,
                    suggested_action="지금 바로 청산하세요. 손절가를 이미 지났습니다.",
                )
            )
        elif sl_distance_pct < 3.0:
            severity = "critical" if sl_distance_pct < 1.0 else "warning"
            signals.append(
                ExitSignal(
                    signal_type="sl_warning",
                    position_id=position.id,
                    symbol=position.symbol,
                    message=f"손절가까지 {sl_distance_pct:.1f}% 남음",
                    severity=severity,
                    pnl_percent=pnl_pct,
                    suggested_action="손절가 도달 시 즉시 청산하세요.",
                )
            )

        # 2. Partial TP at 1.5R
        if position.id not in self._partial_tp_triggered:
            risk = abs(position.entry_price - position.stop_loss)
            reward_1_5r = risk * 1.5
            if position.side == Side.LONG:
                tp_1_5r = position.entry_price + reward_1_5r
                hit = current_price >= tp_1_5r
            else:
                tp_1_5r = position.entry_price - reward_1_5r
                hit = current_price <= tp_1_5r

            if hit:
                self._partial_tp_triggered.add(position.id)
                signals.append(
                    ExitSignal(
                        signal_type="partial_tp",
                        position_id=position.id,
                        symbol=position.symbol,
                        message=f"1차 목표(1.5R) 도달! 현재 PnL {pnl_pct:+.1f}%",
                        severity="info",
                        pnl_percent=pnl_pct,
                        suggested_action="포지션 50% 익절 + 손절가를 본전으로 이동 권장",
                    )
                )

        # 3. Trailing stop (after partial TP triggered)
        if position.id in self._partial_tp_triggered:
            self._update_trailing(position, current_price)
            trailing_sl = self._calc_trailing_sl(position, atr)
            if trailing_sl is not None:
                triggered = False
                if position.side == Side.LONG and current_price <= trailing_sl:
                    triggered = True
                elif position.side == Side.SHORT and current_price >= trailing_sl:
                    triggered = True

                if triggered:
                    signals.append(
                        ExitSignal(
                            signal_type="trailing_stop",
                            position_id=position.id,
                            symbol=position.symbol,
                            message=f"트레일링 스탑 도달! 현재 PnL {pnl_pct:+.1f}%",
                            severity="critical",
                            pnl_percent=pnl_pct,
                            suggested_action="잔여 포지션 전량 청산 권장",
                        )
                    )

        # 4. Time stop: 12h without reaching 0.5R
        if position.created_at:
            elapsed = datetime.now(timezone.utc) - position.created_at
            if elapsed > timedelta(hours=12):
                risk = abs(position.entry_price - position.stop_loss)
                half_r = risk * 0.5
                if position.side == Side.LONG:
                    target = position.entry_price + half_r
                    reached = current_price >= target
                else:
                    target = position.entry_price - half_r
                    reached = current_price <= target

                if not reached:
                    signals.append(
                        ExitSignal(
                            signal_type="time_stop",
                            position_id=position.id,
                            symbol=position.symbol,
                            message="12시간 경과, 0.5R 미도달",
                            severity="warning",
                            pnl_percent=pnl_pct,
                            suggested_action="포지션 재검토 권장. 논리가 유효한지 확인하세요.",
                        )
                    )

        return signals

    def _calc_pnl_pct(self, pos: ManualPosition, current_price: float) -> float:
        if pos.entry_price == 0:
            return 0.0
        pct = (current_price - pos.entry_price) / pos.entry_price * 100
        if pos.side == Side.SHORT:
            pct = -pct
        return pct * pos.leverage

    def _calc_sl_distance_pct(
        self, pos: ManualPosition, current_price: float
    ) -> float:
        """Distance to SL as % of current price."""
        if pos.side == Side.LONG:
            dist = (current_price - pos.stop_loss) / current_price * 100
        else:
            dist = (pos.stop_loss - current_price) / current_price * 100
        return max(0, dist)  # Clamp: if SL already breached, return 0

    def _update_trailing(
        self, pos: ManualPosition, current_price: float
    ) -> None:
        """Track highest favorable price."""
        current_best = self._trailing_highs.get(pos.id)
        if pos.side == Side.LONG:
            if current_best is None or current_price > current_best:
                self._trailing_highs[pos.id] = current_price
        else:
            if current_best is None or current_price < current_best:
                self._trailing_highs[pos.id] = current_price

    def _calc_trailing_sl(
        self, pos: ManualPosition, atr: float
    ) -> float | None:
        """Trailing SL = best price - 2*ATR (LONG) or + 2*ATR (SHORT)."""
        best = self._trailing_highs.get(pos.id)
        if best is None:
            return None
        if pos.side == Side.LONG:
            return best - 2 * atr
        else:
            return best + 2 * atr

    def clear_position(self, position_id: int) -> None:
        self._partial_tp_triggered.discard(position_id)
        self._trailing_highs.pop(position_id, None)

    def get_state(self) -> dict:
        """현재 메모리 상태를 직렬화 가능한 dict로 반환 (영속화용)."""
        return {
            "partial_tp_triggered": list(self._partial_tp_triggered),
            "trailing_highs": dict(self._trailing_highs),
        }

    def restore_state(self, state: dict) -> None:
        """저장된 상태를 복원."""
        self._partial_tp_triggered = set(state.get("partial_tp_triggered", []))
        self._trailing_highs = dict(state.get("trailing_highs", {}))

"""Tests for ExitManager – partial TP, trailing stop, time stop."""

from datetime import datetime, timezone, timedelta

import pytest

from core.types import ManualPosition, Side
from execution.exit_manager import ExitManager, ExitSignal


def _long_pos(
    entry: float = 100.0,
    sl: float = 95.0,
    tp: float = 115.0,
    leverage: float = 1.0,
    created_at: datetime | None = None,
    pos_id: int = 1,
) -> ManualPosition:
    return ManualPosition(
        id=pos_id,
        chat_id="test",
        symbol="BTCUSDT",
        side=Side.LONG,
        entry_price=entry,
        leverage=leverage,
        created_at=created_at or datetime.now(timezone.utc),
        stop_loss=sl,
        take_profit=tp,
    )


def _short_pos(
    entry: float = 100.0,
    sl: float = 105.0,
    tp: float = 85.0,
    leverage: float = 1.0,
    created_at: datetime | None = None,
    pos_id: int = 2,
) -> ManualPosition:
    return ManualPosition(
        id=pos_id,
        chat_id="test",
        symbol="ETHUSDT",
        side=Side.SHORT,
        entry_price=entry,
        leverage=leverage,
        created_at=created_at or datetime.now(timezone.utc),
        stop_loss=sl,
        take_profit=tp,
    )


class TestSLWarning:
    def test_sl_warning_close(self):
        """SL까지 3% 미만이면 warning 시그널 발생."""
        mgr = ExitManager()
        pos = _long_pos(entry=100, sl=95)
        # current=97 → SL distance = (97-95)/97 ≈ 2.06%
        signals = mgr.check_exits(pos, current_price=97.0, atr=2.0)
        sl_warnings = [s for s in signals if s.signal_type == "sl_warning"]
        assert len(sl_warnings) == 1
        assert sl_warnings[0].severity == "warning"

    def test_sl_warning_critical(self):
        """SL까지 1% 미만이면 critical."""
        mgr = ExitManager()
        pos = _long_pos(entry=100, sl=95)
        # current=95.5 → SL distance = (95.5-95)/95.5 ≈ 0.52%
        signals = mgr.check_exits(pos, current_price=95.5, atr=2.0)
        sl_warnings = [s for s in signals if s.signal_type == "sl_warning"]
        assert len(sl_warnings) == 1
        assert sl_warnings[0].severity == "critical"


class TestPartialTP:
    def test_partial_tp_long(self):
        """LONG: 1.5R 도달 시 partial_tp 시그널."""
        mgr = ExitManager()
        pos = _long_pos(entry=100, sl=95, tp=115)
        # risk=5, 1.5R=7.5, target=107.5
        signals = mgr.check_exits(pos, current_price=108.0, atr=2.0)
        tp_signals = [s for s in signals if s.signal_type == "partial_tp"]
        assert len(tp_signals) == 1
        assert tp_signals[0].severity == "info"
        assert "1.5R" in tp_signals[0].message

    def test_partial_tp_short(self):
        """SHORT: 1.5R 도달 시 partial_tp 시그널."""
        mgr = ExitManager()
        pos = _short_pos(entry=100, sl=105, tp=85)
        # risk=5, 1.5R=7.5, target=92.5
        signals = mgr.check_exits(pos, current_price=92.0, atr=2.0)
        tp_signals = [s for s in signals if s.signal_type == "partial_tp"]
        assert len(tp_signals) == 1

    def test_partial_tp_only_triggers_once(self):
        """같은 포지션에서 partial_tp는 한 번만 발생."""
        mgr = ExitManager()
        pos = _long_pos(entry=100, sl=95, tp=115)
        # First call at 108 → triggers
        signals1 = mgr.check_exits(pos, current_price=108.0, atr=2.0)
        assert any(s.signal_type == "partial_tp" for s in signals1)
        # Second call at 110 → does NOT trigger again
        signals2 = mgr.check_exits(pos, current_price=110.0, atr=2.0)
        assert not any(s.signal_type == "partial_tp" for s in signals2)


class TestTrailingStop:
    def test_trailing_stop_long(self):
        """LONG: partial TP 후 trailing stop 도달 시 시그널."""
        mgr = ExitManager()
        pos = _long_pos(entry=100, sl=95, tp=115)
        # Trigger partial TP first at 108
        mgr.check_exits(pos, current_price=108.0, atr=2.0)
        # Price rises to 112 (sets trailing high)
        mgr.check_exits(pos, current_price=112.0, atr=2.0)
        # Trailing SL = 112 - 2*2 = 108. Price drops to 107.5 → trailing stop hit
        signals = mgr.check_exits(pos, current_price=107.5, atr=2.0)
        trailing = [s for s in signals if s.signal_type == "trailing_stop"]
        assert len(trailing) == 1
        assert trailing[0].severity == "critical"

    def test_trailing_stop_short(self):
        """SHORT: partial TP 후 trailing stop 도달 시 시그널."""
        mgr = ExitManager()
        pos = _short_pos(entry=100, sl=105, tp=85)
        # Trigger partial TP at 92
        mgr.check_exits(pos, current_price=92.0, atr=2.0)
        # Price drops to 88 (trailing low)
        mgr.check_exits(pos, current_price=88.0, atr=2.0)
        # Trailing SL = 88 + 2*2 = 92. Price rises to 93 → trailing stop hit
        signals = mgr.check_exits(pos, current_price=93.0, atr=2.0)
        trailing = [s for s in signals if s.signal_type == "trailing_stop"]
        assert len(trailing) == 1


class TestTimeStop:
    def test_time_stop_12h(self):
        """12시간 경과 + 0.5R 미도달 시 time_stop."""
        mgr = ExitManager()
        old_time = datetime.now(timezone.utc) - timedelta(hours=13)
        pos = _long_pos(entry=100, sl=95, tp=115, created_at=old_time)
        # 0.5R = 2.5, target = 102.5.  current=101 → not reached
        signals = mgr.check_exits(pos, current_price=101.0, atr=2.0)
        time_signals = [s for s in signals if s.signal_type == "time_stop"]
        assert len(time_signals) == 1
        assert "12시간" in time_signals[0].message

    def test_no_time_stop_when_target_reached(self):
        """12시간 경과해도 0.5R 이상이면 time_stop 없음."""
        mgr = ExitManager()
        old_time = datetime.now(timezone.utc) - timedelta(hours=13)
        pos = _long_pos(entry=100, sl=95, tp=115, created_at=old_time)
        # 0.5R target = 102.5, current=103 → reached
        signals = mgr.check_exits(pos, current_price=103.0, atr=2.0)
        time_signals = [s for s in signals if s.signal_type == "time_stop"]
        assert len(time_signals) == 0


class TestSLBreached:
    def test_sl_breached_critical(self):
        """SL 돌파 시 distance=0, critical 시그널."""
        em = ExitManager()
        pos = _long_pos(entry=67500, sl=66000, tp=70000)
        signals = em.check_exits(pos, current_price=65500, atr=500)  # Below SL
        assert any(s.signal_type == "sl_warning" and s.severity == "critical" for s in signals)
        assert any("도달/돌파" in s.message for s in signals)

    def test_sl_breached_short(self):
        """SHORT: SL 돌파 시 distance=0, critical 시그널."""
        em = ExitManager()
        pos = _short_pos(entry=100, sl=105, tp=85)
        signals = em.check_exits(pos, current_price=106, atr=2.0)  # Above SL
        assert any(s.signal_type == "sl_warning" and s.severity == "critical" for s in signals)
        assert any("도달/돌파" in s.message for s in signals)


class TestEdgeCases:
    def test_no_signals_when_no_sl_tp(self):
        """SL/TP 없으면 시그널 없음."""
        mgr = ExitManager()
        pos = ManualPosition(
            id=99, chat_id="test", symbol="BTCUSDT", side=Side.LONG,
            entry_price=100, leverage=1, created_at=datetime.now(timezone.utc),
            stop_loss=None, take_profit=None,
        )
        signals = mgr.check_exits(pos, current_price=50.0, atr=2.0)
        assert signals == []

    def test_clear_position(self):
        """clear_position 후 partial_tp 다시 발생 가능."""
        mgr = ExitManager()
        pos = _long_pos(entry=100, sl=95, tp=115)
        mgr.check_exits(pos, current_price=108.0, atr=2.0)
        mgr.clear_position(pos.id)
        signals = mgr.check_exits(pos, current_price=108.0, atr=2.0)
        assert any(s.signal_type == "partial_tp" for s in signals)

"""Tests for conversation state machine."""

import os
import tempfile
import pytest
from datetime import datetime, timezone, timedelta

from conversation.state_machine import ConversationStateMachine
from core.types import (
    ConversationState, Signal, SignalAction, SignalMessage,
    SignalQuality, StrategyName,
)


def _make_signal_msg() -> SignalMessage:
    s = Signal(
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        action=SignalAction.ENTER_LONG,
        strategy=StrategyName.TREND_FOLLOWING,
        entry_price=67450.0,
        stop_loss=66800.0,
        take_profit=68750.0,
        confidence=0.57,
        metadata={"score": 4},
    )
    return SignalMessage(
        signal=s,
        quality=SignalQuality.STRONG,
        explanation=["EMA 골든크로스", "MACD 양전환"],
        indicators={"adx": 28.5},
        risk_reward_ratio=2.0,
    )


@pytest.fixture
def sm(tmp_path):
    db_path = str(tmp_path / "test.db")
    return ConversationStateMachine(db_path=db_path)


class TestNormalFlow:
    def test_idle_to_signal_sent(self, sm):
        msg = _make_signal_msg()
        assert sm.send_signal("123", msg)
        session = sm.get_session("123")
        assert session.state == ConversationState.SIGNAL_SENT
        assert session.active_signal is not None

    def test_signal_sent_to_monitoring(self, sm):
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        assert sm.user_entered("123", entry_price=67500.0)
        session = sm.get_session("123")
        assert session.state == ConversationState.MONITORING
        assert session.user_entry_price == 67500.0

    def test_signal_sent_to_pass(self, sm):
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        assert sm.user_passed("123")
        session = sm.get_session("123")
        assert session.state == ConversationState.IDLE
        assert session.active_signal is None

    def test_monitoring_to_exit_signal(self, sm):
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        sm.user_entered("123")
        assert sm.send_exit_signal("123")
        session = sm.get_session("123")
        assert session.state == ConversationState.EXIT_SIGNAL_SENT

    def test_exit_signal_to_exited(self, sm):
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        sm.user_entered("123")
        sm.send_exit_signal("123")
        assert sm.user_exited("123")
        session = sm.get_session("123")
        assert session.state == ConversationState.IDLE

    def test_exit_signal_to_hold(self, sm):
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        sm.user_entered("123")
        sm.send_exit_signal("123")
        assert sm.user_hold("123")
        session = sm.get_session("123")
        assert session.state == ConversationState.MONITORING

    def test_monitoring_early_exit(self, sm):
        """모니터링 중 조기 청산."""
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        sm.user_entered("123")
        assert sm.user_exited("123")
        session = sm.get_session("123")
        assert session.state == ConversationState.IDLE


class TestInvalidTransitions:
    def test_cannot_enter_from_idle(self, sm):
        assert not sm.user_entered("123")

    def test_cannot_pass_from_monitoring(self, sm):
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        sm.user_entered("123")
        assert not sm.user_passed("123")

    def test_cannot_send_signal_when_not_idle(self, sm):
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        assert not sm.send_signal("123", msg)

    def test_cannot_hold_from_monitoring(self, sm):
        """홀딩은 EXIT_SIGNAL_SENT에서만 가능."""
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        sm.user_entered("123")
        assert not sm.user_hold("123")


class TestExpiry:
    def test_signal_expires(self, sm):
        # 시그널 시간을 2시간 전으로 설정
        s = Signal(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            symbol="BTCUSDT",
            action=SignalAction.ENTER_LONG,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=67450.0,
            stop_loss=66800.0,
            take_profit=68750.0,
            confidence=0.57,
        )
        msg = SignalMessage(
            signal=s,
            quality=SignalQuality.STRONG,
            explanation=["test"],
            indicators={},
            risk_reward_ratio=2.0,
        )
        sm.send_signal("123", msg)
        assert sm.check_expiry("123", expiry_minutes=60)
        session = sm.get_session("123")
        assert session.state == ConversationState.IDLE

    def test_signal_not_expired(self, sm):
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        assert not sm.check_expiry("123", expiry_minutes=60)


class TestPersistence:
    def test_state_survives_reload(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        sm1 = ConversationStateMachine(db_path=db_path)
        msg = _make_signal_msg()
        sm1.send_signal("123", msg)

        # 새 인스턴스로 로드
        sm2 = ConversationStateMachine(db_path=db_path)
        session = sm2.get_session("123")
        assert session.state == ConversationState.SIGNAL_SENT
        assert session.active_signal is not None
        assert session.active_signal.signal.symbol == "BTCUSDT"


class TestForceIdle:
    def test_force_idle(self, sm):
        msg = _make_signal_msg()
        sm.send_signal("123", msg)
        sm.user_entered("123")
        sm.force_idle("123")
        session = sm.get_session("123")
        assert session.state == ConversationState.IDLE

"""Tests for conversation state machine (v2 — simplified)."""

import pytest
from datetime import datetime, timezone

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
        explanation=["EMA golden cross", "MACD positive"],
        indicators={"adx": 28.5},
        risk_reward_ratio=2.0,
    )


@pytest.fixture
def sm(tmp_path):
    db_path = str(tmp_path / "test.db")
    return ConversationStateMachine(db_path=db_path)


class TestGetSession:
    def test_get_session_new_user(self, sm):
        """New user returns IDLE state."""
        session = sm.get_session("123")
        assert session.state == ConversationState.IDLE
        assert session.active_signal is None
        assert session.user_entry_price is None


class TestForceIdle:
    def test_force_idle(self, sm):
        """force_idle sets state to IDLE and clears signal data."""
        # Manually set to MONITORING via _save_session
        from core.types import UserSession
        session = UserSession(
            chat_id="123",
            state=ConversationState.MONITORING,
            active_signal=_make_signal_msg(),
            entry_confirmed_at=datetime.now(timezone.utc),
            user_entry_price=67500.0,
        )
        sm._save_session(session)

        sm.force_idle("123")
        session = sm.get_session("123")
        assert session.state == ConversationState.IDLE
        assert session.active_signal is None
        assert session.user_entry_price is None


class TestUserExited:
    def test_user_exited_from_monitoring(self, sm):
        """MONITORING -> IDLE via user_exited."""
        from core.types import UserSession
        session = UserSession(
            chat_id="123",
            state=ConversationState.MONITORING,
            active_signal=_make_signal_msg(),
            entry_confirmed_at=datetime.now(timezone.utc),
            user_entry_price=67500.0,
        )
        sm._save_session(session)

        assert sm.user_exited("123") is True
        session = sm.get_session("123")
        assert session.state == ConversationState.IDLE
        assert session.active_signal is None

    def test_user_exited_from_idle_fails(self, sm):
        """Cannot exit from IDLE — returns False."""
        assert sm.user_exited("123") is False
        session = sm.get_session("123")
        assert session.state == ConversationState.IDLE


class TestPersistence:
    def test_state_survives_reload(self, tmp_path):
        """State persists across new ConversationStateMachine instances."""
        from core.types import UserSession
        db_path = str(tmp_path / "test.db")

        sm1 = ConversationStateMachine(db_path=db_path)
        session = UserSession(
            chat_id="123",
            state=ConversationState.MONITORING,
            active_signal=_make_signal_msg(),
            entry_confirmed_at=datetime.now(timezone.utc),
            user_entry_price=67500.0,
        )
        sm1._save_session(session)

        sm2 = ConversationStateMachine(db_path=db_path)
        loaded = sm2.get_session("123")
        assert loaded.state == ConversationState.MONITORING
        assert loaded.active_signal is not None
        assert loaded.active_signal.signal.symbol == "BTCUSDT"
        assert loaded.user_entry_price == 67500.0

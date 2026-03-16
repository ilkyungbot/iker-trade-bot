"""Tests for core types."""

from datetime import datetime, timezone
from core.types import (
    Candle,
    Side,
    Signal,
    SignalAction,
    SignalMessage,
    SignalQuality,
    StrategyName,
    PairInfo,
    ConversationState,
    UserSession,
)


class TestCandle:
    def test_candle_is_immutable(self):
        c = Candle(
            timestamp=datetime(2025, 1, 1),
            open=100.0, high=105.0, low=95.0, close=102.0,
            volume=1000.0, symbol="BTCUSDT", interval="1h",
        )
        try:
            c.close = 200.0  # type: ignore
            assert False, "Should not allow mutation"
        except AttributeError:
            pass


class TestSignal:
    def test_signal_creation(self):
        s = Signal(
            timestamp=datetime(2025, 1, 1),
            symbol="BTCUSDT",
            action=SignalAction.ENTER_LONG,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            confidence=0.75,
        )
        assert s.action == SignalAction.ENTER_LONG
        assert s.confidence == 0.75


class TestSignalMessage:
    def test_signal_message_creation(self):
        s = Signal(
            timestamp=datetime(2025, 1, 1),
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
            explanation=["EMA 골든크로스", "MACD 양전환"],
            indicators={"adx": 28.5, "rsi": 55.2},
            risk_reward_ratio=2.0,
        )
        assert msg.quality == SignalQuality.STRONG
        assert len(msg.explanation) == 2
        assert msg.risk_reward_ratio == 2.0


class TestConversationState:
    def test_states_exist(self):
        assert ConversationState.IDLE.value == "idle"
        assert ConversationState.SIGNAL_SENT.value == "signal_sent"
        assert ConversationState.MONITORING.value == "monitoring"
        assert ConversationState.EXIT_SIGNAL_SENT.value == "exit_signal_sent"

    def test_no_user_entered_state(self):
        """USER_ENTERED는 제거됨 (SIGNAL_SENT → 바로 MONITORING)."""
        values = [s.value for s in ConversationState]
        assert "user_entered" not in values


class TestUserSession:
    def test_default_session(self):
        session = UserSession(chat_id="123", state=ConversationState.IDLE)
        assert session.active_signal is None
        assert session.user_entry_price is None


class TestPairInfo:
    def test_pair_info(self):
        pi = PairInfo(
            symbol="BTCUSDT",
            volume_24h=500_000_000.0,
            atr_percent=0.03,
            correlation_to_btc=1.0,
            score=15_000_000.0,
        )
        assert pi.symbol == "BTCUSDT"

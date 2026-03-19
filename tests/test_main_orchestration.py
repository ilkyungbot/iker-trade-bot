"""Tests for SignalBot orchestration logic."""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from core.config import AppConfig, BybitConfig, DatabaseConfig, TelegramConfig, SignalConfig
from core.types import (
    SignalAction, StrategyName, Signal, SignalMessage, SignalQuality,
    ConversationState,
)


def _make_config() -> AppConfig:
    return AppConfig(
        bybit=BybitConfig(api_key="", api_secret="", testnet=True),
        database=DatabaseConfig(url="sqlite:///:memory:"),
        telegram=TelegramConfig(bot_token="", chat_id="123"),
        signal=SignalConfig(
            signal_cooldown_minutes=30,
            monitoring_interval_minutes=15,
            signal_expiry_minutes=60,
            min_signal_quality="moderate",
            primary_interval="240",
            candle_intervals=("15", "60", "240"),
            max_pairs=5,
            pair_rebalance_days=14,
        ),
    )


def _make_bot(tmp_path=None):
    """Create a SignalBot with all IO mocked out."""
    from main import SignalBot

    with patch("main.BybitCollector"), \
         patch("main.Reporter"), \
         patch("main.TelegramCommandHandler"):
        bot = SignalBot(_make_config())

    bot.reporter = AsyncMock()
    bot.pair_selector = MagicMock()
    bot.pair_selector._current_pairs = []

    # 테스트 간 상태 격리를 위해 임시 DB 사용
    if tmp_path:
        from conversation.state_machine import ConversationStateMachine
        from conversation.signal_tracker import SignalTracker
        bot.state_machine = ConversationStateMachine(db_path=str(tmp_path / "test_sm.db"))
        bot.signal_tracker = SignalTracker(db_path=str(tmp_path / "test_tracker.db"))
    else:
        bot.state_machine.force_idle("123")

    return bot


def _make_signal_msg(symbol="BTCUSDT"):
    s = Signal(
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
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
        explanation=["EMA 골든크로스"],
        indicators={"adx": 28.5},
        risk_reward_ratio=2.0,
    )


class TestSignalScore:
    def test_strong_beats_moderate(self):
        from main import SignalBot
        strong = _make_signal_msg()
        s = Signal(
            timestamp=datetime.now(timezone.utc),
            symbol="ETHUSDT",
            action=SignalAction.ENTER_SHORT,
            strategy=StrategyName.FUNDING_RATE,
            entry_price=3000, stop_loss=3100, take_profit=2900,
            confidence=0.4,
        )
        moderate = SignalMessage(
            signal=s,
            quality=SignalQuality.MODERATE,
            explanation=["test"],
            indicators={},
            risk_reward_ratio=1.5,
        )
        assert SignalBot._signal_score(strong) > SignalBot._signal_score(moderate)



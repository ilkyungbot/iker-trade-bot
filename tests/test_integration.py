"""
Integration test: end-to-end signal generation flow.

Simulates a complete signal cycle without real API calls.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from core.config import AppConfig, BybitConfig, DatabaseConfig, TelegramConfig, SignalConfig
from core.types import (
    Candle, Signal, SignalAction, StrategyName,
    SignalMessage, SignalQuality,
)
from data.features import add_all_features, candles_to_dataframe
from data.validator import DataValidator
from strategy.trend_following import TrendFollowingStrategy
from strategy.funding_rate import FundingRateStrategy
from conversation.state_machine import ConversationStateMachine
from conversation.signal_tracker import SignalTracker


def _make_candles(
    n: int = 100, symbol: str = "BTCUSDT", interval: str = "240",
    start_price: float = 50000.0, trend: float = 20.0,
) -> list[Candle]:
    np.random.seed(42)
    candles = []
    price = start_price
    for i in range(n):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=4*i)
        noise = np.random.normal(0, 100)
        price += trend + noise
        open_price = price - noise / 2
        close_price = price
        high = max(open_price, close_price) + abs(np.random.normal(0, 150)) + 1
        low = min(open_price, close_price) - abs(np.random.normal(0, 150)) - 1
        candles.append(Candle(
            timestamp=ts, open=open_price, high=high, low=low,
            close=close_price, volume=np.random.uniform(100, 1000),
            symbol=symbol, interval=interval,
        ))
    return candles


class TestEndToEnd:
    def test_candles_to_features_to_signal(self):
        """캔들 → 지표 → 전략 시그널 파이프라인이 크래시 없이 동작."""
        candles = _make_candles(n=100, trend=20)
        df = candles_to_dataframe(candles)
        df = add_all_features(df)

        strategy = TrendFollowingStrategy()
        result = strategy.generate_signal(df, "BTCUSDT")

        # 시그널이 나올 수도 안 나올 수도 있지만 크래시는 없어야 함
        if result is not None:
            assert result.signal.symbol == "BTCUSDT"
            assert result.signal.entry_price > 0
            assert result.quality in (SignalQuality.STRONG, SignalQuality.MODERATE)

    def test_state_machine_full_flow(self, tmp_path):
        """IDLE → SIGNAL_SENT → MONITORING → EXIT_SIGNAL_SENT → IDLE."""
        sm = ConversationStateMachine(db_path=str(tmp_path / "test.db"))

        s = Signal(
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT", action=SignalAction.ENTER_LONG,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=67450, stop_loss=66800, take_profit=68750,
            confidence=0.6,
        )
        msg = SignalMessage(
            signal=s, quality=SignalQuality.STRONG,
            explanation=["test"], indicators={}, risk_reward_ratio=2.0,
        )

        # IDLE → SIGNAL_SENT
        assert sm.send_signal("123", msg)
        assert sm.get_session("123").state.value == "signal_sent"

        # SIGNAL_SENT → MONITORING
        assert sm.user_entered("123", entry_price=67500)
        assert sm.get_session("123").state.value == "monitoring"

        # MONITORING → EXIT_SIGNAL_SENT
        assert sm.send_exit_signal("123")
        assert sm.get_session("123").state.value == "exit_signal_sent"

        # EXIT_SIGNAL_SENT → IDLE
        assert sm.user_exited("123")
        assert sm.get_session("123").state.value == "idle"

    def test_signal_tracker_accuracy(self, tmp_path):
        """시그널 기록 → 결과 업데이트 → 리포트."""
        tracker = SignalTracker(db_path=str(tmp_path / "tracker.db"))

        sig_id = tracker.record_signal(
            symbol="BTCUSDT", direction="long", strategy="trend_following",
            quality="strong", entry_price=67450, stop_loss=66800,
            take_profit=68750,
            signal_time=datetime.now(timezone.utc) - timedelta(hours=25),
        )

        tracker.update_outcome(sig_id, price_4h=67800, price_24h=69000)

        report = tracker.weekly_report()
        assert report["total"] == 1
        assert report["tp_rate"] == 1.0

    def test_validator_integration(self):
        candles = _make_candles(n=50)
        validator = DataValidator()
        now = candles[-1].timestamp + timedelta(minutes=20)
        result = validator.validate_candles(candles, "240", now)
        assert result.is_valid

    def test_funding_rate_strategy(self):
        """펀딩레이트 전략 파이프라인."""
        candles = _make_candles(n=50, trend=-10)
        df = candles_to_dataframe(candles)
        df = add_all_features(df)

        strategy = FundingRateStrategy()
        result = strategy.generate_signal(
            df, "BTCUSDT", latest_funding_rate=0.002,
        )
        # RSI가 65+가 아니면 시그널이 안 나올 수 있음
        # 크래시만 없으면 OK

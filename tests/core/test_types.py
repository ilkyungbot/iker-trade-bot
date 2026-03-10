"""Tests for core types."""

from datetime import datetime
from core.types import (
    Candle,
    Side,
    Signal,
    SignalAction,
    StrategyName,
    Position,
    Trade,
    PortfolioState,
    PairInfo,
    CircuitBreakerState,
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


class TestPosition:
    def test_notional_value(self):
        p = Position(
            symbol="BTCUSDT", side=Side.LONG, entry_price=50000.0,
            quantity=0.1, leverage=5.0, stop_loss=49000.0,
            trailing_stop=49500.0, strategy=StrategyName.TREND_FOLLOWING,
            entry_time=datetime(2025, 1, 1),
        )
        assert p.notional_value == 5000.0

    def test_margin_used(self):
        p = Position(
            symbol="BTCUSDT", side=Side.LONG, entry_price=50000.0,
            quantity=0.1, leverage=5.0, stop_loss=49000.0,
            trailing_stop=49500.0, strategy=StrategyName.TREND_FOLLOWING,
            entry_time=datetime(2025, 1, 1),
        )
        assert p.margin_used == 1000.0  # 5000 / 5


class TestPortfolioState:
    def test_empty_portfolio(self):
        ps = PortfolioState(total_capital=1000000.0, available_capital=1000000.0)
        assert ps.total_margin_used == 0.0
        assert ps.position_count == 0
        assert ps.circuit_breaker_state == CircuitBreakerState.NORMAL

    def test_portfolio_with_positions(self):
        pos = Position(
            symbol="BTCUSDT", side=Side.LONG, entry_price=50000.0,
            quantity=0.1, leverage=5.0, stop_loss=49000.0,
            trailing_stop=49500.0, strategy=StrategyName.TREND_FOLLOWING,
            entry_time=datetime(2025, 1, 1),
        )
        ps = PortfolioState(
            total_capital=1000000.0,
            available_capital=999000.0,
            positions=[pos],
        )
        assert ps.total_margin_used == 1000.0
        assert ps.position_count == 1


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

    def test_signal_with_metadata(self):
        s = Signal(
            timestamp=datetime(2025, 1, 1),
            symbol="BTCUSDT",
            action=SignalAction.ENTER_LONG,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            confidence=0.75,
            metadata={"adx": 30.5, "ema_slope": "up"},
        )
        assert s.metadata["adx"] == 30.5


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

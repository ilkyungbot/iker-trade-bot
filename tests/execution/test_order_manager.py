"""Tests for order manager."""

from datetime import datetime, timezone
from execution.order_manager import OrderManager
from core.types import (
    Signal, SignalAction, StrategyName, OrderStatus, Side, TradingMode,
)


def _make_signal(action=SignalAction.ENTER_LONG, price=50000.0) -> Signal:
    return Signal(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        symbol="BTCUSDT",
        action=action,
        strategy=StrategyName.TREND_FOLLOWING,
        entry_price=price,
        stop_loss=price - 1000 if action == SignalAction.ENTER_LONG else price + 1000,
        take_profit=price + 2000 if action == SignalAction.ENTER_LONG else price - 2000,
        confidence=0.7,
    )


class TestPaperTrading:
    def test_place_long_order(self):
        om = OrderManager(mode=TradingMode.PAPER)
        signal = _make_signal(SignalAction.ENTER_LONG)
        order = om.place_entry_order(signal, quantity=0.1, leverage=5.0, pair_volume_24h=500e6)

        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.side == Side.LONG

    def test_place_short_order(self):
        om = OrderManager(mode=TradingMode.PAPER)
        signal = _make_signal(SignalAction.ENTER_SHORT)
        order = om.place_entry_order(signal, quantity=0.1, leverage=5.0, pair_volume_24h=500e6)

        assert order is not None
        assert order.side == Side.SHORT

    def test_paper_position_tracked(self):
        om = OrderManager(mode=TradingMode.PAPER)
        signal = _make_signal()
        om.place_entry_order(signal, quantity=0.1, leverage=5.0, pair_volume_24h=500e6)

        positions = om.get_paper_positions()
        assert "BTCUSDT" in positions
        assert positions["BTCUSDT"].quantity == 0.1

    def test_position_size_clamped_to_volume(self):
        om = OrderManager(mode=TradingMode.PAPER)
        signal = _make_signal(price=100.0)
        # volume = $10M, max position = 2% = $200K, qty at $100 = 2000
        order = om.place_entry_order(signal, quantity=5000.0, leverage=1.0, pair_volume_24h=10e6)

        assert order is not None
        assert order.quantity <= 2000.0

    def test_zero_quantity_rejected(self):
        om = OrderManager(mode=TradingMode.PAPER)
        signal = _make_signal()
        order = om.place_entry_order(signal, quantity=0.0, leverage=5.0, pair_volume_24h=500e6)

        assert order is None

    def test_set_stop_loss_paper(self):
        om = OrderManager(mode=TradingMode.PAPER)
        result = om.set_stop_loss("BTCUSDT", 49000.0)
        assert result is True

    def test_close_position_paper(self):
        om = OrderManager(mode=TradingMode.PAPER)
        signal = _make_signal()
        om.place_entry_order(signal, quantity=0.1, leverage=5.0, pair_volume_24h=500e6)

        positions = om.get_paper_positions()
        assert "BTCUSDT" in positions

        result = om.close_position(positions["BTCUSDT"])
        assert result is True
        assert "BTCUSDT" not in om.get_paper_positions()

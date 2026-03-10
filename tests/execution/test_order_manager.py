"""Tests for order manager."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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


def _make_mock_client() -> MagicMock:
    """Create a MagicMock that satisfies ExchangeOrderClient Protocol."""
    client = MagicMock()
    client.place_order.return_value = {
        "result": {"orderId": "test-order-123"},
    }
    client.set_leverage.return_value = {"retCode": 0}
    client.get_open_orders.return_value = {"result": {"list": []}}
    client.cancel_order.return_value = {"retCode": 0}
    client.set_trading_stop.return_value = {"retCode": 0}
    client.get_positions.return_value = {"result": {"list": []}}
    return client


class TestLiveMode:
    def test_set_leverage_called_before_order_in_live_mode(self):
        client = _make_mock_client()
        om = OrderManager(client=client, mode=TradingMode.LIVE)
        signal = _make_signal(price=50000.0)

        om.place_entry_order(signal, quantity=0.1, leverage=5.0, pair_volume_24h=500e6)

        client.set_leverage.assert_called_once_with(
            category="linear",
            symbol="BTCUSDT",
            buyLeverage="5.0",
            sellLeverage="5.0",
        )
        # set_leverage must be called before place_order
        call_names = [c[0] for c in client.mock_calls]
        lev_idx = call_names.index("set_leverage")
        order_idx = call_names.index("place_order")
        assert lev_idx < order_idx, "set_leverage must be called before place_order"

    def test_poll_order_filled(self):
        """get_open_orders returns empty list → order treated as filled."""
        client = _make_mock_client()
        # Empty list means order is no longer open → filled
        client.get_open_orders.return_value = {"result": {"list": []}}
        om = OrderManager(client=client, mode=TradingMode.LIVE)
        signal = _make_signal()

        order = om.place_entry_order(signal, quantity=0.1, leverage=5.0, pair_volume_24h=500e6)

        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None

    def test_poll_order_cancelled(self):
        """get_open_orders returns cancelled status → order rejected."""
        client = _make_mock_client()
        client.get_open_orders.return_value = {
            "result": {
                "list": [{"orderStatus": "Cancelled"}],
            }
        }
        om = OrderManager(client=client, mode=TradingMode.LIVE)
        signal = _make_signal()

        order = om.place_entry_order(signal, quantity=0.1, leverage=5.0, pair_volume_24h=500e6)

        assert order is not None
        assert order.status == OrderStatus.REJECTED

    @patch("execution.order_manager.time.sleep", return_value=None)
    @patch("execution.order_manager.time.monotonic")
    def test_poll_order_timeout_cancels(self, mock_monotonic, mock_sleep):
        """Order still pending after timeout → cancel_order is called."""
        client = _make_mock_client()
        # Always return a pending order so it never fills
        client.get_open_orders.return_value = {
            "result": {
                "list": [{"orderStatus": "New"}],
            }
        }

        # _poll_order_status default: max_wait_seconds=30, poll_interval=2.0
        # monotonic() is called: once for deadline (0.0 → deadline=30.0),
        # then once per loop iteration check.  We let 2 iterations run then
        # exceed the deadline.
        mock_monotonic.side_effect = [0.0, 10.0, 20.0, 31.0]

        om = OrderManager(client=client, mode=TradingMode.LIVE)
        signal = _make_signal()

        order = om.place_entry_order(
            signal, quantity=0.1, leverage=5.0, pair_volume_24h=500e6,
        )

        assert order is not None
        assert order.status == OrderStatus.REJECTED
        client.cancel_order.assert_called_once_with(
            category="linear",
            symbol="BTCUSDT",
            orderId="test-order-123",
        )

    def test_set_leverage_error_ignored(self):
        """set_leverage raising exception should not prevent order placement."""
        client = _make_mock_client()
        client.set_leverage.side_effect = Exception("leverage already set")
        # Order fills immediately (empty open-orders list)
        client.get_open_orders.return_value = {"result": {"list": []}}

        om = OrderManager(client=client, mode=TradingMode.LIVE)
        signal = _make_signal()

        order = om.place_entry_order(signal, quantity=0.1, leverage=5.0, pair_volume_24h=500e6)

        assert order is not None
        assert order.status == OrderStatus.FILLED
        # place_order should still have been called despite set_leverage failure
        client.place_order.assert_called_once()

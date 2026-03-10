"""
Layer 5: Order management.

Places orders, sets stop-losses, manages order lifecycle.
All exchange interactions go through this module.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Protocol

from core.types import (
    Order, OrderType, OrderStatus, Side, Signal, SignalAction,
    Position, StrategyName, TradingMode,
)
from core.safety import REQUIRE_EXCHANGE_STOP_LOSS, validate_leverage, validate_position_size

logger = logging.getLogger(__name__)


class ExchangeOrderClient(Protocol):
    """Protocol for exchange order operations."""

    def place_order(
        self, category: str, symbol: str, side: str, orderType: str,
        qty: str, price: str | None = None, **kwargs,
    ) -> dict: ...

    def set_trading_stop(
        self, category: str, symbol: str, stopLoss: str, **kwargs,
    ) -> dict: ...

    def cancel_order(self, category: str, symbol: str, orderId: str) -> dict: ...

    def get_positions(self, category: str, symbol: str) -> dict: ...

    def set_leverage(
        self, category: str, symbol: str, buyLeverage: str, sellLeverage: str,
    ) -> dict: ...

    def get_open_orders(self, category: str, symbol: str, orderId: str) -> dict: ...


class OrderManager:
    """Manages order placement and position lifecycle."""

    def __init__(
        self,
        client: ExchangeOrderClient | None = None,
        mode: TradingMode = TradingMode.PAPER,
    ):
        self.client = client
        self.mode = mode
        self._paper_positions: dict[str, Position] = {}
        self._paper_orders: list[Order] = []

    def place_entry_order(
        self,
        signal: Signal,
        quantity: float,
        leverage: float,
        pair_volume_24h: float,
    ) -> Order | None:
        """
        Place an entry order based on signal.

        Returns Order object or None if validation fails.
        """
        leverage = validate_leverage(leverage)

        # Validate position size against volume
        position_value = signal.entry_price * quantity
        clamped_value = validate_position_size(position_value, pair_volume_24h)
        if clamped_value < position_value:
            quantity = clamped_value / signal.entry_price
            logger.info(f"Position size clamped to volume limit: {quantity}")

        if quantity <= 0:
            return None

        side = Side.LONG if signal.action == SignalAction.ENTER_LONG else Side.SHORT
        bybit_side = "Buy" if side == Side.LONG else "Sell"

        order = Order(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.LIMIT,
            price=signal.entry_price,
            quantity=quantity,
            leverage=leverage,
        )

        if self.mode == TradingMode.PAPER:
            return self._paper_fill_order(order, signal)

        if self.client is None:
            logger.error("No exchange client configured for live trading")
            return None

        try:
            # Set leverage before placing the order
            try:
                self.client.set_leverage(
                    category="linear",
                    symbol=signal.symbol,
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage),
                )
                logger.info(f"Set leverage {leverage}x for {signal.symbol}")
            except Exception as e:
                # Bybit returns an error if leverage is already set to the same value;
                # that is safe to ignore.
                logger.debug(f"set_leverage note for {signal.symbol}: {e}")

            # Place limit order
            response = self.client.place_order(
                category="linear",
                symbol=signal.symbol,
                side=bybit_side,
                orderType="Limit",
                qty=str(quantity),
                price=str(signal.entry_price),
                timeInForce="GTC",
            )

            result = response.get("result", {})
            order.exchange_order_id = result.get("orderId")
            order.status = OrderStatus.PENDING

            logger.info(
                f"Placed {side.value} order for {signal.symbol}: "
                f"qty={quantity}, price={signal.entry_price}, "
                f"order_id={order.exchange_order_id}"
            )

            # Poll order status until filled, cancelled, or timeout
            if order.exchange_order_id:
                order = self._poll_order_status(order)

            return order

        except Exception as e:
            logger.error(f"Failed to place order for {signal.symbol}: {e}")
            order.status = OrderStatus.REJECTED
            return order

    def set_stop_loss(self, symbol: str, stop_loss_price: float) -> bool:
        """Set exchange-side stop-loss. Returns True if confirmed."""
        if self.mode == TradingMode.PAPER:
            logger.info(f"[PAPER] Stop-loss set for {symbol} at {stop_loss_price}")
            return True

        if self.client is None:
            return False

        try:
            self.client.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(stop_loss_price),
            )
            logger.info(f"Stop-loss set for {symbol} at {stop_loss_price}")
            return True
        except Exception as e:
            logger.error(f"Failed to set stop-loss for {symbol}: {e}")
            return False

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an unfilled order."""
        if self.mode == TradingMode.PAPER:
            return True

        if self.client is None:
            return False

        try:
            self.client.cancel_order(
                category="linear",
                symbol=symbol,
                orderId=order_id,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def close_position(self, position: Position) -> bool:
        """Close a position with market order."""
        if self.mode == TradingMode.PAPER:
            if position.symbol in self._paper_positions:
                del self._paper_positions[position.symbol]
            return True

        if self.client is None:
            return False

        try:
            close_side = "Sell" if position.side == Side.LONG else "Buy"
            self.client.place_order(
                category="linear",
                symbol=position.symbol,
                side=close_side,
                orderType="Market",
                qty=str(position.quantity),
                reduceOnly=True,
            )
            logger.info(f"Closed {position.side.value} position for {position.symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to close position for {position.symbol}: {e}")
            return False

    def _poll_order_status(
        self,
        order: Order,
        max_wait_seconds: int = 30,
        poll_interval: float = 2.0,
    ) -> Order:
        """
        Poll the exchange for order status until it is filled, cancelled, or
        the timeout elapses.  Returns the updated Order.
        """
        if self.client is None or order.exchange_order_id is None:
            return order

        deadline = time.monotonic() + max_wait_seconds
        while time.monotonic() < deadline:
            try:
                resp = self.client.get_open_orders(
                    category="linear",
                    symbol=order.symbol,
                    orderId=order.exchange_order_id,
                )
                orders_list = resp.get("result", {}).get("list", [])

                if not orders_list:
                    # Order no longer in open-orders list → treat as filled
                    order.status = OrderStatus.FILLED
                    order.filled_at = datetime.now(timezone.utc)
                    logger.info(
                        f"Order {order.exchange_order_id} for {order.symbol} filled"
                    )
                    return order

                exchange_status = orders_list[0].get("orderStatus", "").lower()
                if exchange_status == "filled":
                    order.status = OrderStatus.FILLED
                    order.filled_at = datetime.now(timezone.utc)
                    logger.info(
                        f"Order {order.exchange_order_id} for {order.symbol} filled"
                    )
                    return order
                elif exchange_status in ("cancelled", "rejected", "deactivated"):
                    order.status = OrderStatus.REJECTED
                    logger.warning(
                        f"Order {order.exchange_order_id} for {order.symbol} "
                        f"status: {exchange_status}"
                    )
                    return order

            except Exception as e:
                logger.warning(f"Error polling order status: {e}")

            time.sleep(poll_interval)

        # Timed out — order still open
        logger.warning(
            f"Order {order.exchange_order_id} for {order.symbol} still pending "
            f"after {max_wait_seconds}s, cancelling"
        )
        self.cancel_order(order.symbol, order.exchange_order_id)
        order.status = OrderStatus.REJECTED
        return order

    def _paper_fill_order(self, order: Order, signal: Signal) -> Order:
        """Simulate order fill in paper trading mode."""
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now(timezone.utc)
        order.order_id = f"paper_{signal.symbol}_{int(datetime.now(timezone.utc).timestamp())}"

        position = Position(
            symbol=signal.symbol,
            side=order.side,
            entry_price=order.price,
            quantity=order.quantity,
            leverage=order.leverage,
            stop_loss=signal.stop_loss,
            trailing_stop=signal.take_profit,
            strategy=signal.strategy,
            entry_time=datetime.now(timezone.utc),
        )
        self._paper_positions[signal.symbol] = position

        logger.info(
            f"[PAPER] Filled {order.side.value} for {signal.symbol}: "
            f"qty={order.quantity}, price={order.price}"
        )

        return order

    def get_paper_positions(self) -> dict[str, Position]:
        """Get current paper trading positions."""
        return self._paper_positions.copy()

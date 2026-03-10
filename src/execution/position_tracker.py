"""
Layer 5: Position tracking.

Tracks open positions, calculates PnL, manages trailing stops.
"""

import logging
from datetime import datetime, timezone

from core.types import Position, Side, Trade, StrategyName, PortfolioState
from core.safety import MAX_CONCURRENT_POSITIONS, MAX_MDD

logger = logging.getLogger(__name__)


class PositionTracker:
    """Tracks portfolio state and manages position lifecycle."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.state = PortfolioState(
            total_capital=initial_capital,
            available_capital=initial_capital,
            peak_capital=initial_capital,
        )

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return self.state.position_count < MAX_CONCURRENT_POSITIONS

    def add_position(self, position: Position) -> bool:
        """Add a new position to tracking."""
        if not self.can_open_position():
            logger.warning("Max concurrent positions reached")
            return False

        self.state.positions.append(position)
        self.state.available_capital -= position.margin_used
        logger.info(f"Tracking new {position.side.value} position for {position.symbol}")
        return True

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        fees: float = 0.0,
        slippage: float = 0.0,
    ) -> Trade | None:
        """Close a position and record the trade."""
        position = self._find_position(symbol)
        if position is None:
            logger.warning(f"No position found for {symbol}")
            return None

        # Calculate PnL
        if position.side == Side.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity

        pnl -= fees
        pnl_percent = (pnl / position.margin_used) * 100 if position.margin_used > 0 else 0

        # Determine exit type
        stop_loss_hit = False
        trailing_stop_hit = False
        if position.side == Side.LONG:
            stop_loss_hit = exit_price <= position.stop_loss
        else:
            stop_loss_hit = exit_price >= position.stop_loss

        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            strategy=position.strategy,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            leverage=position.leverage,
            entry_time=position.entry_time,
            exit_time=datetime.now(timezone.utc),
            pnl=pnl,
            pnl_percent=pnl_percent,
            fees=fees,
            slippage=slippage,
            stop_loss_hit=stop_loss_hit,
            trailing_stop_hit=trailing_stop_hit,
        )

        # Update portfolio state
        self.state.positions = [p for p in self.state.positions if p.symbol != symbol]
        self.state.available_capital += position.margin_used + pnl
        self.state.total_capital += pnl
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl

        # Update peak and drawdown
        if self.state.total_capital > self.state.peak_capital:
            self.state.peak_capital = self.state.total_capital
        if self.state.peak_capital > 0:
            self.state.current_mdd = (
                (self.state.peak_capital - self.state.total_capital)
                / self.state.peak_capital
            )

        # Update consecutive win/loss tracking
        if pnl > 0:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

        logger.info(
            f"Closed {position.side.value} {symbol}: "
            f"PnL={pnl:.2f} USDT ({pnl_percent:.1f}%)"
        )

        return trade

    def update_unrealized_pnl(self, symbol: str, current_price: float) -> None:
        """Update unrealized PnL for an open position."""
        position = self._find_position(symbol)
        if position is None:
            return

        if position.side == Side.LONG:
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

        # Track highest PnL for trailing stop
        if position.unrealized_pnl > position.highest_pnl:
            position.highest_pnl = position.unrealized_pnl

    def update_trailing_stop(self, symbol: str, atr: float, multiplier: float = 2.0) -> None:
        """Update trailing stop based on current profit."""
        position = self._find_position(symbol)
        if position is None:
            return

        if position.unrealized_pnl <= 0:
            return  # only trail when in profit

        trail_distance = atr * multiplier

        if position.side == Side.LONG:
            current_price_approx = position.entry_price + (
                position.unrealized_pnl / position.quantity
            )
            new_stop = current_price_approx - trail_distance
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                position.trailing_stop = new_stop
        else:
            current_price_approx = position.entry_price - (
                position.unrealized_pnl / position.quantity
            )
            new_stop = current_price_approx + trail_distance
            if new_stop < position.stop_loss:
                position.stop_loss = new_stop
                position.trailing_stop = new_stop

    def check_stop_losses(self, prices: dict[str, float]) -> list[str]:
        """Check if any position's stop-loss has been hit. Returns symbols to close."""
        to_close = []
        for position in self.state.positions:
            if position.symbol not in prices:
                continue
            price = prices[position.symbol]
            if position.side == Side.LONG and price <= position.stop_loss:
                to_close.append(position.symbol)
            elif position.side == Side.SHORT and price >= position.stop_loss:
                to_close.append(position.symbol)
        return to_close

    def reset_daily_pnl(self) -> None:
        self.state.daily_pnl = 0.0

    def reset_weekly_pnl(self) -> None:
        self.state.weekly_pnl = 0.0

    def _find_position(self, symbol: str) -> Position | None:
        for p in self.state.positions:
            if p.symbol == symbol:
                return p
        return None

"""Extra tests for position tracker — gap coverage."""

from datetime import datetime, timezone

from execution.position_tracker import PositionTracker
from core.types import Position, Side, StrategyName


def _make_position(
    symbol: str = "BTCUSDT",
    side: Side = Side.LONG,
    entry_price: float = 50_000.0,
    quantity: float = 0.1,
    leverage: float = 5.0,
    stop_loss: float = 49_000.0,
) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        leverage=leverage,
        stop_loss=stop_loss,
        trailing_stop=0.0,
        strategy=StrategyName.TREND_FOLLOWING,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


class TestDuplicateSymbolRejection:
    def test_add_position_rejects_duplicate_symbol(self):
        tracker = PositionTracker(initial_capital=100_000.0)
        pos1 = _make_position(symbol="BTCUSDT")
        pos2 = _make_position(symbol="BTCUSDT", entry_price=51_000.0)

        assert tracker.add_position(pos1) is True
        assert tracker.add_position(pos2) is False
        assert tracker.state.position_count == 1

    def test_different_symbols_accepted(self):
        tracker = PositionTracker(initial_capital=100_000.0)
        pos1 = _make_position(symbol="BTCUSDT")
        pos2 = _make_position(symbol="ETHUSDT", entry_price=3_000.0, stop_loss=2_900.0)

        assert tracker.add_position(pos1) is True
        assert tracker.add_position(pos2) is True
        assert tracker.state.position_count == 2


class TestPnlZeroNotLoss:
    def test_zero_pnl_does_not_increment_consecutive_losses(self):
        tracker = PositionTracker(initial_capital=100_000.0)
        pos = _make_position(symbol="BTCUSDT", entry_price=50_000.0, stop_loss=49_000.0)
        tracker.add_position(pos)

        # Close at exact entry price, zero fees => PnL = 0
        trade = tracker.close_position("BTCUSDT", exit_price=50_000.0, fees=0.0)

        assert trade is not None
        assert trade.pnl == 0.0
        assert tracker.state.consecutive_losses == 0
        assert tracker.state.consecutive_wins == 0


class TestCheckStopLosses:
    def test_long_stop_loss_hit(self):
        tracker = PositionTracker(initial_capital=100_000.0)
        pos = _make_position(side=Side.LONG, stop_loss=49_000.0)
        tracker.add_position(pos)

        # Price at stop loss
        to_close = tracker.check_stop_losses({"BTCUSDT": 49_000.0})
        assert "BTCUSDT" in to_close

    def test_long_stop_loss_not_hit(self):
        tracker = PositionTracker(initial_capital=100_000.0)
        pos = _make_position(side=Side.LONG, stop_loss=49_000.0)
        tracker.add_position(pos)

        to_close = tracker.check_stop_losses({"BTCUSDT": 49_500.0})
        assert "BTCUSDT" not in to_close

    def test_short_stop_loss_hit(self):
        tracker = PositionTracker(initial_capital=100_000.0)
        pos = _make_position(
            side=Side.SHORT,
            entry_price=50_000.0,
            stop_loss=51_000.0,
        )
        tracker.add_position(pos)

        to_close = tracker.check_stop_losses({"BTCUSDT": 51_000.0})
        assert "BTCUSDT" in to_close

    def test_short_stop_loss_not_hit(self):
        tracker = PositionTracker(initial_capital=100_000.0)
        pos = _make_position(
            side=Side.SHORT,
            entry_price=50_000.0,
            stop_loss=51_000.0,
        )
        tracker.add_position(pos)

        to_close = tracker.check_stop_losses({"BTCUSDT": 50_500.0})
        assert "BTCUSDT" not in to_close

    def test_missing_price_ignored(self):
        tracker = PositionTracker(initial_capital=100_000.0)
        pos = _make_position(symbol="BTCUSDT")
        tracker.add_position(pos)

        to_close = tracker.check_stop_losses({"ETHUSDT": 3_000.0})
        assert len(to_close) == 0

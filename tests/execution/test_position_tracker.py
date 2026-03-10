"""Tests for position tracker."""

from datetime import datetime, timezone
from execution.position_tracker import PositionTracker
from core.types import Position, Side, StrategyName


def _make_position(
    symbol: str = "BTCUSDT",
    side: Side = Side.LONG,
    entry_price: float = 50000.0,
    quantity: float = 0.1,
    leverage: float = 5.0,
    stop_loss: float = 49000.0,
) -> Position:
    return Position(
        symbol=symbol, side=side, entry_price=entry_price,
        quantity=quantity, leverage=leverage, stop_loss=stop_loss,
        trailing_stop=stop_loss, strategy=StrategyName.TREND_FOLLOWING,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


class TestPositionTracking:
    def test_add_position(self):
        tracker = PositionTracker(initial_capital=1000000.0)
        pos = _make_position()
        assert tracker.add_position(pos) is True
        assert tracker.state.position_count == 1

    def test_max_positions_enforced(self):
        tracker = PositionTracker(initial_capital=1000000.0)
        for i in range(5):
            pos = _make_position(symbol=f"PAIR{i}")
            tracker.add_position(pos)
        # 6th should fail
        pos6 = _make_position(symbol="PAIR5")
        assert tracker.add_position(pos6) is False

    def test_close_long_profit(self):
        tracker = PositionTracker(initial_capital=1000000.0)
        pos = _make_position(entry_price=50000.0, quantity=0.1)
        tracker.add_position(pos)

        trade = tracker.close_position("BTCUSDT", exit_price=51000.0, fees=5.0)
        assert trade is not None
        assert trade.pnl == 95.0  # (51000-50000)*0.1 - 5
        assert trade.pnl_percent > 0
        assert tracker.state.consecutive_wins == 1

    def test_close_long_loss(self):
        tracker = PositionTracker(initial_capital=1000000.0)
        pos = _make_position(entry_price=50000.0, quantity=0.1)
        tracker.add_position(pos)

        trade = tracker.close_position("BTCUSDT", exit_price=49000.0)
        assert trade is not None
        assert trade.pnl == -100.0  # (49000-50000)*0.1
        assert tracker.state.consecutive_losses == 1

    def test_close_short_profit(self):
        tracker = PositionTracker(initial_capital=1000000.0)
        pos = _make_position(side=Side.SHORT, entry_price=50000.0, quantity=0.1, stop_loss=51000.0)
        tracker.add_position(pos)

        trade = tracker.close_position("BTCUSDT", exit_price=49000.0)
        assert trade is not None
        assert trade.pnl == 100.0  # (50000-49000)*0.1

    def test_capital_updates(self):
        tracker = PositionTracker(initial_capital=10000.0)
        pos = _make_position(entry_price=100.0, quantity=10.0, leverage=5.0)
        # margin = 100*10/5 = 200
        tracker.add_position(pos)
        assert tracker.state.available_capital == 9800.0

        tracker.close_position("BTCUSDT", exit_price=110.0)
        # PnL = (110-100)*10 = 100
        # available = 9800 + 200 + 100 = 10100
        assert tracker.state.available_capital == 10100.0
        assert tracker.state.total_capital == 10100.0


class TestDrawdownTracking:
    def test_drawdown_calculated(self):
        tracker = PositionTracker(initial_capital=10000.0)
        pos = _make_position(entry_price=100.0, quantity=10.0, leverage=5.0)
        tracker.add_position(pos)

        tracker.close_position("BTCUSDT", exit_price=90.0)
        # PnL = -100, capital = 9900, peak = 10000
        assert tracker.state.current_mdd > 0
        assert abs(tracker.state.current_mdd - 0.01) < 0.001  # 1%

    def test_peak_updates_on_profit(self):
        tracker = PositionTracker(initial_capital=10000.0)
        pos = _make_position(entry_price=100.0, quantity=10.0, leverage=5.0)
        tracker.add_position(pos)

        tracker.close_position("BTCUSDT", exit_price=110.0)
        assert tracker.state.peak_capital == 10100.0


class TestStopLossCheck:
    def test_long_stop_hit(self):
        tracker = PositionTracker(initial_capital=10000.0)
        pos = _make_position(stop_loss=49000.0)
        tracker.add_position(pos)

        to_close = tracker.check_stop_losses({"BTCUSDT": 48500.0})
        assert "BTCUSDT" in to_close

    def test_long_stop_not_hit(self):
        tracker = PositionTracker(initial_capital=10000.0)
        pos = _make_position(stop_loss=49000.0)
        tracker.add_position(pos)

        to_close = tracker.check_stop_losses({"BTCUSDT": 50000.0})
        assert to_close == []

    def test_short_stop_hit(self):
        tracker = PositionTracker(initial_capital=10000.0)
        pos = _make_position(side=Side.SHORT, stop_loss=51000.0)
        tracker.add_position(pos)

        to_close = tracker.check_stop_losses({"BTCUSDT": 51500.0})
        assert "BTCUSDT" in to_close


class TestTrailingStop:
    def test_trailing_stop_moves_up_for_long(self):
        tracker = PositionTracker(initial_capital=10000.0)
        pos = _make_position(entry_price=100.0, quantity=10.0, stop_loss=95.0)
        tracker.add_position(pos)

        tracker.update_unrealized_pnl("BTCUSDT", 110.0)
        tracker.update_trailing_stop("BTCUSDT", atr=3.0)

        position = tracker.state.positions[0]
        assert position.stop_loss > 95.0  # moved up

    def test_trailing_stop_doesnt_move_when_losing(self):
        tracker = PositionTracker(initial_capital=10000.0)
        pos = _make_position(entry_price=100.0, quantity=10.0, stop_loss=95.0)
        tracker.add_position(pos)

        tracker.update_unrealized_pnl("BTCUSDT", 90.0)
        tracker.update_trailing_stop("BTCUSDT", atr=3.0)

        position = tracker.state.positions[0]
        assert position.stop_loss == 95.0  # unchanged

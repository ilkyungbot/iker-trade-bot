import pytest
from core.types import Side
from core.calc import calculate_pnl_percent, calculate_pnl_usdt


class TestCalculatePnlPercent:
    def test_long_profit(self):
        assert calculate_pnl_percent(Side.LONG, 100.0, 110.0, 5.0) == pytest.approx(50.0)

    def test_long_loss(self):
        assert calculate_pnl_percent(Side.LONG, 100.0, 95.0, 5.0) == pytest.approx(-25.0)

    def test_short_profit(self):
        assert calculate_pnl_percent(Side.SHORT, 100.0, 90.0, 3.0) == pytest.approx(30.0)

    def test_short_loss(self):
        assert calculate_pnl_percent(Side.SHORT, 100.0, 105.0, 3.0) == pytest.approx(-15.0)

    def test_entry_zero_returns_zero(self):
        assert calculate_pnl_percent(Side.LONG, 0.0, 100.0, 5.0) == 0.0

    def test_negative_entry_returns_zero(self):
        assert calculate_pnl_percent(Side.LONG, -1.0, 100.0, 5.0) == 0.0

    def test_no_movement(self):
        assert calculate_pnl_percent(Side.LONG, 100.0, 100.0, 5.0) == 0.0

    def test_leverage_one(self):
        assert calculate_pnl_percent(Side.LONG, 100.0, 110.0, 1.0) == pytest.approx(10.0)


class TestCalculatePnlUsdt:
    def test_basic(self):
        assert calculate_pnl_usdt(Side.LONG, 100.0, 110.0, 5.0, 500.0) == pytest.approx(250.0)

    def test_loss(self):
        assert calculate_pnl_usdt(Side.LONG, 100.0, 95.0, 5.0, 500.0) == pytest.approx(-125.0)

    def test_zero_margin(self):
        assert calculate_pnl_usdt(Side.LONG, 100.0, 110.0, 5.0, 0.0) == pytest.approx(0.0)

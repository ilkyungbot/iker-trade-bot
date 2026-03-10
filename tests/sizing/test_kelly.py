"""Tests for Kelly Criterion sizing."""

from sizing.kelly import kelly_fraction, calculate_position_size
from core.safety import MAX_RISK_PER_TRADE


class TestKellyFraction:
    def test_positive_edge(self):
        # 50% win rate, 2:1 payoff → should be positive
        f = kelly_fraction(0.5, 0.02, 0.01)
        assert f > 0

    def test_no_edge(self):
        # 50% win rate, 1:1 payoff → Kelly = 0
        f = kelly_fraction(0.5, 0.01, 0.01)
        assert f == 0.0

    def test_negative_edge(self):
        # 30% win rate, 1:1 payoff → negative Kelly
        f = kelly_fraction(0.3, 0.01, 0.01)
        assert f == 0.0

    def test_capped_at_max_risk(self):
        # Very high edge → should cap at MAX_RISK_PER_TRADE
        f = kelly_fraction(0.8, 0.10, 0.01)
        assert f <= MAX_RISK_PER_TRADE

    def test_half_kelly_applied(self):
        # Full Kelly for 50% wr, 2:1 = (0.5*2-0.5)/2 = 0.25
        # Half Kelly = 0.125
        f = kelly_fraction(0.5, 0.02, 0.01)
        assert f < 0.25  # must be less than full Kelly

    def test_zero_win_rate(self):
        assert kelly_fraction(0.0, 0.02, 0.01) == 0.0

    def test_one_win_rate(self):
        assert kelly_fraction(1.0, 0.02, 0.01) == 0.0

    def test_zero_avg_win(self):
        assert kelly_fraction(0.5, 0.0, 0.01) == 0.0

    def test_zero_avg_loss(self):
        assert kelly_fraction(0.5, 0.02, 0.0) == 0.0


class TestCalculatePositionSize:
    def test_basic_calculation(self):
        # $10,000 capital, risk 1%, entry $100, SL $98, 5x leverage
        qty = calculate_position_size(
            capital=10000, risk_fraction=0.01,
            entry_price=100, stop_loss=98, leverage=5,
        )
        # Risk amount = $100, risk per unit = $2, qty = 50
        assert qty == 50.0

    def test_zero_capital(self):
        assert calculate_position_size(0, 0.01, 100, 98, 5) == 0.0

    def test_stop_at_entry(self):
        assert calculate_position_size(10000, 0.01, 100, 100, 5) == 0.0

    def test_short_position(self):
        # Short: entry $100, SL $102
        qty = calculate_position_size(
            capital=10000, risk_fraction=0.01,
            entry_price=100, stop_loss=102, leverage=5,
        )
        assert qty == 50.0  # same math, abs(100-102) = 2

"""Extra tests for Kelly sizing — gap coverage."""

import math

from sizing.kelly import kelly_fraction, calculate_position_size


class TestLeverageApplied:
    def test_5x_leverage_gives_5x_bigger_position(self):
        capital = 10_000.0
        risk_fraction = 0.01
        entry_price = 50_000.0
        stop_loss = 49_000.0  # risk distance = 1_000

        qty_1x = calculate_position_size(
            capital, risk_fraction, entry_price, stop_loss, leverage=1.0,
        )
        qty_5x = calculate_position_size(
            capital, risk_fraction, entry_price, stop_loss, leverage=5.0,
        )

        assert qty_5x > 0
        assert qty_1x > 0
        assert math.isclose(qty_5x, qty_1x * 5.0, rel_tol=1e-9), (
            f"5x leverage should give exactly 5x position: {qty_5x} vs {qty_1x * 5}"
        )

    def test_leverage_10x(self):
        qty_1x = calculate_position_size(10_000, 0.01, 100.0, 95.0, leverage=1.0)
        qty_10x = calculate_position_size(10_000, 0.01, 100.0, 95.0, leverage=10.0)
        assert math.isclose(qty_10x, qty_1x * 10.0, rel_tol=1e-9)


class TestNaNInputs:
    """NaN inputs should not raise exceptions. The functions may return NaN or
    0.0 depending on which guard clause (if any) catches the NaN."""

    def test_kelly_fraction_nan_win_rate(self):
        result = kelly_fraction(float("nan"), 0.02, 0.01)
        # NaN <= 0 and NaN >= 1 are both False, so guard doesn't catch it.
        # The function will propagate NaN through arithmetic — that's acceptable.
        assert result == 0.0 or math.isnan(result)

    def test_kelly_fraction_nan_avg_win(self):
        result = kelly_fraction(0.55, float("nan"), 0.01)
        assert result == 0.0 or math.isnan(result)

    def test_kelly_fraction_nan_avg_loss(self):
        # avg_loss=NaN → division by zero in b = avg_win / avg_loss won't raise
        # (float NaN division), but avg_loss <= 0 is False for NaN.
        result = kelly_fraction(0.55, 0.02, float("nan"))
        assert result == 0.0 or math.isnan(result)

    def test_position_size_nan_capital(self):
        result = calculate_position_size(float("nan"), 0.01, 100.0, 95.0, 5.0)
        assert result == 0.0 or math.isnan(result)

    def test_position_size_nan_entry_price(self):
        result = calculate_position_size(10_000, 0.01, float("nan"), 95.0, 5.0)
        assert result == 0.0 or math.isnan(result)

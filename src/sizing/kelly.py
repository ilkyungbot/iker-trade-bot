"""
Layer 4: Kelly Criterion position sizing.

Calculates optimal position size based on historical win rate and payoff ratio.
Uses half-Kelly for safety (full Kelly is too aggressive in practice).
"""

from core.safety import MAX_RISK_PER_TRADE


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly fraction (half-Kelly for safety).

    Args:
        win_rate: historical win rate (0.0 to 1.0)
        avg_win: average winning trade return (positive value, e.g., 0.02 for 2%)
        avg_loss: average losing trade return (positive value, e.g., 0.01 for 1%)

    Returns:
        Fraction of capital to risk (0.0 to MAX_RISK_PER_TRADE)
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0

    # Kelly formula: f = (p * b - q) / b
    # where p = win_rate, q = 1 - win_rate, b = avg_win / avg_loss
    b = avg_win / avg_loss
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b

    if kelly <= 0:
        return 0.0

    # Half-Kelly for safety
    half_kelly = kelly / 2.0

    # Clamp to safety limit
    return min(half_kelly, MAX_RISK_PER_TRADE)


def calculate_position_size(
    capital: float,
    risk_fraction: float,
    entry_price: float,
    stop_loss: float,
    leverage: float,
) -> float:
    """
    Calculate position size in base currency units.

    Args:
        capital: total available capital in USDT
        risk_fraction: fraction of capital to risk (from kelly_fraction)
        entry_price: planned entry price
        stop_loss: stop-loss price
        leverage: leverage multiplier

    Returns:
        Position size in base currency units (e.g., BTC quantity)
    """
    if capital <= 0 or risk_fraction <= 0 or entry_price <= 0 or leverage <= 0:
        return 0.0

    risk_distance = abs(entry_price - stop_loss)
    if risk_distance <= 0:
        return 0.0

    # Dollar amount we're willing to lose
    risk_amount = capital * risk_fraction

    # Position size = risk_amount / risk_per_unit
    # risk_per_unit = how much we lose per 1 unit of base currency if stop is hit
    risk_per_unit = risk_distance

    # Quantity in base currency (margin-based)
    margin_quantity = risk_amount / risk_per_unit

    # Scale by leverage: notional position = leverage * margin
    quantity = margin_quantity * leverage

    return quantity

"""Centralized PnL calculations. Single source of truth."""

from core.types import Side


def calculate_pnl_percent(
    side: Side, entry_price: float, current_price: float, leverage: float
) -> float:
    if entry_price <= 0:
        return 0.0
    direction = 1.0 if side == Side.LONG else -1.0
    return direction * (current_price - entry_price) / entry_price * leverage * 100


def calculate_pnl_usdt(
    side: Side, entry_price: float, current_price: float,
    leverage: float, margin_usdt: float,
) -> float:
    pct = calculate_pnl_percent(side, entry_price, current_price, leverage)
    return margin_usdt * pct / 100

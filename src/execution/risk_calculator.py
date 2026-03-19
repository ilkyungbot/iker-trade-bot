"""Risk calculation utilities for 3-7x leverage trading.

Provides stop-loss / take-profit suggestions, R:R ratio calculation,
and position validation with safety warnings.
"""

from __future__ import annotations

from src.core.types import Side


# ── Round-number offset ──────────────────────────────────────────────

_ROUND_OFFSET = 0.12  # nudge away from .000 levels


def _apply_round_offset(price: float, side: Side) -> float:
    """Shift price away from round numbers (.000 levels).

    For LONG SL (below entry): subtract offset → lower.
    For SHORT SL (above entry): add offset → higher.
    """
    if side == Side.LONG:
        return price - _ROUND_OFFSET
    return price + _ROUND_OFFSET


# ── Public API ───────────────────────────────────────────────────────


def suggest_stop_loss(
    entry_price: float,
    side: Side,
    atr: float,
    leverage: int,  # noqa: ARG001 — reserved for future dynamic adjustment
) -> float:
    """Suggest a stop-loss price based on ATR.

    LONG : entry - 1.5 * ATR, then round-number offset.
    SHORT: entry + 1.5 * ATR, then round-number offset.
    """
    distance = 1.5 * atr
    if side == Side.LONG:
        base = entry_price - distance
    else:
        base = entry_price + distance
    return _apply_round_offset(base, side)


def suggest_take_profit(
    entry_price: float,
    side: Side,
    stop_loss: float,
) -> float:
    """Suggest a take-profit price at R:R 2:1."""
    risk = abs(entry_price - stop_loss)
    if side == Side.LONG:
        return entry_price + 2 * risk
    return entry_price - 2 * risk


def calculate_rr_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    side: Side,  # noqa: ARG001 — kept for API consistency
) -> float:
    """Return reward / risk ratio."""
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    if risk == 0:
        return 0.0
    return reward / risk


def validate_position(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    leverage: int,
    margin_usdt: float,
    side: Side,
) -> dict:
    """Validate a position and return risk metrics + warnings.

    Returns
    -------
    dict with keys:
        valid, rr_ratio, max_loss_usdt, max_loss_pct,
        leveraged_sl_pct, warnings
    """
    warnings: list[str] = []
    valid = True

    # SL wrong side check
    if side == Side.LONG and stop_loss >= entry_price:
        warnings.append("SL is on wrong side (above entry for LONG)")
        valid = False
    elif side == Side.SHORT and stop_loss <= entry_price:
        warnings.append("SL is on wrong side (below entry for SHORT)")
        valid = False

    rr_ratio = calculate_rr_ratio(entry_price, stop_loss, take_profit, side)

    sl_pct = abs(entry_price - stop_loss) / entry_price * 100  # unleveraged %
    leveraged_sl_pct = sl_pct * leverage

    notional = margin_usdt * leverage
    max_loss_usdt = notional * (abs(entry_price - stop_loss) / entry_price)
    max_loss_pct = (max_loss_usdt / margin_usdt) * 100 if margin_usdt else 0.0

    # Warnings
    if rr_ratio < 1.5:
        warnings.append(f"R:R ratio {rr_ratio:.2f} is below 1.5")
    if leveraged_sl_pct > 20:
        warnings.append(
            f"Leveraged SL loss {leveraged_sl_pct:.1f}% exceeds 20% of margin"
        )
    if sl_pct < 0.5 and valid:
        warnings.append(f"SL too close to entry ({sl_pct:.2f}%)")

    return {
        "valid": valid,
        "rr_ratio": rr_ratio,
        "max_loss_usdt": max_loss_usdt,
        "max_loss_pct": max_loss_pct,
        "leveraged_sl_pct": leveraged_sl_pct,
        "warnings": warnings,
    }


def add_slippage_buffer(
    stop_loss: float,
    side: Side,
    buffer_pct: float = 0.005,
) -> float:
    """Widen SL by a slippage buffer.

    LONG : SL * (1 - buffer_pct)  → lower
    SHORT: SL * (1 + buffer_pct)  → higher
    """
    if side == Side.LONG:
        return stop_loss * (1 - buffer_pct)
    return stop_loss * (1 + buffer_pct)

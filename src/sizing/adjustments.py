"""
Layer 4: Position size adjustment factors.

Each factor returns a multiplier (0.0 to 1.0) applied to the base Kelly size.
All functions are pure — no side effects.
"""

import numpy as np
from core.safety import (
    MDD_SIZE_REDUCTION_THRESHOLD,
    MDD_SIZE_REDUCTION_FACTOR,
    CONSECUTIVE_LOSS_THRESHOLD,
    CONSECUTIVE_LOSS_SIZE_FACTOR,
)


def volatility_adjustment(current_atr_pct: float, baseline_atr_pct: float = 0.02) -> float:
    """
    Scale position size inversely with volatility.
    Higher volatility → smaller position.

    Args:
        current_atr_pct: current ATR as percentage of price
        baseline_atr_pct: "normal" ATR percentage (default 2%)

    Returns:
        Multiplier (0.1 to 1.5)
    """
    if current_atr_pct <= 0:
        return 1.0

    ratio = baseline_atr_pct / current_atr_pct
    # Clamp between 0.1 (very high vol) and 1.5 (very low vol)
    return max(0.1, min(ratio, 1.5))


def correlation_adjustment(
    existing_position_sides: list[str],
    new_signal_side: str,
    correlation_to_existing: float,
) -> float:
    """
    Reduce size if new position is highly correlated with existing positions.

    Args:
        existing_position_sides: list of "long"/"short" for current positions
        new_signal_side: "long" or "short"
        correlation_to_existing: max correlation with any existing position's pair

    Returns:
        Multiplier (0.3 to 1.0)
    """
    if not existing_position_sides:
        return 1.0

    # Count same-direction positions
    same_direction = sum(1 for s in existing_position_sides if s == new_signal_side)

    if same_direction == 0:
        return 1.0

    # High correlation + same direction = reduce more
    if correlation_to_existing > 0.85:
        # Highly correlated: reduce aggressively
        return max(0.3, 1.0 - (same_direction * 0.2))
    elif correlation_to_existing > 0.6:
        return max(0.5, 1.0 - (same_direction * 0.1))
    else:
        return 1.0


def drawdown_adjustment(current_mdd: float) -> float:
    """
    Reduce size proportionally to drawdown.
    At MDD_SIZE_REDUCTION_THRESHOLD (10%), halve the size.

    Args:
        current_mdd: current maximum drawdown (0.0 to 1.0)

    Returns:
        Multiplier (MDD_SIZE_REDUCTION_FACTOR to 1.0)
    """
    if current_mdd <= 0:
        return 1.0

    if current_mdd >= MDD_SIZE_REDUCTION_THRESHOLD:
        return MDD_SIZE_REDUCTION_FACTOR

    # Linear interpolation between 1.0 and reduction factor
    ratio = current_mdd / MDD_SIZE_REDUCTION_THRESHOLD
    return 1.0 - ratio * (1.0 - MDD_SIZE_REDUCTION_FACTOR)


def consecutive_loss_adjustment(consecutive_losses: int) -> float:
    """
    Reduce size after consecutive losses.

    Args:
        consecutive_losses: number of consecutive losing trades

    Returns:
        Multiplier (CONSECUTIVE_LOSS_SIZE_FACTOR to 1.0)
    """
    if consecutive_losses < CONSECUTIVE_LOSS_THRESHOLD:
        return 1.0
    return CONSECUTIVE_LOSS_SIZE_FACTOR


def apply_all_adjustments(
    base_size: float,
    current_atr_pct: float,
    existing_position_sides: list[str],
    new_signal_side: str,
    correlation_to_existing: float,
    current_mdd: float,
    consecutive_losses: int,
    ml_confidence: float = 1.0,
    min_size: float = 0.0,
    max_size: float = float("inf"),
) -> float:
    """Apply all adjustment factors to base position size."""
    vol_adj = volatility_adjustment(current_atr_pct)
    corr_adj = correlation_adjustment(existing_position_sides, new_signal_side, correlation_to_existing)
    dd_adj = drawdown_adjustment(current_mdd)
    loss_adj = consecutive_loss_adjustment(consecutive_losses)

    # ML confidence: 0.5 to 1.5 multiplier (centered at 1.0)
    ml_adj = max(0.5, min(ml_confidence, 1.5))

    adjusted = base_size * vol_adj * corr_adj * dd_adj * loss_adj * ml_adj

    return max(min_size, min(adjusted, max_size))

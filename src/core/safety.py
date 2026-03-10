"""
Layer 0: Hard-coded safety constants.

These values are IMMUTABLE. No ML model, no config file, no environment variable
can override them. They exist to prevent catastrophic loss regardless of what
any other part of the system decides.
"""

from typing import Final

# Risk per trade: maximum 1% of total capital
MAX_RISK_PER_TRADE: Final[float] = 0.01

# Daily loss limit: 3% of total capital → halt new entries for 24h
MAX_DAILY_LOSS: Final[float] = 0.03

# Weekly loss limit: 5% of total capital → halt new entries for 1 week
MAX_WEEKLY_LOSS: Final[float] = 0.05

# Absolute leverage ceiling
MAX_LEVERAGE: Final[int] = 10

# Maximum drawdown before full system halt
MAX_MDD: Final[float] = 0.15

# Minimum 24h volume for a pair to be tradeable ($USD)
MIN_PAIR_VOLUME_24H: Final[float] = 10_000_000.0

# Position size must not exceed 2% of pair's 24h volume
MAX_POSITION_VS_VOLUME: Final[float] = 0.02

# Every position MUST have an exchange-side stop-loss
REQUIRE_EXCHANGE_STOP_LOSS: Final[bool] = True

# Maximum concurrent open positions
MAX_CONCURRENT_POSITIONS: Final[int] = 5

# Maximum API errors per hour before trading halt
MAX_API_ERRORS_PER_HOUR: Final[int] = 5

# Maximum data staleness in seconds before trading halt
MAX_DATA_STALENESS_SECONDS: Final[int] = 1800  # 30 minutes

# Consecutive losses before position size reduction
CONSECUTIVE_LOSS_THRESHOLD: Final[int] = 5

# Size reduction factor on consecutive losses
CONSECUTIVE_LOSS_SIZE_FACTOR: Final[float] = 0.7  # reduce to 70%

# Consecutive wins to recover from loss reduction
CONSECUTIVE_WIN_RECOVERY: Final[int] = 3

# MDD threshold for position size halving
MDD_SIZE_REDUCTION_THRESHOLD: Final[float] = 0.10

# MDD size reduction factor
MDD_SIZE_REDUCTION_FACTOR: Final[float] = 0.5


def validate_leverage(leverage: float) -> float:
    """Clamp leverage to safety limit."""
    if leverage <= 0:
        return 1.0
    return min(leverage, float(MAX_LEVERAGE))


def validate_position_risk(risk_fraction: float) -> float:
    """Clamp risk fraction to safety limit."""
    if risk_fraction <= 0:
        return 0.0
    return min(risk_fraction, MAX_RISK_PER_TRADE)


def validate_position_size(
    position_value_usd: float,
    pair_volume_24h: float,
) -> float:
    """Clamp position size relative to pair's 24h volume."""
    max_allowed = pair_volume_24h * MAX_POSITION_VS_VOLUME
    if position_value_usd <= 0:
        return 0.0
    return min(position_value_usd, max_allowed)


def is_pair_eligible(volume_24h: float) -> bool:
    """Check if pair meets minimum volume requirement."""
    return volume_24h >= MIN_PAIR_VOLUME_24H

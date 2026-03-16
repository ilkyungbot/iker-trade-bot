"""
Layer 0: Hard-coded safety constants.

시그널 봇 전환 후 남은 안전 상수: API 에러/데이터 신선도 관련만 유지.
자금 관련 상수 제거 (자동 주문 없음).
"""

from typing import Final

# Maximum API errors per hour before signal halt
MAX_API_ERRORS_PER_HOUR: Final[int] = 5

# Maximum data staleness in seconds before signal halt
MAX_DATA_STALENESS_SECONDS: Final[int] = 1800  # 30 minutes

# Minimum 24h volume for a pair to be eligible
MIN_PAIR_VOLUME_24H: Final[float] = 10_000_000.0


def is_pair_eligible(volume_24h: float) -> bool:
    """Check if pair meets minimum volume requirement."""
    return volume_24h >= MIN_PAIR_VOLUME_24H

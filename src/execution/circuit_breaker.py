"""
Layer 7: Circuit breakers.

Monitors portfolio health and halts trading when thresholds are breached.
These are the last line of defense against catastrophic loss.
"""

import logging
from datetime import datetime, timezone, timedelta

from core.types import CircuitBreakerState, PortfolioState
from core.safety import (
    MAX_DAILY_LOSS,
    MAX_WEEKLY_LOSS,
    MAX_MDD,
    MDD_SIZE_REDUCTION_THRESHOLD,
    MAX_API_ERRORS_PER_HOUR,
    MAX_DATA_STALENESS_SECONDS,
    CONSECUTIVE_LOSS_THRESHOLD,
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Monitors and enforces trading halts."""

    def __init__(self):
        self._api_errors: list[datetime] = []
        self._halt_until: datetime | None = None
        self._full_stop: bool = False
        self._size_reduction: float = 1.0

    @property
    def is_halted(self) -> bool:
        """Check if trading is currently halted."""
        if self._full_stop:
            return True
        if self._halt_until and datetime.now(timezone.utc) < self._halt_until:
            return True
        return False

    @property
    def size_multiplier(self) -> float:
        """Get current position size multiplier."""
        return self._size_reduction

    @property
    def is_full_stop(self) -> bool:
        return self._full_stop

    def check(self, state: PortfolioState) -> CircuitBreakerState:
        """
        Check all circuit breaker conditions.
        Returns the most severe state triggered.
        """
        # MDD > 15% → full stop
        if state.current_mdd >= MAX_MDD:
            self._full_stop = True
            state.circuit_breaker_state = CircuitBreakerState.FULL_STOP
            logger.critical(
                f"FULL STOP: MDD {state.current_mdd:.1%} >= {MAX_MDD:.0%}. "
                "Manual intervention required to resume."
            )
            return CircuitBreakerState.FULL_STOP

        # MDD > 10% → size reduction
        if state.current_mdd >= MDD_SIZE_REDUCTION_THRESHOLD:
            self._size_reduction = 0.5
            state.circuit_breaker_state = CircuitBreakerState.SIZE_REDUCED
            logger.warning(
                f"SIZE REDUCED: MDD {state.current_mdd:.1%} >= {MDD_SIZE_REDUCTION_THRESHOLD:.0%}. "
                "Positions halved."
            )

        # Weekly loss > 5% → halt 1 week
        initial_cap = state.total_capital - state.weekly_pnl  # approximate initial
        if initial_cap > 0:
            weekly_loss_pct = abs(min(state.weekly_pnl, 0)) / initial_cap
            if weekly_loss_pct >= MAX_WEEKLY_LOSS:
                self._halt_until = datetime.now(timezone.utc) + timedelta(weeks=1)
                state.circuit_breaker_state = CircuitBreakerState.WEEKLY_HALT
                logger.warning(
                    f"WEEKLY HALT: Loss {weekly_loss_pct:.1%} >= {MAX_WEEKLY_LOSS:.0%}. "
                    f"Halted until {self._halt_until}."
                )
                return CircuitBreakerState.WEEKLY_HALT

        # Daily loss > 3% → halt 24h
        if initial_cap > 0:
            daily_loss_pct = abs(min(state.daily_pnl, 0)) / initial_cap
            if daily_loss_pct >= MAX_DAILY_LOSS:
                self._halt_until = datetime.now(timezone.utc) + timedelta(hours=24)
                state.circuit_breaker_state = CircuitBreakerState.DAILY_HALT
                logger.warning(
                    f"DAILY HALT: Loss {daily_loss_pct:.1%} >= {MAX_DAILY_LOSS:.0%}. "
                    f"Halted for 24h."
                )
                return CircuitBreakerState.DAILY_HALT

        # Consecutive losses → size reduction
        if state.consecutive_losses >= CONSECUTIVE_LOSS_THRESHOLD:
            self._size_reduction = 0.7
            logger.warning(
                f"Consecutive loss streak: {state.consecutive_losses}. Size reduced to 70%."
            )

        # Recovery from consecutive losses
        if state.consecutive_wins >= 3 and self._size_reduction < 1.0:
            self._size_reduction = 1.0
            logger.info("Consecutive win recovery: size restored to 100%.")

        return state.circuit_breaker_state

    def record_api_error(self) -> bool:
        """
        Record an API error. Returns True if threshold breached (should halt).
        """
        now = datetime.now(timezone.utc)
        self._api_errors.append(now)

        # Remove errors older than 1 hour
        cutoff = now - timedelta(hours=1)
        self._api_errors = [e for e in self._api_errors if e > cutoff]

        if len(self._api_errors) >= MAX_API_ERRORS_PER_HOUR:
            self._halt_until = now + timedelta(hours=1)
            logger.error(
                f"API ERROR HALT: {len(self._api_errors)} errors in 1 hour. "
                "Halted for 1 hour."
            )
            return True
        return False

    def check_data_staleness(self, last_data_time: datetime) -> bool:
        """
        Check if data is too stale. Returns True if should halt.
        """
        now = datetime.now(timezone.utc)
        if last_data_time.tzinfo is None:
            last_data_time = last_data_time.replace(tzinfo=timezone.utc)

        staleness = (now - last_data_time).total_seconds()
        if staleness > MAX_DATA_STALENESS_SECONDS:
            logger.warning(
                f"DATA STALE: Last data {staleness:.0f}s ago "
                f"(limit: {MAX_DATA_STALENESS_SECONDS}s). Trading halted."
            )
            return True
        return False

    def manual_resume(self) -> None:
        """Manual resume after full stop. Requires explicit user action."""
        self._full_stop = False
        self._halt_until = None
        self._size_reduction = 1.0
        logger.info("Circuit breaker manually resumed.")

    def reset(self) -> None:
        """Full reset of all circuit breaker state."""
        self._api_errors = []
        self._halt_until = None
        self._full_stop = False
        self._size_reduction = 1.0

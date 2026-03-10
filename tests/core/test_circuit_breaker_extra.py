"""Extra tests for circuit breaker — gap coverage."""

from datetime import datetime, timezone, timedelta

from execution.circuit_breaker import CircuitBreaker
from core.types import PortfolioState, CircuitBreakerState
from core.safety import MDD_SIZE_REDUCTION_THRESHOLD, MAX_DATA_STALENESS_SECONDS


class TestWinStreakDoesNotResetSizeDuringHighMDD:
    def test_size_stays_reduced_when_mdd_above_threshold(self):
        """Win streak should NOT restore size if MDD is still >= threshold."""
        cb = CircuitBreaker()

        # First, trigger MDD size reduction
        state = PortfolioState(
            total_capital=90_000.0,
            available_capital=90_000.0,
            peak_capital=100_000.0,
            current_mdd=MDD_SIZE_REDUCTION_THRESHOLD,  # exactly at threshold
            consecutive_wins=5,
            consecutive_losses=0,
        )
        cb.check(state)
        assert cb.size_multiplier == 0.5, "Size should be halved when MDD >= threshold"

        # Now simulate 3+ consecutive wins but MDD is still at threshold
        state.consecutive_wins = 5
        state.consecutive_losses = 0
        cb.check(state)
        # Size should still be reduced because MDD is still at threshold
        assert cb.size_multiplier == 0.5, (
            "Win streak should NOT restore size while MDD >= threshold"
        )


class TestCheckDataStaleness:
    def test_stale_data_sets_halt(self):
        cb = CircuitBreaker()
        assert cb.is_halted is False

        # Data from well beyond the staleness threshold
        old_time = datetime.now(timezone.utc) - timedelta(
            seconds=MAX_DATA_STALENESS_SECONDS + 60
        )
        result = cb.check_data_staleness(old_time)

        assert result is True
        assert cb.is_halted is True

    def test_fresh_data_does_not_halt(self):
        cb = CircuitBreaker()
        fresh_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        result = cb.check_data_staleness(fresh_time)

        assert result is False
        assert cb.is_halted is False


class TestDailyLossUsesCorrectBase:
    def test_daily_loss_uses_daily_initial_capital(self):
        """Daily loss percentage should be computed from start-of-day capital
        (total_capital - daily_pnl), not from weekly or all-time capital."""
        cb = CircuitBreaker()

        # Scenario: started the day with 10_000, lost 300 (3%).
        # weekly_pnl kept below weekly threshold so it doesn't fire first.
        state = PortfolioState(
            total_capital=9_700.0,
            available_capital=9_700.0,
            peak_capital=10_000.0,
            current_mdd=0.03,  # below MDD thresholds
            daily_pnl=-300.0,  # 3% of daily-start capital (10_000)
            weekly_pnl=-300.0,  # same as daily — below 5% weekly threshold
        )

        result = cb.check(state)
        # daily_initial_cap = total_capital - daily_pnl = 9700 - (-300) = 10_000
        # daily_loss_pct = 300 / 10_000 = 0.03 = 3% — exactly at MAX_DAILY_LOSS
        assert result == CircuitBreakerState.DAILY_HALT

    def test_daily_loss_below_threshold_no_halt(self):
        cb = CircuitBreaker()

        state = PortfolioState(
            total_capital=9_800.0,
            available_capital=9_800.0,
            peak_capital=10_000.0,
            current_mdd=0.02,
            daily_pnl=-200.0,  # 2% of 10_000 — below 3% threshold
            weekly_pnl=-200.0,
        )

        result = cb.check(state)
        assert result != CircuitBreakerState.DAILY_HALT

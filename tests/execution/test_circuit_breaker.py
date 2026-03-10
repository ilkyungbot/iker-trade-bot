"""Tests for circuit breaker."""

from datetime import datetime, timezone, timedelta
from execution.circuit_breaker import CircuitBreaker
from core.types import PortfolioState, CircuitBreakerState


def _make_state(
    total_capital: float = 10000.0,
    daily_pnl: float = 0.0,
    weekly_pnl: float = 0.0,
    current_mdd: float = 0.0,
    consecutive_losses: int = 0,
    consecutive_wins: int = 0,
) -> PortfolioState:
    return PortfolioState(
        total_capital=total_capital,
        available_capital=total_capital,
        daily_pnl=daily_pnl,
        weekly_pnl=weekly_pnl,
        peak_capital=total_capital - (total_capital * current_mdd) + total_capital * current_mdd,
        current_mdd=current_mdd,
        consecutive_losses=consecutive_losses,
        consecutive_wins=consecutive_wins,
    )


class TestDailyHalt:
    def test_triggers_on_3pct_loss(self):
        cb = CircuitBreaker()
        state = _make_state(total_capital=9700, daily_pnl=-310)  # 310/10010 > 3%
        result = cb.check(state)
        assert result == CircuitBreakerState.DAILY_HALT
        assert cb.is_halted

    def test_no_halt_on_small_loss(self):
        cb = CircuitBreaker()
        state = _make_state(total_capital=9900, daily_pnl=-50)
        result = cb.check(state)
        assert result != CircuitBreakerState.DAILY_HALT


class TestWeeklyHalt:
    def test_triggers_on_5pct_loss(self):
        cb = CircuitBreaker()
        state = _make_state(total_capital=9500, weekly_pnl=-530)  # 530/10030 > 5%
        result = cb.check(state)
        assert result == CircuitBreakerState.WEEKLY_HALT


class TestMDDFullStop:
    def test_triggers_at_15pct(self):
        cb = CircuitBreaker()
        state = _make_state(current_mdd=0.16)
        result = cb.check(state)
        assert result == CircuitBreakerState.FULL_STOP
        assert cb.is_full_stop
        assert cb.is_halted

    def test_requires_manual_resume(self):
        cb = CircuitBreaker()
        state = _make_state(current_mdd=0.16)
        cb.check(state)
        assert cb.is_halted

        cb.manual_resume()
        assert not cb.is_halted
        assert not cb.is_full_stop


class TestMDDSizeReduction:
    def test_halves_at_10pct(self):
        cb = CircuitBreaker()
        state = _make_state(current_mdd=0.11)
        cb.check(state)
        assert cb.size_multiplier == 0.5


class TestConsecutiveLosses:
    def test_reduces_size_after_5_losses(self):
        cb = CircuitBreaker()
        state = _make_state(consecutive_losses=5)
        cb.check(state)
        assert cb.size_multiplier == 0.7

    def test_recovers_after_3_wins(self):
        cb = CircuitBreaker()
        # First trigger reduction
        state = _make_state(consecutive_losses=5)
        cb.check(state)
        assert cb.size_multiplier == 0.7

        # Then recover
        state2 = _make_state(consecutive_losses=0, consecutive_wins=3)
        cb.check(state2)
        assert cb.size_multiplier == 1.0


class TestAPIErrors:
    def test_halt_after_5_errors(self):
        cb = CircuitBreaker()
        for _ in range(4):
            assert cb.record_api_error() is False
        assert cb.record_api_error() is True
        assert cb.is_halted


class TestReset:
    def test_full_reset(self):
        cb = CircuitBreaker()
        state = _make_state(current_mdd=0.16)
        cb.check(state)
        assert cb.is_halted

        cb.reset()
        assert not cb.is_halted
        assert cb.size_multiplier == 1.0

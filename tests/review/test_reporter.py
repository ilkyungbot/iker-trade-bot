"""Tests for Telegram reporter formatting."""

from datetime import datetime, timezone
from review.reporter import Reporter
from review.performance import PerformanceMetrics
from core.types import (
    Trade, Side, StrategyName, PortfolioState, CircuitBreakerState,
    Signal, SignalAction,
)


def _make_trade(pnl: float = 100.0) -> Trade:
    return Trade(
        symbol="BTCUSDT", side=Side.LONG, strategy=StrategyName.TREND_FOLLOWING,
        entry_price=50000, exit_price=51000, quantity=0.1, leverage=5,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        pnl=pnl, pnl_percent=2.0, fees=5, slippage=0,
        stop_loss_hit=False, trailing_stop_hit=True,
    )


def _make_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_trades=50, winning_trades=25, losing_trades=25,
        win_rate=0.5, avg_win=200, avg_loss=100, payoff_ratio=2.0,
        total_pnl=2500, total_fees=250, net_pnl=2500,
        max_drawdown=500, sharpe_ratio=1.5, profit_factor=2.0,
        avg_trade_duration_hours=6.0,
    )


class TestTradeAlert:
    def test_format_winning_trade(self):
        reporter = Reporter()
        msg = reporter.format_trade_alert(_make_trade(pnl=100))
        assert "BTCUSDT" in msg
        assert "+100" in msg
        assert "LONG" in msg

    def test_format_losing_trade(self):
        reporter = Reporter()
        msg = reporter.format_trade_alert(_make_trade(pnl=-50))
        assert "-50" in msg


class TestSignalAlert:
    def test_format_long_signal(self):
        reporter = Reporter()
        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT", action=SignalAction.ENTER_LONG,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=50000, stop_loss=49000, take_profit=52000,
            confidence=0.8,
        )
        msg = reporter.format_signal_alert(signal)
        assert "LONG" in msg
        assert "80%" in msg


class TestDailyReport:
    def test_format(self):
        reporter = Reporter()
        state = PortfolioState(
            total_capital=1050000, available_capital=1000000,
            daily_pnl=5000, current_mdd=0.02,
        )
        msg = reporter.format_daily_report(
            state, _make_metrics(), datetime(2024, 1, 1),
        )
        assert "Daily Report" in msg
        assert "1,050,000" in msg
        assert "+5,000" in msg


class TestWeeklyReport:
    def test_format_with_strategies(self):
        reporter = Reporter()
        state = PortfolioState(
            total_capital=1050000, available_capital=1000000,
            weekly_pnl=15000, current_mdd=0.03,
        )
        strategy_attrs = {
            "trend_following": _make_metrics(),
            "funding_rate": _make_metrics(),
        }
        msg = reporter.format_weekly_report(state, _make_metrics(), strategy_attrs)
        assert "Weekly Report" in msg
        assert "trend_following" in msg
        assert "funding_rate" in msg

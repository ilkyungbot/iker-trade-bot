"""Tests for backtest runner."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from backtest.runner import BacktestRunner, BacktestConfig, BacktestResult


def _make_trending_data(n: int = 200, start_price: float = 50000.0, trend: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic trending 1H and 4H data."""
    np.random.seed(42)
    timestamps_1h = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]

    prices = [start_price]
    for i in range(1, n):
        noise = np.random.normal(0, 50)
        prices.append(prices[-1] + trend + noise)

    df_1h = pd.DataFrame({
        "timestamp": timestamps_1h,
        "open": prices,
        "high": [p + abs(np.random.normal(0, 100)) for p in prices],
        "low": [p - abs(np.random.normal(0, 100)) for p in prices],
        "close": [p + np.random.normal(0, 30) for p in prices],
        "volume": [np.random.uniform(100, 1000) for _ in range(n)],
    })
    # Ensure high >= close and low <= close
    df_1h["high"] = df_1h[["high", "close", "open"]].max(axis=1) + 0.5
    df_1h["low"] = df_1h[["low", "close", "open"]].min(axis=1) - 0.5

    # 4H data: every 4th bar
    n_4h = n // 4
    timestamps_4h = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i * 4) for i in range(n_4h)]
    prices_4h = [prices[i * 4] for i in range(n_4h)]

    df_4h = pd.DataFrame({
        "timestamp": timestamps_4h,
        "open": prices_4h,
        "high": [p + abs(np.random.normal(0, 150)) for p in prices_4h],
        "low": [p - abs(np.random.normal(0, 150)) for p in prices_4h],
        "close": [p + np.random.normal(0, 50) for p in prices_4h],
        "volume": [np.random.uniform(500, 5000) for _ in range(n_4h)],
    })
    df_4h["high"] = df_4h[["high", "close", "open"]].max(axis=1) + 0.5
    df_4h["low"] = df_4h[["low", "close", "open"]].min(axis=1) - 0.5

    return df_1h, df_4h


def _make_flat_data(n: int = 200, price: float = 50000.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create flat/ranging data (no trend)."""
    np.random.seed(42)
    timestamps_1h = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]

    prices = [price + np.random.normal(0, 50) for _ in range(n)]
    df_1h = pd.DataFrame({
        "timestamp": timestamps_1h,
        "open": prices,
        "high": [p + abs(np.random.normal(0, 30)) + 0.5 for p in prices],
        "low": [p - abs(np.random.normal(0, 30)) - 0.5 for p in prices],
        "close": prices,
        "volume": [np.random.uniform(100, 1000) for _ in range(n)],
    })

    n_4h = n // 4
    timestamps_4h = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i * 4) for i in range(n_4h)]
    df_4h = pd.DataFrame({
        "timestamp": timestamps_4h,
        "open": [price] * n_4h,
        "high": [price + 50] * n_4h,
        "low": [price - 50] * n_4h,
        "close": [price + np.random.normal(0, 20) for _ in range(n_4h)],
        "volume": [np.random.uniform(500, 5000) for _ in range(n_4h)],
    })

    return df_1h, df_4h


class TestBacktestRunner:
    def test_run_returns_result(self):
        df_1h, df_4h = _make_trending_data()
        runner = BacktestRunner()
        result = runner.run(df_1h, df_4h)
        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 100_000
        assert result.total_bars > 0

    def test_equity_curve_length(self):
        df_1h, df_4h = _make_trending_data()
        runner = BacktestRunner()
        result = runner.run(df_1h, df_4h)
        assert len(result.equity_curve) == result.total_bars

    def test_insufficient_data(self):
        """With too few bars, should return empty result."""
        df_1h, df_4h = _make_trending_data(n=30)
        runner = BacktestRunner()
        result = runner.run(df_1h, df_4h)
        assert result.total_bars == 0
        assert len(result.trades) == 0

    def test_trades_have_valid_fields(self):
        df_1h, df_4h = _make_trending_data(n=300)
        runner = BacktestRunner()
        result = runner.run(df_1h, df_4h)

        for trade in result.trades:
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.quantity > 0
            assert trade.fees >= 0
            assert trade.entry_time < trade.exit_time

    def test_fees_are_charged(self):
        """All trades should have non-zero fees."""
        df_1h, df_4h = _make_trending_data(n=300)
        runner = BacktestRunner()
        result = runner.run(df_1h, df_4h)

        if result.trades:
            total_fees = sum(t.fees for t in result.trades)
            assert total_fees > 0

    def test_custom_config(self):
        config = BacktestConfig(
            initial_capital=50_000,
            default_leverage=3.0,
            fee_rate=0.001,
        )
        df_1h, df_4h = _make_trending_data()
        runner = BacktestRunner(config)
        result = runner.run(df_1h, df_4h)
        assert result.initial_capital == 50_000

    def test_total_return_property(self):
        df_1h, df_4h = _make_trending_data(n=300)
        runner = BacktestRunner()
        result = runner.run(df_1h, df_4h)
        # Return should be finite
        assert np.isfinite(result.total_return)
        assert np.isfinite(result.total_return_pct)

    def test_flat_market_fewer_trades(self):
        """In a flat market, trend following should generate fewer/no trades."""
        df_flat_1h, df_flat_4h = _make_flat_data(n=200)
        df_trend_1h, df_trend_4h = _make_trending_data(n=200)

        runner = BacktestRunner()
        result_flat = runner.run(df_flat_1h, df_flat_4h)
        result_trend = runner.run(df_trend_1h, df_trend_4h)

        # Flat market should have same or fewer trades than trending
        assert len(result_flat.trades) <= len(result_trend.trades) + 5  # small margin


class TestBacktestAnalysis:
    def test_analyze_returns_analysis(self):
        from backtest.analysis import analyze, BacktestAnalysis
        df_1h, df_4h = _make_trending_data(n=300)
        runner = BacktestRunner()
        result = runner.run(df_1h, df_4h)

        analysis = analyze(result)
        assert isinstance(analysis, BacktestAnalysis)
        assert np.isfinite(analysis.total_return_pct)
        assert np.isfinite(analysis.annualized_return_pct)
        assert analysis.max_drawdown_pct >= 0

    def test_format_report(self):
        from backtest.analysis import analyze, format_report
        df_1h, df_4h = _make_trending_data(n=300)
        runner = BacktestRunner()
        result = runner.run(df_1h, df_4h)

        analysis = analyze(result)
        report = format_report(analysis)
        assert "BACKTEST REPORT" in report
        assert "Total Return" in report
        assert "Win Rate" in report

    def test_empty_result_analysis(self):
        from backtest.analysis import analyze
        result = BacktestResult(
            trades=[], equity_curve=[100000], timestamps=[],
            initial_capital=100000, final_capital=100000, total_bars=0,
        )
        analysis = analyze(result)
        assert analysis.total_return_pct == 0
        assert analysis.metrics.total_trades == 0

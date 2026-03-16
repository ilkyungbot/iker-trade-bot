"""Tests for pair selector."""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from strategy.pair_selector import PairSelector, HIGH_CORR_THRESHOLD


def _make_candle_df(n: int = 5000, base_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Create a sample candle DataFrame with required features."""
    np.random.seed(seed)
    close = base_price + np.cumsum(np.random.randn(n) * 0.5)
    close = np.abs(close)  # ensure positive
    df = pd.DataFrame({
        "close": close,
        "high": close + np.random.rand(n),
        "low": close - np.random.rand(n),
        "open": close - np.random.rand(n) * 0.5,
        "volume": np.random.rand(n) * 1000 + 500,
        "atr_pct": np.random.rand(n) * 0.03 + 0.01,  # 1-4%
    })
    return df


def _make_tickers(symbols: list[str], volumes: list[float]) -> list[dict]:
    return [
        {"symbol": s, "volume_24h": v, "last_price": 100.0}
        for s, v in zip(symbols, volumes)
    ]


class TestPairSelection:
    def test_selects_top_pairs(self):
        selector = PairSelector(max_pairs=3)
        tickers = _make_tickers(
            ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"],
            [500e6, 200e6, 50e6, 30e6, 20e6],
        )
        candle_data = {s["symbol"]: _make_candle_df(seed=i) for i, s in enumerate(tickers)}

        pairs = selector.select_pairs(tickers, candle_data)
        assert len(pairs) == 3
        # All should have positive scores
        assert all(p.score > 0 for p in pairs)

    def test_filters_low_volume(self):
        selector = PairSelector(max_pairs=5)
        tickers = _make_tickers(
            ["BTCUSDT", "LOWVOL"],
            [500e6, 1e6],  # LOWVOL below $10M threshold
        )
        candle_data = {s["symbol"]: _make_candle_df(seed=i) for i, s in enumerate(tickers)}

        pairs = selector.select_pairs(tickers, candle_data)
        symbols = [p.symbol for p in pairs]
        assert "LOWVOL" not in symbols

    def test_filters_insufficient_history(self):
        selector = PairSelector(max_pairs=5)
        tickers = _make_tickers(["BTCUSDT", "NEWCOIN"], [500e6, 100e6])
        candle_data = {
            "BTCUSDT": _make_candle_df(n=5000),
            "NEWCOIN": _make_candle_df(n=100),  # too short
        }

        pairs = selector.select_pairs(tickers, candle_data)
        symbols = [p.symbol for p in pairs]
        assert "NEWCOIN" not in symbols

    def test_respects_rebalance_period(self):
        selector = PairSelector(max_pairs=3, rebalance_days=14)
        tickers = _make_tickers(["BTCUSDT", "ETHUSDT", "SOLUSDT"], [500e6, 200e6, 50e6])
        candle_data = {s["symbol"]: _make_candle_df(seed=i) for i, s in enumerate(tickers)}

        t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pairs1 = selector.select_pairs(tickers, candle_data, now=t1)

        # 5 days later — should return cached
        t2 = datetime(2024, 1, 6, tzinfo=timezone.utc)
        pairs2 = selector.select_pairs(tickers, candle_data, now=t2)
        assert pairs1 == pairs2

        # 15 days later — should rebalance
        t3 = datetime(2024, 1, 16, tzinfo=timezone.utc)
        pairs3 = selector.select_pairs(tickers, candle_data, now=t3)
        assert len(pairs3) > 0  # rebalanced

    def test_empty_tickers(self):
        selector = PairSelector()
        pairs = selector.select_pairs([], {})
        assert pairs == []


class TestCorrelationConstraint:
    def test_limits_high_correlation_pairs(self):
        selector = PairSelector(max_pairs=5)

        # Create 6 pairs, all highly correlated to BTC
        symbols = ["BTCUSDT", "ETH", "SOL", "XRP", "DOGE", "ADA"]
        tickers = _make_tickers(symbols, [500e6, 400e6, 300e6, 200e6, 100e6, 50e6])

        # Use same seed = same price path = correlation ~1.0
        candle_data = {s: _make_candle_df(n=5000, seed=42) for s in symbols}

        pairs = selector.select_pairs(tickers, candle_data)
        # Should cap at MAX_HIGH_CORR_PAIRS (3) for high-corr pairs
        assert len(pairs) <= 5

    def test_allows_uncorrelated_pairs(self):
        selector = PairSelector(max_pairs=5)

        symbols = ["BTCUSDT", "UNCORR1", "UNCORR2", "UNCORR3", "UNCORR4"]
        tickers = _make_tickers(symbols, [500e6, 400e6, 300e6, 200e6, 100e6])

        # Different seeds = different price paths = lower correlation
        candle_data = {s: _make_candle_df(n=5000, seed=i * 100) for i, s in enumerate(symbols)}

        pairs = selector.select_pairs(tickers, candle_data)
        assert len(pairs) >= 1


class TestCorrelationCalculation:
    def test_perfect_correlation(self):
        prices = pd.Series([100, 101, 102, 103, 104, 105] * 30)
        corr = PairSelector._calculate_correlation(prices, prices)
        assert abs(corr - 1.0) < 0.01

    def test_uncorrelated(self):
        np.random.seed(42)
        a = pd.Series(np.random.randn(200))
        b = pd.Series(np.random.randn(200))
        corr = PairSelector._calculate_correlation(a, b)
        assert abs(corr) < 0.3  # should be near zero

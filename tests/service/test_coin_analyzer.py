import pytest
import pandas as pd
from service.coin_analyzer import CoinAnalyzer


def _make_df(n=60, **overrides):
    data = {
        "close": [100.0] * n, "open": [99.0] * n, "high": [101.0] * n, "low": [98.0] * n,
        "atr": [2.0] * n, "adx": [25.0] * n, "rsi": [50.0] * n,
        "ema_golden_cross": [False] * n, "ema_death_cross": [False] * n,
        "rsi_cross_up": [False] * n, "rsi_cross_down": [False] * n,
        "bb_lower": [95.0] * n, "bb_upper": [105.0] * n,
        "macd_hist_cross_up": [False] * n, "macd_hist_cross_down": [False] * n,
        "volume_anomaly": [False] * n,
        "candle_hammer": [False] * n, "candle_bullish_engulfing": [False] * n,
        "candle_morning_star": [False] * n,
        "candle_inverted_hammer": [False] * n, "candle_bearish_engulfing": [False] * n,
    }
    for k, v in overrides.items():
        data[k] = [v] * n if not isinstance(v, list) else v
    return pd.DataFrame(data)


class TestScorePair:
    def setup_method(self):
        self.analyzer = CoinAnalyzer(collector=None, config=None)

    def test_insufficient_data(self):
        assert self.analyzer.score_pair(_make_df(n=10), "BTCUSDT") is None

    def test_zero_atr(self):
        assert self.analyzer.score_pair(_make_df(atr=0.0), "BTCUSDT") is None

    def test_strong_long(self):
        result = self.analyzer.score_pair(
            _make_df(ema_golden_cross=True, rsi_cross_up=True, macd_hist_cross_up=True),
            "BTCUSDT",
        )
        assert result is not None
        assert result["direction"] == "long"
        assert result["score"] >= 3
        assert result["quality"] == "strong"

    def test_short_signals(self):
        result = self.analyzer.score_pair(
            _make_df(ema_death_cross=True, rsi_cross_down=True, macd_hist_cross_down=True, adx=10.0),
            "ETHUSDT",
        )
        assert result is not None
        assert result["direction"] == "short"
        assert result["score"] >= 3

    def test_adx_bonus(self):
        result = self.analyzer.score_pair(_make_df(adx=25.0), "BTCUSDT")
        assert result is not None
        assert result["score"] >= 1  # ADX > 20 contributes

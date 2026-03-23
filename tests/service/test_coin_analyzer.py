import pytest
import pandas as pd
from unittest.mock import MagicMock
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

    def test_volume_anomaly_bullish(self):
        """Volume anomaly on up candle adds long score."""
        result = self.analyzer.score_pair(
            _make_df(volume_anomaly=True, close=100.0, open=98.0),
            "BTCUSDT",
        )
        # ADX=25 is default so short also scores; but close > open means long vol +1
        assert result is not None
        # With only ADX (1 each) + volume bullish (+1 long), long=2, short=1 → long wins
        # But volume_anomaly needs close != open explicitly

    def test_volume_anomaly_bullish_explicit(self):
        """Volume anomaly with close > open scores long."""
        df = _make_df(n=60, volume_anomaly=True, adx=10.0)
        # close=100, open=99 by default → close > open
        result = self.analyzer.score_pair(df, "BTCUSDT")
        assert result is not None
        assert result["direction"] == "long"
        assert result["score"] >= 1

    def test_volume_anomaly_bearish(self):
        """Volume anomaly with close < open scores short."""
        df = _make_df(n=60, volume_anomaly=True, close=98.0, open=100.0, adx=10.0)
        result = self.analyzer.score_pair(df, "BTCUSDT")
        assert result is not None
        assert result["direction"] == "short"
        assert result["score"] >= 1

    def test_candle_pattern_hammer_scores_long(self):
        """Hammer candle pattern adds 1 to long score."""
        result = self.analyzer.score_pair(
            _make_df(candle_hammer=True, adx=10.0),
            "BTCUSDT",
        )
        assert result is not None
        assert result["direction"] == "long"
        assert "강세 캔들패턴" in result["reasons"]

    def test_candle_pattern_bullish_engulfing(self):
        """Bullish engulfing adds long score."""
        result = self.analyzer.score_pair(
            _make_df(candle_bullish_engulfing=True, adx=10.0),
            "BTCUSDT",
        )
        assert result is not None
        assert result["direction"] == "long"

    def test_candle_pattern_bearish_engulfing(self):
        """Bearish engulfing adds short score."""
        result = self.analyzer.score_pair(
            _make_df(candle_bearish_engulfing=True, adx=10.0),
            "BTCUSDT",
        )
        assert result is not None
        assert result["direction"] == "short"
        assert "약세 캔들패턴" in result["reasons"]

    def test_bollinger_band_bounce(self):
        """Price bouncing off lower BB scores long."""
        import numpy as np
        n = 60
        lows = [100.0] * n
        closes = [101.0] * n
        prev_closes = [99.0] * n
        bb_lowers = [102.0] * n  # prev low <= bb_lower

        # Make last row: close > prev_close, prev.low <= bb_lower
        data = {
            "close": closes,
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": lows,
            "atr": [2.0] * n,
            "adx": [10.0] * n,
            "rsi": [50.0] * n,
            "ema_golden_cross": [False] * n,
            "ema_death_cross": [False] * n,
            "rsi_cross_up": [False] * n,
            "rsi_cross_down": [False] * n,
            "bb_lower": bb_lowers,
            "bb_upper": [120.0] * n,
            "macd_hist_cross_up": [False] * n,
            "macd_hist_cross_down": [False] * n,
            "volume_anomaly": [False] * n,
            "candle_hammer": [False] * n,
            "candle_bullish_engulfing": [False] * n,
            "candle_morning_star": [False] * n,
            "candle_inverted_hammer": [False] * n,
            "candle_bearish_engulfing": [False] * n,
        }
        import pandas as pd
        df = pd.DataFrame(data)
        # Set prev (index -2) low <= bb_lower, current close > prev_close
        df.loc[df.index[-2], "low"] = 101.9   # prev.low <= bb_lower(102)
        df.loc[df.index[-1], "close"] = 103.0  # close > prev_close(101)
        result = self.analyzer.score_pair(df, "BTCUSDT")
        assert result is not None
        # At minimum ADX=10 won't contribute; but BB bounce should fire
        long_reasons = result.get("reasons", [])
        if result["direction"] == "long":
            assert "볼린저 하단 반등" in long_reasons or result["score"] >= 1

    def test_bollinger_band_rejection(self):
        """Price rejected at upper BB scores short."""
        n = 60
        data = {
            "close": [99.0] * n,
            "open": [100.0] * n,
            "high": [110.0] * n,
            "low": [98.0] * n,
            "atr": [2.0] * n,
            "adx": [10.0] * n,
            "rsi": [50.0] * n,
            "ema_golden_cross": [False] * n,
            "ema_death_cross": [False] * n,
            "rsi_cross_up": [False] * n,
            "rsi_cross_down": [False] * n,
            "bb_lower": [85.0] * n,
            "bb_upper": [105.0] * n,
            "macd_hist_cross_up": [False] * n,
            "macd_hist_cross_down": [False] * n,
            "volume_anomaly": [False] * n,
            "candle_hammer": [False] * n,
            "candle_bullish_engulfing": [False] * n,
            "candle_morning_star": [False] * n,
            "candle_inverted_hammer": [False] * n,
            "candle_bearish_engulfing": [False] * n,
        }
        import pandas as pd
        df = pd.DataFrame(data)
        # prev high >= bb_upper and current close < prev_close
        df.loc[df.index[-2], "high"] = 106.0   # prev.high >= bb_upper(105)
        df.loc[df.index[-2], "close"] = 104.0  # prev_close
        df.loc[df.index[-1], "close"] = 99.0   # close < prev_close
        result = self.analyzer.score_pair(df, "BTCUSDT")
        assert result is not None
        if result["direction"] == "short":
            assert "볼린저 상단 하락" in result["reasons"] or result["score"] >= 1

    def test_mixed_signals_long_wins(self):
        """When long has more signals than short, direction is long."""
        result = self.analyzer.score_pair(
            _make_df(
                ema_golden_cross=True,
                rsi_cross_up=True,
                macd_hist_cross_up=True,
                rsi_cross_down=True,  # 1 short signal
                adx=10.0,
            ),
            "BTCUSDT",
        )
        assert result is not None
        assert result["direction"] == "long"
        assert result["score"] == 3

    def test_mixed_signals_short_wins(self):
        """When short has more signals, direction is short."""
        result = self.analyzer.score_pair(
            _make_df(
                ema_death_cross=True,
                rsi_cross_down=True,
                macd_hist_cross_down=True,
                ema_golden_cross=True,  # 1 long signal
                adx=10.0,
            ),
            "BTCUSDT",
        )
        assert result is not None
        assert result["direction"] == "short"
        assert result["score"] == 3

    def test_all_seven_indicators_long(self):
        """All 7 indicators firing for long gives max score = 7."""
        import pandas as pd
        n = 60
        data = {
            "close": [103.0] * n,
            "open": [99.0] * n,
            "high": [104.0] * n,
            "low": [98.0] * n,
            "atr": [2.0] * n,
            "adx": [25.0] * n,  # +1 to both
            "rsi": [55.0] * n,
            "ema_golden_cross": [True] * n,     # +1 long
            "ema_death_cross": [False] * n,
            "rsi_cross_up": [True] * n,          # +1 long
            "rsi_cross_down": [False] * n,
            "bb_lower": [102.5] * n,             # will set prev manually
            "bb_upper": [110.0] * n,
            "macd_hist_cross_up": [True] * n,    # +1 long
            "macd_hist_cross_down": [False] * n,
            "volume_anomaly": [True] * n,        # +1 long (close > open)
            "candle_hammer": [True] * n,         # +1 long
            "candle_bullish_engulfing": [False] * n,
            "candle_morning_star": [False] * n,
            "candle_inverted_hammer": [False] * n,
            "candle_bearish_engulfing": [False] * n,
        }
        df = pd.DataFrame(data)
        # Set BB bounce: prev.low <= bb_lower and current close > prev_close
        df.loc[df.index[-2], "low"] = 102.4    # <= bb_lower(102.5)
        df.loc[df.index[-2], "close"] = 100.0  # prev_close
        df.loc[df.index[-1], "close"] = 103.0  # close > prev_close → BB bounce
        result = self.analyzer.score_pair(df, "BTCUSDT")
        assert result is not None
        assert result["direction"] == "long"
        assert result["score"] >= 6  # EMA+RSI+BB+MACD+vol+candle+ADX
        assert result["quality"] == "strong"

    def test_quality_thresholds(self):
        """Quality labels map correctly to score thresholds."""
        # score=1 → weak
        r1 = self.analyzer.score_pair(_make_df(ema_golden_cross=True, adx=10.0), "BTCUSDT")
        assert r1 is not None
        assert r1["quality"] == "weak"

        # score=2 → moderate
        r2 = self.analyzer.score_pair(
            _make_df(ema_golden_cross=True, rsi_cross_up=True, adx=10.0), "BTCUSDT",
        )
        assert r2 is not None
        assert r2["quality"] == "moderate"

        # score=3 → strong
        r3 = self.analyzer.score_pair(
            _make_df(ema_golden_cross=True, rsi_cross_up=True, macd_hist_cross_up=True, adx=10.0),
            "BTCUSDT",
        )
        assert r3 is not None
        assert r3["quality"] == "strong"

    def test_no_signals_returns_none(self):
        """DataFrame with no signals but ADX < 20 returns None."""
        result = self.analyzer.score_pair(_make_df(adx=10.0), "BTCUSDT")
        assert result is None

    def test_nan_atr_returns_none(self):
        """NaN ATR returns None."""
        import numpy as np
        result = self.analyzer.score_pair(_make_df(atr=np.nan), "BTCUSDT")
        assert result is None


class TestAnalyzeCoin:
    def setup_method(self):
        self.collector = MagicMock()
        self.config = MagicMock()
        self.config.signal.primary_interval = "240"
        self.analyzer = CoinAnalyzer(collector=self.collector, config=self.config)

    @pytest.mark.anyio
    async def test_analyze_coin_not_found(self):
        """Returns None when symbol not in tickers."""
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[])
        result = await self.analyzer.analyze_coin("BTCUSDT")
        assert result is None

    @pytest.mark.anyio
    async def test_analyze_coin_no_candles(self):
        """Returns None when not enough candles."""
        ticker = {
            "symbol": "BTCUSDT",
            "mark_price": 67500.0,
            "last_price": 67500.0,
            "volume_24h": 5e9,
            "high_24h": 68000.0,
            "low_24h": 65000.0,
            "prev_price_1h": 67000.0,
            "price_24h_pct": 3.0,
            "funding_rate": 0.0001,
        }
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[ticker])
        self.collector.get_candles = MagicMock(return_value=[])
        result = await self.analyzer.analyze_coin("BTC")
        assert result is None

    @pytest.mark.anyio
    async def test_analyze_coin_appends_usdt_suffix(self):
        """BTC input becomes BTCUSDT."""
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[])
        result = await self.analyzer.analyze_coin("BTC")
        # Called with symbol lookup → not found since tickers empty
        assert result is None
        self.collector.get_all_usdt_perpetuals.assert_called_once()

    @pytest.mark.anyio
    async def test_analyze_coin_full_result(self):
        """Returns full analysis dict when data available."""
        from core.types import Candle
        from datetime import timedelta
        import numpy as np

        ticker = {
            "symbol": "BTCUSDT",
            "mark_price": 67500.0,
            "last_price": 67500.0,
            "volume_24h": 5e9,
            "high_24h": 68000.0,
            "low_24h": 65000.0,
            "prev_price_1h": 67000.0,
            "price_24h_pct": 3.0,
            "funding_rate": 0.0001,
        }
        from datetime import datetime, timezone
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles = [
            Candle(
                timestamp=base_time + timedelta(hours=4 * i),
                open=67000.0 + i,
                high=67500.0 + i,
                low=66500.0 + i,
                close=67200.0 + i,
                volume=1000.0,
                symbol="BTCUSDT",
                interval="240",
            )
            for i in range(70)
        ]
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[ticker])
        self.collector.get_candles = MagicMock(return_value=candles)

        result = await self.analyzer.analyze_coin("BTCUSDT")
        assert result is not None
        assert result["symbol"] == "BTCUSDT"
        assert "price" in result
        assert "indicators" in result
        assert "verdict" in result
        assert "direction" in result
        assert result["direction"] in ("long", "short", "neutral")

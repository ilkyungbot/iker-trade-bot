"""Tests for feature engineering."""

import pandas as pd
import numpy as np
from data.features import (
    add_atr,
    add_atr_percent,
    add_adx,
    add_ema,
    add_ema_slope,
    add_donchian,
    add_rsi,
    add_bollinger,
    add_volume_sma,
    add_all_features,
    candles_to_dataframe,
    add_ema_crossover,
    add_rsi_signal,
    add_macd,
    add_sideways_filter,
    add_candle_patterns,
    add_volume_anomaly,
)
from datetime import datetime, timezone, timedelta
from core.types import Candle


def _make_df(n: int = 50) -> pd.DataFrame:
    """Create a sample OHLCV DataFrame with trending price."""
    np.random.seed(42)
    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.rand(n) * 1000 + 500,
    })


class TestATR:
    def test_atr_column_added(self):
        df = add_atr(_make_df())
        assert "atr" in df.columns

    def test_atr_positive(self):
        df = add_atr(_make_df())
        valid = df["atr"].dropna()
        assert (valid > 0).all()

    def test_atr_first_values_nan(self):
        df = add_atr(_make_df(), period=14)
        assert df["atr"].iloc[:13].isna().all()


class TestATRPercent:
    def test_atr_pct_column_added(self):
        df = add_atr_percent(_make_df())
        assert "atr_pct" in df.columns

    def test_atr_pct_reasonable_range(self):
        df = add_atr_percent(_make_df())
        valid = df["atr_pct"].dropna()
        assert (valid > 0).all()
        assert (valid < 1).all()


class TestADX:
    def test_adx_columns_added(self):
        df = add_adx(_make_df())
        assert "adx" in df.columns
        assert "plus_di" in df.columns
        assert "minus_di" in df.columns

    def test_adx_range(self):
        df = add_adx(_make_df())
        valid = df["adx"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestEMA:
    def test_ema_column_added(self):
        df = add_ema(_make_df(), period=20)
        assert "ema_20" in df.columns

    def test_ema_tracks_price(self):
        df = add_ema(_make_df(), period=5)
        diff = (df["ema_5"] - df["close"]).abs().dropna()
        assert diff.mean() < 5.0


class TestEMASlope:
    def test_slope_column_added(self):
        df = add_ema_slope(_make_df(), period=20)
        assert "ema_20_slope" in df.columns


class TestDonchian:
    def test_donchian_columns_added(self):
        df = add_donchian(_make_df())
        assert "donchian_high" in df.columns
        assert "donchian_low" in df.columns
        assert "donchian_mid" in df.columns

    def test_donchian_high_gte_close(self):
        df = add_donchian(_make_df())
        valid_idx = df["donchian_high"].dropna().index
        assert (df.loc[valid_idx, "donchian_high"] >= df.loc[valid_idx, "close"]).all()

    def test_donchian_low_lte_close(self):
        df = add_donchian(_make_df())
        valid_idx = df["donchian_low"].dropna().index
        assert (df.loc[valid_idx, "donchian_low"] <= df.loc[valid_idx, "close"]).all()


class TestRSI:
    def test_rsi_column_added(self):
        df = add_rsi(_make_df())
        assert "rsi" in df.columns

    def test_rsi_range(self):
        df = add_rsi(_make_df())
        valid = df["rsi"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestBollinger:
    def test_bollinger_columns_added(self):
        df = add_bollinger(_make_df())
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        assert "bb_mid" in df.columns
        assert "bb_width" in df.columns

    def test_upper_above_lower(self):
        df = add_bollinger(_make_df())
        valid_idx = df["bb_upper"].dropna().index
        assert (df.loc[valid_idx, "bb_upper"] > df.loc[valid_idx, "bb_lower"]).all()


class TestVolumeSMA:
    def test_volume_columns_added(self):
        df = add_volume_sma(_make_df())
        assert "volume_sma" in df.columns
        assert "volume_ratio" in df.columns


# --- Phase 1 신규 지표 테스트 ---


class TestEMACrossover:
    def test_columns_added(self):
        df = _make_df()
        df = add_ema(df, 20)
        df = add_ema(df, 50)
        df = add_ema_crossover(df, 20, 50)
        assert "ema_golden_cross" in df.columns
        assert "ema_death_cross" in df.columns

    def test_golden_cross_detected(self):
        n = 60
        closes = [100.0 - i * 0.5 for i in range(30)] + [85.0 + i * 1.0 for i in range(30)]
        df = pd.DataFrame({
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000] * n,
        })
        df = add_ema(df, 20)
        df = add_ema(df, 50)
        df = add_ema_crossover(df, 20, 50)
        assert df["ema_golden_cross"].any()

    def test_auto_adds_ema_if_missing(self):
        df = _make_df()
        df = add_ema_crossover(df, 20, 50)
        assert "ema_20" in df.columns
        assert "ema_50" in df.columns


class TestRSISignal:
    def test_columns_added(self):
        df = _make_df()
        df = add_rsi(df)
        df = add_rsi_signal(df)
        assert "rsi_signal" in df.columns
        assert "rsi_cross_up" in df.columns
        assert "rsi_cross_down" in df.columns

    def test_auto_adds_rsi_if_missing(self):
        df = _make_df()
        df = add_rsi_signal(df)
        assert "rsi" in df.columns


class TestMACD:
    def test_columns_added(self):
        df = _make_df()
        df = add_macd(df)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns
        assert "macd_hist_cross_up" in df.columns
        assert "macd_hist_cross_down" in df.columns

    def test_histogram_sign_change(self):
        df = _make_df(100)
        df = add_macd(df)
        assert df["macd_hist_cross_up"].any() or df["macd_hist_cross_down"].any()


class TestSidewaysFilter:
    def test_columns_added(self):
        df = _make_df()
        df = add_adx(df)
        df = add_atr(df)
        df = add_sideways_filter(df)
        assert "is_sideways" in df.columns

    def test_flat_market_detected(self):
        n = 60
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.1] * n,
            "low": [99.9] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        })
        df = add_adx(df)
        df = add_atr(df)
        df = add_sideways_filter(df)
        assert "is_sideways" in df.columns


class TestCandlePatterns:
    def test_columns_added(self):
        df = _make_df()
        df = add_candle_patterns(df)
        expected = [
            "candle_hammer", "candle_inverted_hammer",
            "candle_bullish_engulfing", "candle_bearish_engulfing",
            "candle_doji", "candle_morning_star",
        ]
        for col in expected:
            assert col in df.columns

    def test_doji_detection(self):
        n = 5
        df = pd.DataFrame({
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [102.0, 102.0, 102.0, 102.0, 102.0],
            "low": [98.0, 98.0, 98.0, 98.0, 98.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1000] * n,
        })
        df = add_candle_patterns(df)
        assert df["candle_doji"].all()


class TestVolumeAnomaly:
    def test_columns_added(self):
        df = _make_df()
        df = add_volume_sma(df)
        df = add_volume_anomaly(df)
        assert "volume_anomaly" in df.columns

    def test_spike_detected(self):
        n = 30
        volumes = [1000.0] * 29 + [3000.0]
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": volumes,
        })
        df = add_volume_sma(df)
        df = add_volume_anomaly(df, threshold=2.0)
        assert df["volume_anomaly"].iloc[-1] == True


class TestAddAllFeatures:
    def test_all_features_present(self):
        df = add_all_features(_make_df())
        expected = [
            "atr", "atr_pct", "adx", "plus_di", "minus_di",
            "ema_20", "ema_50", "ema_20_slope", "ema_50_slope",
            "donchian_high", "donchian_low", "donchian_mid",
            "rsi", "bb_upper", "bb_lower", "bb_mid", "bb_width",
            "volume_sma", "volume_ratio",
            # Phase 1 additions
            "ema_golden_cross", "ema_death_cross",
            "rsi_signal", "rsi_cross_up", "rsi_cross_down",
            "macd", "macd_signal", "macd_hist",
            "is_sideways",
            "candle_hammer", "candle_doji",
            "volume_anomaly",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self):
        df = add_all_features(_make_df())
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns


class TestCandlesToDataframe:
    def test_converts_candles(self):
        candles = [
            Candle(datetime(2024, 1, 1, i, tzinfo=timezone.utc), 100, 105, 95, 102, 1000, "BTC", "60")
            for i in range(5)
        ]
        df = candles_to_dataframe(candles)
        assert len(df) == 5
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "symbol", "interval"]

    def test_empty_list(self):
        df = candles_to_dataframe([])
        assert len(df) == 0

    def test_sorted_by_timestamp(self):
        candles = [
            Candle(datetime(2024, 1, 1, 2, tzinfo=timezone.utc), 100, 105, 95, 102, 1000, "BTC", "60"),
            Candle(datetime(2024, 1, 1, 0, tzinfo=timezone.utc), 100, 105, 95, 102, 1000, "BTC", "60"),
            Candle(datetime(2024, 1, 1, 1, tzinfo=timezone.utc), 100, 105, 95, 102, 1000, "BTC", "60"),
        ]
        df = candles_to_dataframe(candles)
        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

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
        assert (valid < 1).all()  # should be well under 100%


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
        # EMA should be close to price
        diff = (df["ema_5"] - df["close"]).abs().dropna()
        assert diff.mean() < 5.0  # within a few dollars


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


class TestAddAllFeatures:
    def test_all_features_present(self):
        df = add_all_features(_make_df())
        expected = [
            "atr", "atr_pct", "adx", "plus_di", "minus_di",
            "ema_20", "ema_50", "ema_20_slope", "ema_50_slope",
            "donchian_high", "donchian_low", "donchian_mid",
            "rsi", "bb_upper", "bb_lower", "bb_mid", "bb_width",
            "volume_sma", "volume_ratio",
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

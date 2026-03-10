"""
Layer 1: Feature engineering.

Compute technical indicators from OHLCV data.
All functions are pure: DataFrame in, DataFrame out.
No side effects, no API calls, no state.
"""

import pandas as pd
import numpy as np


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range (ATR) column."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df = df.copy()
    df["atr"] = tr.rolling(window=period).mean()
    return df


def add_atr_percent(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ATR as percentage of price."""
    if "atr" not in df.columns:
        df = add_atr(df, period)
    df = df.copy()
    df["atr_pct"] = df["atr"] / df["close"]
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average Directional Index (ADX)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed averages
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(window=period).mean()

    df = df.copy()
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def add_ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """Add Exponential Moving Average."""
    df = df.copy()
    df[f"ema_{period}"] = df[column].ewm(span=period, adjust=False).mean()
    return df


def add_ema_slope(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add EMA slope (direction indicator)."""
    col = f"ema_{period}"
    if col not in df.columns:
        df = add_ema(df, period)
    df = df.copy()
    df[f"ema_{period}_slope"] = df[col] - df[col].shift(1)
    return df


def add_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Donchian Channel (highest high / lowest low over period)."""
    df = df.copy()
    df["donchian_high"] = df["high"].rolling(window=period).max()
    df["donchian_low"] = df["low"].rolling(window=period).min()
    df["donchian_mid"] = (df["donchian_high"] + df["donchian_low"]) / 2
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df = df.copy()
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands."""
    df = df.copy()
    sma = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()

    df["bb_upper"] = sma + std_dev * std
    df["bb_lower"] = sma - std_dev * std
    df["bb_mid"] = sma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    return df


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add volume simple moving average and relative volume."""
    df = df.copy()
    df["volume_sma"] = df["volume"].rolling(window=period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all standard features to a candle DataFrame."""
    df = add_atr(df, 14)
    df = add_atr_percent(df, 14)
    df = add_adx(df, 14)
    df = add_ema(df, 20)
    df = add_ema(df, 50)
    df = add_ema_slope(df, 20)
    df = add_ema_slope(df, 50)
    df = add_donchian(df, 20)
    df = add_rsi(df, 14)
    df = add_bollinger(df, 20)
    df = add_volume_sma(df, 20)
    return df


def candles_to_dataframe(candles: list) -> pd.DataFrame:
    """Convert list of Candle objects to pandas DataFrame."""
    if not candles:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol", "interval"])

    data = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "symbol": c.symbol,
            "interval": c.interval,
        }
        for c in candles
    ]
    df = pd.DataFrame(data)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

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


# --- Phase 1 신규 지표 ---


def add_ema_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """EMA 크로스오버 감지. 골든크로스/데드크로스 bool 컬럼 추가."""
    fast_col = f"ema_{fast}"
    slow_col = f"ema_{slow}"

    if fast_col not in df.columns:
        df = add_ema(df, fast)
    if slow_col not in df.columns:
        df = add_ema(df, slow)

    df = df.copy()
    fast_ema = df[fast_col]
    slow_ema = df[slow_col]

    # 이전 봉에서 fast < slow 이고 현재 봉에서 fast > slow → 골든크로스
    prev_below = fast_ema.shift(1) < slow_ema.shift(1)
    prev_above = fast_ema.shift(1) > slow_ema.shift(1)
    curr_above = fast_ema > slow_ema
    curr_below = fast_ema < slow_ema

    df["ema_golden_cross"] = prev_below & curr_above
    df["ema_death_cross"] = prev_above & curr_below
    return df


def add_rsi_signal(df: pd.DataFrame, rsi_period: int = 14, signal_period: int = 9) -> pd.DataFrame:
    """RSI의 시그널선(EMA) 교차 감지."""
    if "rsi" not in df.columns:
        df = add_rsi(df, rsi_period)

    df = df.copy()
    df["rsi_signal"] = df["rsi"].ewm(span=signal_period, adjust=False).mean()

    prev_below = df["rsi"].shift(1) < df["rsi_signal"].shift(1)
    prev_above = df["rsi"].shift(1) > df["rsi_signal"].shift(1)
    curr_above = df["rsi"] > df["rsi_signal"]
    curr_below = df["rsi"] < df["rsi_signal"]

    df["rsi_cross_up"] = prev_below & curr_above
    df["rsi_cross_down"] = prev_above & curr_below
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 히스토그램. 모멘텀 전환 감지."""
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 히스토그램 부호 전환
    prev_hist = df["macd_hist"].shift(1)
    df["macd_hist_cross_up"] = (prev_hist < 0) & (df["macd_hist"] >= 0)
    df["macd_hist_cross_down"] = (prev_hist > 0) & (df["macd_hist"] <= 0)
    return df


def add_sideways_filter(df: pd.DataFrame, adx_threshold: int = 15, atr_squeeze_period: int = 20) -> pd.DataFrame:
    """횡보장 필터. ADX < 15 AND ATR이 하위 25%면 횡보."""
    if "adx" not in df.columns:
        df = add_adx(df)
    if "atr" not in df.columns:
        df = add_atr(df)

    df = df.copy()
    atr_q25 = df["atr"].rolling(window=atr_squeeze_period).quantile(0.25)
    df["is_sideways"] = (df["adx"] < adx_threshold) & (df["atr"] <= atr_q25)
    return df


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """핵심 캔들 패턴 감지 (순수 pandas, TA-Lib 불필요)."""
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    total_range = h - l
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l

    # 망치형 (Hammer): 하단 긴 꼬리, 짧은 몸통, 하락추세 후
    df["candle_hammer"] = (
        (lower_shadow >= body * 2)
        & (upper_shadow <= body * 0.5)
        & (total_range > 0)
        & (c.shift(1) < o.shift(1))  # 이전 봉 하락
    )

    # 역망치 (Inverted Hammer / Shooting Star)
    df["candle_inverted_hammer"] = (
        (upper_shadow >= body * 2)
        & (lower_shadow <= body * 0.5)
        & (total_range > 0)
        & (c.shift(1) > o.shift(1))  # 이전 봉 상승
    )

    # 상승 장악형 (Bullish Engulfing)
    prev_bearish = c.shift(1) < o.shift(1)
    curr_bullish = c > o
    df["candle_bullish_engulfing"] = (
        prev_bearish
        & curr_bullish
        & (c >= o.shift(1))
        & (o <= c.shift(1))
    )

    # 하락 장악형 (Bearish Engulfing)
    prev_bullish = c.shift(1) > o.shift(1)
    curr_bearish = c < o
    df["candle_bearish_engulfing"] = (
        prev_bullish
        & curr_bearish
        & (o >= c.shift(1))
        & (c <= o.shift(1))
    )

    # 도지 (Doji): 몸통이 전체 범위의 5% 미만
    df["candle_doji"] = (body <= total_range * 0.05) & (total_range > 0)

    # 모닝스타 간이판 (3봉: 큰하락 → 작은몸통 → 큰상승)
    big_bearish = (o.shift(2) - c.shift(2)) > body.shift(2).rolling(5).mean()
    small_body = body.shift(1) < body.rolling(5).mean() * 0.5
    big_bullish = (c - o) > body.rolling(5).mean()
    df["candle_morning_star"] = big_bearish & small_body & big_bullish

    return df


def add_volume_anomaly(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """거래량 이상치 감지. 평균 대비 threshold배 이상이면 True."""
    if "volume_sma" not in df.columns:
        df = add_volume_sma(df)

    df = df.copy()
    df["volume_anomaly"] = df["volume_ratio"] >= threshold
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
    # Phase 1 신규
    df = add_ema_crossover(df, 20, 50)
    df = add_rsi_signal(df, 14, 9)
    df = add_macd(df)
    df = add_sideways_filter(df)
    df = add_candle_patterns(df)
    df = add_volume_anomaly(df)
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

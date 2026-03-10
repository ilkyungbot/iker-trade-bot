"""
Layer 3: Strategy A — Trend Following.

Logic: Donchian Channel breakout + ADX > 25 + 4H EMA direction alignment.
Stop-loss: ATR × 1.5
Take-profit: Trailing stop starting at ATR × 2.0

Known weakness: consecutive stop-outs in ranging markets. This is accepted —
the edge comes from capturing large trends that more than compensate.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from core.types import Signal, SignalAction, StrategyName
from strategy.base import Strategy

logger = logging.getLogger(__name__)

# Strategy parameters (kept minimal to reduce overfitting risk)
DONCHIAN_PERIOD = 20
ADX_THRESHOLD = 25
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0
EMA_PERIOD_4H = 50


class TrendFollowingStrategy(Strategy):
    """Donchian Channel breakout with ADX and EMA confirmation."""

    @property
    def name(self) -> str:
        return StrategyName.TREND_FOLLOWING.value

    def generate_signal(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        symbol: str,
        current_position_side: str | None = None,
    ) -> Signal | None:
        """Generate trend following signal."""
        if len(df_1h) < DONCHIAN_PERIOD + 5 or len(df_4h) < EMA_PERIOD_4H + 5:
            return None

        # Required columns check
        required_1h = ["close", "high", "low", "donchian_high", "donchian_low", "adx", "atr"]
        required_4h = ["close", f"ema_{EMA_PERIOD_4H}"]
        for col in required_1h:
            if col not in df_1h.columns:
                logger.warning(f"Missing column {col} in 1H data for {symbol}")
                return None
        for col in required_4h:
            if col not in df_4h.columns:
                logger.warning(f"Missing column {col} in 4H data for {symbol}")
                return None

        # Current values (latest completed candle = second to last, since last may be forming)
        # Use iloc[-1] assuming the caller passes completed candles only
        current = df_1h.iloc[-1]
        prev = df_1h.iloc[-2]

        close = current["close"]
        adx = current["adx"]
        atr = current["atr"]
        donchian_high = prev["donchian_high"]  # use previous candle's channel
        donchian_low = prev["donchian_low"]

        # 4H trend direction
        ema_4h = df_4h[f"ema_{EMA_PERIOD_4H}"].iloc[-1]
        ema_4h_prev = df_4h[f"ema_{EMA_PERIOD_4H}"].iloc[-2]
        ema_trending_up = ema_4h > ema_4h_prev
        ema_trending_down = ema_4h < ema_4h_prev

        # Skip if data is invalid
        if pd.isna(close) or pd.isna(adx) or pd.isna(atr) or pd.isna(donchian_high) or pd.isna(donchian_low):
            return None
        if pd.isna(ema_4h) or pd.isna(ema_4h_prev):
            return None
        if atr <= 0:
            return None

        timestamp = current.get("timestamp", datetime.now(timezone.utc))
        if not isinstance(timestamp, datetime):
            timestamp = datetime.now(timezone.utc)

        # --- Exit logic ---
        if current_position_side == "long":
            # Exit long if price breaks below Donchian low
            if close < donchian_low:
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    action=SignalAction.EXIT,
                    strategy=StrategyName.TREND_FOLLOWING,
                    entry_price=close,
                    stop_loss=0.0,
                    take_profit=0.0,
                    confidence=0.7,
                    metadata={"reason": "donchian_low_break"},
                )

        elif current_position_side == "short":
            if close > donchian_high:
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    action=SignalAction.EXIT,
                    strategy=StrategyName.TREND_FOLLOWING,
                    entry_price=close,
                    stop_loss=0.0,
                    take_profit=0.0,
                    confidence=0.7,
                    metadata={"reason": "donchian_high_break"},
                )

        # --- Entry logic (only if no position) ---
        if current_position_side is not None:
            return None

        # ADX must confirm trend exists
        if adx < ADX_THRESHOLD:
            return None

        # LONG: price breaks above Donchian high + 4H EMA trending up
        if close > donchian_high and ema_trending_up:
            stop_loss = close - (atr * ATR_SL_MULTIPLIER)
            take_profit = close + (atr * ATR_TP_MULTIPLIER)

            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                action=SignalAction.ENTER_LONG,
                strategy=StrategyName.TREND_FOLLOWING,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=min(adx / 50.0, 1.0),  # higher ADX = higher confidence
                metadata={
                    "adx": round(adx, 2),
                    "atr": round(atr, 4),
                    "donchian_high": round(donchian_high, 4),
                    "ema_4h": round(ema_4h, 4),
                },
            )

        # SHORT: price breaks below Donchian low + 4H EMA trending down
        if close < donchian_low and ema_trending_down:
            stop_loss = close + (atr * ATR_SL_MULTIPLIER)
            take_profit = close - (atr * ATR_TP_MULTIPLIER)

            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                action=SignalAction.ENTER_SHORT,
                strategy=StrategyName.TREND_FOLLOWING,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=min(adx / 50.0, 1.0),
                metadata={
                    "adx": round(adx, 2),
                    "atr": round(atr, 4),
                    "donchian_low": round(donchian_low, 4),
                    "ema_4h": round(ema_4h, 4),
                },
            )

        return None

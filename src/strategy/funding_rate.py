"""
Layer 3: Strategy B — Funding Rate Reversal.

Logic: When 8H funding rate exceeds ±0.1%, take opposite position
but ONLY when price confirms reversal (not blindly counter-trend).

Stop-loss: ATR × 1.0 (tight)
Take-profit: Funding rate normalization or ATR × 1.5
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from core.types import Signal, SignalAction, StrategyName, FundingRate
from strategy.base import Strategy

logger = logging.getLogger(__name__)

# Funding rate thresholds
FUNDING_EXTREME_THRESHOLD = 0.001  # 0.1% per 8h
FUNDING_NORMAL_THRESHOLD = 0.0005  # 0.05% — considered "normalized"

# ATR multipliers
ATR_SL_MULTIPLIER = 1.0
ATR_TP_MULTIPLIER = 1.5

# Price confirmation: RSI must show reversal tendency
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30


class FundingRateStrategy(Strategy):
    """Counter-trade extreme funding rates with price confirmation."""

    @property
    def name(self) -> str:
        return StrategyName.FUNDING_RATE.value

    def generate_signal(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        symbol: str,
        current_position_side: str | None = None,
        latest_funding_rate: float | None = None,
    ) -> Signal | None:
        """
        Generate funding rate reversal signal.

        Args:
            df_1h: 1H candle data with features
            df_4h: 4H candle data (used for context, not primary)
            symbol: trading pair
            current_position_side: existing position direction
            latest_funding_rate: most recent funding rate value
        """
        if latest_funding_rate is None:
            return None

        if len(df_1h) < 20:
            return None

        required = ["close", "atr", "rsi"]
        for col in required:
            if col not in df_1h.columns:
                return None

        current = df_1h.iloc[-1]
        close = current["close"]
        atr = current["atr"]
        rsi = current["rsi"]

        if pd.isna(atr) or pd.isna(rsi) or atr <= 0:
            return None

        timestamp = current.get("timestamp", datetime.now(timezone.utc))
        if not isinstance(timestamp, datetime):
            timestamp = datetime.now(timezone.utc)

        # --- Exit logic ---
        if current_position_side is not None:
            # Exit when funding normalizes
            if abs(latest_funding_rate) < FUNDING_NORMAL_THRESHOLD:
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    action=SignalAction.EXIT,
                    strategy=StrategyName.FUNDING_RATE,
                    entry_price=close,
                    stop_loss=0.0,
                    take_profit=0.0,
                    confidence=0.6,
                    metadata={"reason": "funding_normalized", "funding_rate": latest_funding_rate},
                )
            return None

        # --- Entry logic ---

        # Funding rate must be extreme
        if abs(latest_funding_rate) < FUNDING_EXTREME_THRESHOLD:
            return None

        if latest_funding_rate > FUNDING_EXTREME_THRESHOLD:
            # Funding is very positive → longs are paying shorts
            # → Market is overleveraged long → short opportunity
            # BUT only if price shows weakness (RSI overbought or declining)
            if rsi < RSI_OVERBOUGHT:
                return None  # price not confirming reversal yet

            stop_loss = close + (atr * ATR_SL_MULTIPLIER)
            take_profit = close - (atr * ATR_TP_MULTIPLIER)

            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                action=SignalAction.ENTER_SHORT,
                strategy=StrategyName.FUNDING_RATE,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=min(abs(latest_funding_rate) / 0.003, 1.0),
                metadata={
                    "funding_rate": latest_funding_rate,
                    "rsi": round(rsi, 2),
                    "atr": round(atr, 4),
                },
            )

        elif latest_funding_rate < -FUNDING_EXTREME_THRESHOLD:
            # Funding is very negative → shorts are paying longs
            # → Market is overleveraged short → long opportunity
            if rsi > RSI_OVERSOLD:
                return None  # price not confirming reversal yet

            stop_loss = close - (atr * ATR_SL_MULTIPLIER)
            take_profit = close + (atr * ATR_TP_MULTIPLIER)

            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                action=SignalAction.ENTER_LONG,
                strategy=StrategyName.FUNDING_RATE,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=min(abs(latest_funding_rate) / 0.003, 1.0),
                metadata={
                    "funding_rate": latest_funding_rate,
                    "rsi": round(rsi, 2),
                    "atr": round(atr, 4),
                },
            )

        return None

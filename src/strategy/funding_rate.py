"""
Layer 3: Strategy B — Funding Rate Reversal.

펀딩레이트 0.07% 초과 시 역전 포지션 + RSI 확인.
임계값 완화: 0.1% → 0.07%, RSI 70/30 → 65/35.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from core.types import Signal, SignalAction, SignalMessage, SignalQuality, StrategyName
from strategy.base import Strategy

logger = logging.getLogger(__name__)

# 완화된 임계값
FUNDING_EXTREME_THRESHOLD = 0.0007  # 0.07% per 8h
FUNDING_NORMAL_THRESHOLD = 0.0005   # 0.05% — considered "normalized"

ATR_SL_MULTIPLIER = 1.0
ATR_TP_MULTIPLIER = 1.5

RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35


class FundingRateStrategy(Strategy):
    """Counter-trade extreme funding rates with price confirmation."""

    @property
    def name(self) -> str:
        return StrategyName.FUNDING_RATE.value

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        *,
        latest_funding_rate: float | None = None,
    ) -> SignalMessage | None:
        if latest_funding_rate is None:
            return None

        if len(df) < 20:
            return None

        required = ["close", "atr", "rsi"]
        for col in required:
            if col not in df.columns:
                return None

        current = df.iloc[-1]
        close = current["close"]
        atr = current["atr"]
        rsi = current["rsi"]

        if pd.isna(close) or pd.isna(atr) or pd.isna(rsi) or atr <= 0:
            return None

        timestamp = current.get("timestamp", datetime.now(timezone.utc))
        if not isinstance(timestamp, datetime):
            timestamp = datetime.now(timezone.utc)

        # 횡보장이면 스킵
        if current.get("is_sideways", False):
            return None

        # 펀딩레이트가 극단적이지 않으면 스킵
        if abs(latest_funding_rate) < FUNDING_EXTREME_THRESHOLD:
            return None

        explanation: list[str] = []
        indicators = {
            "funding_rate": round(latest_funding_rate * 100, 4),
            "rsi": round(float(rsi), 1),
            "atr": round(float(atr), 4),
        }

        if latest_funding_rate >= FUNDING_EXTREME_THRESHOLD:
            # 양의 펀딩 → 롱 과열 → 숏 기회
            if rsi < RSI_OVERBOUGHT:
                return None  # RSI 확인 안 됨
            explanation.append(f"펀딩레이트 +{latest_funding_rate*100:.3f}% (롱 과열)")
            explanation.append(f"RSI {rsi:.0f} > {RSI_OVERBOUGHT} (과매수 확인)")

            sl = close + (atr * ATR_SL_MULTIPLIER)
            tp = close - (atr * ATR_TP_MULTIPLIER)
            action = SignalAction.ENTER_SHORT

        elif latest_funding_rate <= -FUNDING_EXTREME_THRESHOLD:
            # 음의 펀딩 → 숏 과열 → 롱 기회
            if rsi > RSI_OVERSOLD:
                return None
            explanation.append(f"펀딩레이트 {latest_funding_rate*100:.3f}% (숏 과열)")
            explanation.append(f"RSI {rsi:.0f} < {RSI_OVERSOLD} (과매도 확인)")

            sl = close - (atr * ATR_SL_MULTIPLIER)
            tp = close + (atr * ATR_TP_MULTIPLIER)
            action = SignalAction.ENTER_LONG
        else:
            return None

        confidence = min(abs(latest_funding_rate) / 0.003, 1.0)
        sl_dist = abs(close - sl)
        tp_dist = abs(tp - close)
        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0

        signal = Signal(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            strategy=StrategyName.FUNDING_RATE,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            metadata={"funding_rate": latest_funding_rate},
        )

        quality = SignalQuality.STRONG if abs(latest_funding_rate) >= 0.001 else SignalQuality.MODERATE

        return SignalMessage(
            signal=signal,
            quality=quality,
            explanation=explanation,
            indicators=indicators,
            risk_reward_ratio=round(rr_ratio, 1),
        )

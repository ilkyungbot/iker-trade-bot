"""
Layer 3: Strategy A — Multi-indicator Trend Signal.

4H 봉 기반 복합 시그널 스코어링.
7개 지표 점수 합산 → SignalQuality 결정.
횡보장 필터 적용.
"""

import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from core.types import Signal, SignalAction, SignalMessage, SignalQuality, StrategyName
from strategy.base import Strategy

logger = logging.getLogger(__name__)

ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0


class TrendFollowingStrategy(Strategy):
    """Multi-indicator trend signal with scoring."""

    @property
    def name(self) -> str:
        return StrategyName.TREND_FOLLOWING.value

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> SignalMessage | None:
        """4H 봉 기반 복합 시그널 생성."""
        if len(df) < 55:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # 필수 컬럼 체크
        required = ["close", "atr", "adx", "rsi", "is_sideways"]
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column {col} for {symbol}")
                return None

        close = current["close"]
        atr = current["atr"]
        adx = current["adx"]

        if pd.isna(close) or pd.isna(atr) or atr <= 0 or pd.isna(adx):
            return None

        # 횡보장 필터: 횡보이면 시그널 자체를 안 냄
        if current.get("is_sideways", False):
            return None

        timestamp = current.get("timestamp", datetime.now(timezone.utc))
        if not isinstance(timestamp, datetime):
            timestamp = datetime.now(timezone.utc)

        # --- 롱/숏 스코어링 ---
        long_score = 0
        short_score = 0
        long_reasons: list[str] = []
        short_reasons: list[str] = []
        indicators: dict = {}

        # 1. EMA 크로스오버
        if current.get("ema_golden_cross", False):
            long_score += 1
            long_reasons.append("EMA 골든크로스 (20>50 교차)")
        if current.get("ema_death_cross", False):
            short_score += 1
            short_reasons.append("EMA 데드크로스 (20<50 교차)")

        # 2. RSI 시그널 크로스
        rsi_val = current.get("rsi", 50)
        rsi_signal_val = current.get("rsi_signal", 50)
        if not pd.isna(rsi_val):
            indicators["rsi"] = round(float(rsi_val), 1)
        if current.get("rsi_cross_up", False):
            long_score += 1
            long_reasons.append(f"RSI 시그널선 상향돌파 ({indicators.get('rsi', '?')})")
        if current.get("rsi_cross_down", False):
            short_score += 1
            short_reasons.append(f"RSI 시그널선 하향돌파 ({indicators.get('rsi', '?')})")

        # 3. 볼린저밴드
        bb_lower = current.get("bb_lower", np.nan)
        bb_upper = current.get("bb_upper", np.nan)
        prev_low = prev.get("low", np.nan)
        prev_close = prev.get("close", np.nan)
        if not pd.isna(bb_lower) and not pd.isna(prev_low):
            # 이전 봉이 하단 터치 + 현재 봉 반등
            if prev_low <= bb_lower and close > prev_close:
                long_score += 1
                long_reasons.append("볼린저 하단 터치 후 반등")
        if not pd.isna(bb_upper):
            prev_high = prev.get("high", np.nan)
            if not pd.isna(prev_high) and prev_high >= bb_upper and close < prev_close:
                short_score += 1
                short_reasons.append("볼린저 상단 터치 후 하락")

        # 4. MACD 히스토그램 전환
        if current.get("macd_hist_cross_up", False):
            long_score += 1
            long_reasons.append("MACD 히스토그램 양전환 (모멘텀 상승)")
        if current.get("macd_hist_cross_down", False):
            short_score += 1
            short_reasons.append("MACD 히스토그램 음전환 (모멘텀 하락)")

        # 5. 거래량 이상치
        vol_anomaly = current.get("volume_anomaly", False)
        vol_ratio = current.get("volume_ratio", 1.0)
        if vol_anomaly and not pd.isna(vol_ratio):
            indicators["volume_ratio"] = round(float(vol_ratio), 1)
            if close > current.get("open", close):
                long_score += 1
                long_reasons.append(f"거래량 평균 대비 {indicators['volume_ratio']}배 급증 (상승 캔들)")
            elif close < current.get("open", close):
                short_score += 1
                short_reasons.append(f"거래량 평균 대비 {indicators['volume_ratio']}배 급증 (하락 캔들)")

        # 6. 캔들 패턴
        if current.get("candle_hammer", False) or current.get("candle_bullish_engulfing", False) or current.get("candle_morning_star", False):
            long_score += 1
            patterns = []
            if current.get("candle_hammer", False):
                patterns.append("망치형")
            if current.get("candle_bullish_engulfing", False):
                patterns.append("상승장악형")
            if current.get("candle_morning_star", False):
                patterns.append("모닝스타")
            long_reasons.append(f"캔들 패턴: {', '.join(patterns)}")

        if current.get("candle_inverted_hammer", False) or current.get("candle_bearish_engulfing", False):
            short_score += 1
            patterns = []
            if current.get("candle_inverted_hammer", False):
                patterns.append("역망치형")
            if current.get("candle_bearish_engulfing", False):
                patterns.append("하락장악형")
            short_reasons.append(f"캔들 패턴: {', '.join(patterns)}")

        # 7. ADX > 20 (추세 존재)
        if not pd.isna(adx) and adx > 20:
            long_score += 1
            short_score += 1
            long_reasons.append(f"ADX {adx:.0f} > 20 (추세 확인)")
            short_reasons.append(f"ADX {adx:.0f} > 20 (추세 확인)")
            indicators["adx"] = round(float(adx), 1)

        # --- 시그널 결정 ---
        # 롱과 숏 모두 점수가 있으면 높은 쪽 선택
        if long_score >= short_score and long_score >= 2:
            score = long_score
            reasons = long_reasons
            action = SignalAction.ENTER_LONG
            sl = close - (atr * ATR_SL_MULTIPLIER)
            tp = close + (atr * ATR_TP_MULTIPLIER)
        elif short_score > long_score and short_score >= 2:
            score = short_score
            reasons = short_reasons
            action = SignalAction.ENTER_SHORT
            sl = close + (atr * ATR_SL_MULTIPLIER)
            tp = close - (atr * ATR_TP_MULTIPLIER)
        else:
            return None

        quality = SignalQuality.STRONG if score >= 3 else SignalQuality.MODERATE
        confidence = min(score / 7.0, 1.0)

        sl_dist = abs(close - sl)
        tp_dist = abs(tp - close)
        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0

        indicators["atr"] = round(float(atr), 4)
        indicators["close"] = round(float(close), 2)

        signal = Signal(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            metadata={"score": score, "indicators": indicators},
        )

        return SignalMessage(
            signal=signal,
            quality=quality,
            explanation=reasons,
            indicators=indicators,
            risk_reward_ratio=round(rr_ratio, 1),
        )

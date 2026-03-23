"""CoinAnalyzer — 코인 스코어링 및 심층 분석 서비스."""

import asyncio
import functools
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from data.features import add_all_features, candles_to_dataframe

logger = logging.getLogger(__name__)


def pandas_isna(val) -> bool:
    """Null check that handles both pandas NA and numpy NaN."""
    if val is None:
        return True
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


class CoinAnalyzer:
    """코인 스코어링 및 심층 분석."""

    def __init__(self, collector, config):
        self.collector = collector
        self.config = config

    async def _run_blocking(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs),
        )

    def score_pair(self, df, symbol: str) -> dict | None:
        """단일 페어의 롱/숏 스코어를 점수만 추출 (시그널 미발생도 포함)."""
        if len(df) < 55:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        close = current.get("close")
        atr = current.get("atr")
        adx = current.get("adx")

        if close is None or atr is None or pandas_isna(close) or pandas_isna(atr) or atr <= 0:
            return None

        long_score = 0
        short_score = 0
        long_reasons: list[str] = []
        short_reasons: list[str] = []

        # 1. EMA 크로스오버
        if current.get("ema_golden_cross", False):
            long_score += 1
            long_reasons.append("EMA 골든크로스")
        if current.get("ema_death_cross", False):
            short_score += 1
            short_reasons.append("EMA 데드크로스")

        # 2. RSI 시그널
        if current.get("rsi_cross_up", False):
            long_score += 1
            long_reasons.append(f"RSI 상향돌파 ({current.get('rsi', 0):.0f})")
        if current.get("rsi_cross_down", False):
            short_score += 1
            short_reasons.append(f"RSI 하향돌파 ({current.get('rsi', 0):.0f})")

        # 3. 볼린저밴드
        bb_lower = current.get("bb_lower", np.nan)
        bb_upper = current.get("bb_upper", np.nan)
        prev_close = prev.get("close", np.nan)
        if not pandas_isna(bb_lower) and not pandas_isna(prev.get("low", np.nan)):
            if prev.get("low", np.nan) <= bb_lower and close > prev_close:
                long_score += 1
                long_reasons.append("볼린저 하단 반등")
        if not pandas_isna(bb_upper) and not pandas_isna(prev.get("high", np.nan)):
            if prev.get("high", np.nan) >= bb_upper and close < prev_close:
                short_score += 1
                short_reasons.append("볼린저 상단 하락")

        # 4. MACD
        if current.get("macd_hist_cross_up", False):
            long_score += 1
            long_reasons.append("MACD 양전환")
        if current.get("macd_hist_cross_down", False):
            short_score += 1
            short_reasons.append("MACD 음전환")

        # 5. 거래량 이상치
        if current.get("volume_anomaly", False):
            if close > current.get("open", close):
                long_score += 1
                long_reasons.append(f"거래량 급증 (상승)")
            elif close < current.get("open", close):
                short_score += 1
                short_reasons.append(f"거래량 급증 (하락)")

        # 6. 캔들 패턴
        if current.get("candle_hammer", False) or current.get("candle_bullish_engulfing", False) or current.get("candle_morning_star", False):
            long_score += 1
            long_reasons.append("강세 캔들패턴")
        if current.get("candle_inverted_hammer", False) or current.get("candle_bearish_engulfing", False):
            short_score += 1
            short_reasons.append("약세 캔들패턴")

        # 7. ADX
        if not pandas_isna(adx) and adx > 20:
            long_score += 1
            short_score += 1
            long_reasons.append(f"ADX {adx:.0f}")
            short_reasons.append(f"ADX {adx:.0f}")

        max_score = max(long_score, short_score)
        if max_score < 1:
            return None

        if long_score >= short_score:
            direction = "long"
            score = long_score
            reasons = long_reasons
        else:
            direction = "short"
            score = short_score
            reasons = short_reasons

        quality = "strong" if score >= 3 else "moderate" if score >= 2 else "weak"

        return {
            "symbol": symbol,
            "direction": direction,
            "score": score,
            "quality": quality,
            "reasons": reasons,
        }

    async def analyze_coin(self, query: str) -> dict | None:
        """단일 코인 심층 분석. 티커 심볼(예: SOL, BTC) 입력."""
        query = query.strip().upper()
        # USDT 접미사 붙이기
        symbol = query if query.endswith("USDT") else query + "USDT"

        now = datetime.now(timezone.utc)

        # 1. 티커 정보
        tickers = await self._run_blocking(self.collector.get_all_usdt_perpetuals)
        ticker = None
        for t in tickers:
            if t["symbol"] == symbol:
                ticker = t
                break
        if not ticker:
            return None

        # 2. 4H 캔들 + 지표 (55개면 충분 → 60일)
        candles = await self._run_blocking(
            self.collector.get_candles,
            symbol, self.config.signal.primary_interval,
            start_time=now - timedelta(days=60),
        )
        if not candles or len(candles) < 55:
            return None

        df = candles_to_dataframe(candles)
        df = add_all_features(df)
        current = df.iloc[-1]

        # 3. 스코어링 (롱/숏 모두)
        score_result = self.score_pair(df, symbol)

        # 4. 펀딩비
        funding_rate = ticker.get("funding_rate", 0)

        # 5. 개별 지표 상세 (ticker는 이미 최신 — 중복 호출 제거)
        prev_1h = ticker.get("prev_price_1h", 0)
        last = ticker["mark_price"]  # mark_price가 거래소 UI와 동일
        change_1h = (last - prev_1h) / prev_1h * 100 if prev_1h > 0 else 0

        indicators = {
            "rsi": round(float(current.get("rsi", 0)), 1) if not pandas_isna(current.get("rsi")) else None,
            "adx": round(float(current.get("adx", 0)), 1) if not pandas_isna(current.get("adx")) else None,
            "macd_hist": round(float(current.get("macd_hist", 0)), 4) if not pandas_isna(current.get("macd_hist")) else None,
            "bb_width": round(float(current.get("bb_width", 0)), 4) if not pandas_isna(current.get("bb_width")) else None,
            "ema_20": round(float(current.get("ema_20", 0)), 2) if not pandas_isna(current.get("ema_20")) else None,
            "ema_50": round(float(current.get("ema_50", 0)), 2) if not pandas_isna(current.get("ema_50")) else None,
            "atr": round(float(current.get("atr", 0)), 4) if not pandas_isna(current.get("atr")) else None,
            "volume_ratio": round(float(current.get("volume_ratio", 0)), 2) if not pandas_isna(current.get("volume_ratio")) else None,
            "is_sideways": bool(current.get("is_sideways", False)),
        }

        # 6. EMA 위치 판단
        close = float(current.get("close", 0))
        ema_20 = indicators["ema_20"] or 0
        ema_50 = indicators["ema_50"] or 0
        if ema_20 > 0 and ema_50 > 0:
            if close > ema_20 > ema_50:
                ema_position = "강세 정배열 (가격→EMA20→EMA50)"
            elif close < ema_20 < ema_50:
                ema_position = "약세 역배열 (EMA50→EMA20→가격)"
            elif close > ema_20:
                ema_position = "단기 상승 (가격이 EMA20 위)"
            elif close < ema_20:
                ema_position = "단기 하락 (가격이 EMA20 아래)"
            else:
                ema_position = "중립"
        else:
            ema_position = "데이터 부족"

        # 7. 매수/매도 판단
        if score_result:
            direction = score_result["direction"]
            score = score_result["score"]
            reasons = score_result["reasons"]
        else:
            direction = "neutral"
            score = 0
            reasons = []

        # 종합 판단
        if indicators["is_sideways"]:
            verdict = "관망"
            verdict_reason = "횡보장 구간 — 추세 부재로 진입 비추"
        elif score >= 4:
            verdict = "적극 매수" if direction == "long" else "적극 매도(숏)"
            verdict_reason = f"{score}/7점 — 강한 시그널"
        elif score >= 3:
            verdict = "매수 고려" if direction == "long" else "매도(숏) 고려"
            verdict_reason = f"{score}/7점 — 시그널 양호"
        elif score >= 2:
            verdict = "조건부 매수" if direction == "long" else "조건부 매도(숏)"
            verdict_reason = f"{score}/7점 — 추가 확인 필요"
        elif score == 1:
            verdict = "관망 (약한 신호)"
            verdict_reason = f"1/7점 — 근거 부족"
        else:
            verdict = "관망"
            verdict_reason = "시그널 없음"

        atr_val = indicators["atr"] or 0
        entry = close
        if direction == "long" and score >= 2:
            sl = round(close - atr_val * 1.5, 2)
            tp = round(close + atr_val * 3.0, 2)
        elif direction == "short" and score >= 2:
            sl = round(close + atr_val * 1.5, 2)
            tp = round(close - atr_val * 3.0, 2)
        else:
            sl = None
            tp = None

        return {
            "symbol": symbol,
            "price": last,
            "change_1h": round(change_1h, 2),
            "change_24h": round(ticker.get("price_24h_pct", 0), 2),
            "high_24h": ticker.get("high_24h", 0),
            "low_24h": ticker.get("low_24h", 0),
            "volume_24h": ticker["volume_24h"],
            "funding_rate": funding_rate,
            "indicators": indicators,
            "ema_position": ema_position,
            "direction": direction,
            "score": score,
            "reasons": reasons,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "entry": entry,
            "sl": sl,
            "tp": tp,
        }

"""BriefingService — 시장 브리핑 생성 서비스."""

import asyncio
import functools
import logging
from datetime import datetime, timezone, timedelta

from data.features import add_all_features, candles_to_dataframe

logger = logging.getLogger(__name__)


class BriefingService:
    """시장 브리핑 데이터 생성."""

    def __init__(self, collector, config, coin_analyzer):
        self.collector = collector
        self.config = config
        self.coin_analyzer = coin_analyzer

    async def _run_blocking(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs),
        )

    async def generate_briefing(self) -> dict:
        """시장 브리핑 데이터 생성 (스케줄러 + 온디맨드 공용)."""
        now = datetime.now(timezone.utc)
        briefing: dict = {
            "time": now.strftime("%m/%d %H:%M UTC"),
            "market_summary": {},
            "scored_coins": [],
            "funding_alerts": [],
            "watched_pairs": [],
        }

        # 1. 티커 가져오기
        tickers = await self._run_blocking(self.collector.get_all_usdt_perpetuals)
        top_tickers = tickers[:20]

        # 2. 스코어링 스캔 먼저 (시간 소요됨)
        #    가격은 스캔 후 다시 가져옴
        scored_coins = []
        for t in top_tickers:
            symbol = t["symbol"]
            try:
                candles = await self._run_blocking(
                    self.collector.get_candles,
                    symbol, self.config.signal.primary_interval,
                    start_time=now - timedelta(days=120),
                )
                if not candles or len(candles) < 55:
                    continue

                df = candles_to_dataframe(candles)
                df = add_all_features(df)

                # Trend Following 스코어 계산 (시그널 안 나와도 점수 추출)
                result = self.coin_analyzer.score_pair(df, symbol)
                if result and result["score"] >= 1:
                    scored_coins.append(result)

            except Exception as e:
                logger.debug(f"Briefing score error {symbol}: {e}")

        scored_coins.sort(key=lambda x: x["score"], reverse=True)
        briefing["scored_coins"] = scored_coins

        # 4. 펀딩비 이상 스캔
        funding_alerts = []
        for t in top_tickers[:15]:
            symbol = t["symbol"]
            try:
                rates = await self._run_blocking(
                    self.collector.get_funding_rates,
                    symbol, start_time=now - timedelta(hours=24),
                )
                if rates:
                    latest = rates[-1].rate
                    if abs(latest) >= 0.0005:  # 0.05% 이상이면 주의
                        funding_alerts.append({
                            "symbol": symbol,
                            "rate": latest,
                        })
            except Exception as e:
                logger.debug(f"Optional data fetch failed (funding rate for {symbol}): {e}")

        funding_alerts.sort(key=lambda x: abs(x["rate"]), reverse=True)
        briefing["funding_alerts"] = funding_alerts

        # 5. 현재 관찰 페어 — caller가 주입해야 함
        # (watched_pairs는 main.py에서 설정)

        # 6. 주요 코인 현황 — 스캔 후 최신 가격 재조회
        fresh_tickers = await self._run_blocking(self.collector.get_all_usdt_perpetuals)
        fresh_map = {t["symbol"]: t for t in fresh_tickers}
        fetch_time = datetime.now(timezone.utc)
        briefing["time"] = fetch_time.strftime("%m/%d %H:%M:%S UTC")

        top_coins = []
        for t in fresh_tickers[:10]:
            symbol = t["symbol"]
            price = t["mark_price"]  # mark_price가 거래소 UI와 동일
            high = t.get("high_24h", 0)
            low = t.get("low_24h", 0)

            # 가격 검증: mark_price가 24h 범위 밖이면 로그
            if high > 0 and price > high * 1.05:
                logger.warning(
                    f"Price sanity check failed for {symbol}: "
                    f"mark={price}, high24h={high}, last={t['last_price']}"
                )
                price = t["last_price"]  # fallback

            prev_1h = t.get("prev_price_1h", 0)
            change_1h = (price - prev_1h) / prev_1h * 100 if prev_1h > 0 else 0

            top_coins.append({
                "symbol": symbol,
                "price": price,
                "change_1h": round(change_1h, 2),
                "change_24h": round(t.get("price_24h_pct", 0), 2),
                "high_24h": high,
                "low_24h": low,
                "volume_24h": t["volume_24h"],
            })

        briefing["market_summary"]["top_coins"] = top_coins

        return briefing

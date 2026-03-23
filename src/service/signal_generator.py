"""SignalGenerator — 시그널 생성 사이클 서비스."""

import asyncio
import functools
import logging
from datetime import datetime, timezone, timedelta

from core.types import SignalAction, SignalQuality
from data.features import add_all_features, candles_to_dataframe

logger = logging.getLogger(__name__)


class SignalGenerator:
    """시그널 생성 및 발송."""

    def __init__(
        self,
        collector,
        validator,
        config,
        pair_selector,
        trend_strategy,
        funding_strategy,
        cooldown,
        reporter,
        signal_tracker,
    ):
        self.collector = collector
        self.validator = validator
        self.config = config
        self.pair_selector = pair_selector
        self.trend_strategy = trend_strategy
        self.funding_strategy = funding_strategy
        self.cooldown = cooldown
        self.reporter = reporter
        self.signal_tracker = signal_tracker
        self._last_signal_time: datetime | None = None

    async def _run_blocking(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs),
        )

    async def run_cycle(self) -> None:
        """4시간 봉 기반 시그널 생성 사이클."""
        now = datetime.now(timezone.utc)
        logger.info(f"=== Signal cycle: {now.isoformat()} ===")

        # 쿨다운 체크만 (상태 체크 제거 — 시그널은 참고 정보로만 발송)
        if not self.cooldown.can_send_signal(self._last_signal_time):
            logger.info("Skipping: cooldown active")
            return

        try:
            # 페어 업데이트
            await self._update_pairs()

            # 최고 품질 시그널 수집
            best_signal = None

            for pair in self.pair_selector._current_pairs:
                signal_msg = await self.generate_for_pair(pair.symbol, now)
                if signal_msg is None:
                    continue

                # 최소 품질 필터
                min_quality = self.config.signal.min_signal_quality
                if min_quality == "strong" and signal_msg.quality != SignalQuality.STRONG:
                    continue

                # 최고 품질 선택
                if best_signal is None or self._signal_score(signal_msg) > self._signal_score(best_signal):
                    best_signal = signal_msg

            # 시그널 발송
            if best_signal:
                await self._send_signal(best_signal)

        except Exception as e:
            logger.error(f"Error in signal cycle: {e}", exc_info=True)
            if self.cooldown.record_api_error():
                await self.reporter.send_alert(f"API 에러 임계값 초과: {e}")

    async def generate_for_pair(self, symbol: str, now: datetime):
        """단일 페어에 대해 시그널 생성."""
        try:
            lookback = timedelta(days=120)
            candles = await self._run_blocking(
                self.collector.get_candles,
                symbol, self.config.signal.primary_interval,
                start_time=now - lookback,
            )
            if not candles:
                return None

            validation = self.validator.validate_candles(
                candles, self.config.signal.primary_interval, now,
            )
            if not validation.is_valid:
                return None

            df = candles_to_dataframe(candles)
            df = add_all_features(df)

            # Strategy A: Trend Following
            signal_a = self.trend_strategy.generate_signal(df, symbol)

            # Strategy B: Funding Rate
            latest_funding = None
            try:
                rates = await self._run_blocking(
                    self.collector.get_funding_rates,
                    symbol, start_time=now - timedelta(hours=24),
                )
                if rates:
                    latest_funding = rates[-1].rate
            except Exception as e:
                logger.debug(f"Optional data fetch failed (funding rate for {symbol}): {e}")

            signal_b = self.funding_strategy.generate_signal(
                df, symbol, latest_funding_rate=latest_funding,
            )

            # 최고 점수 시그널 선택
            candidates = [s for s in [signal_a, signal_b] if s is not None]
            if not candidates:
                return None

            return max(candidates, key=self._signal_score)

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    async def _send_signal(self, signal_msg) -> None:
        """시그널 발송 (참고 정보로만). 상태 전이 없음."""
        await self.reporter.send_signal(signal_msg)

        s = signal_msg.signal
        direction = "long" if s.action == SignalAction.ENTER_LONG else "short"
        self.signal_tracker.record_signal(
            symbol=s.symbol, direction=direction, strategy=s.strategy.value,
            quality=signal_msg.quality.value, entry_price=s.entry_price,
            stop_loss=s.stop_loss, take_profit=s.take_profit, signal_time=s.timestamp,
        )
        self._last_signal_time = datetime.now(timezone.utc)
        logger.info(f"Signal sent (info only): {direction} {s.symbol}")

    @staticmethod
    def _signal_score(msg) -> float:
        """시그널 스코어 (정렬용)."""
        quality_score = {
            SignalQuality.STRONG: 3,
            SignalQuality.MODERATE: 2,
            SignalQuality.WEAK: 1,
        }
        return quality_score.get(msg.quality, 0) + msg.signal.confidence

    async def _update_pairs(self) -> None:
        try:
            tickers = await self._run_blocking(self.collector.get_all_usdt_perpetuals)

            candle_data = {}
            for t in tickers[:30]:
                symbol = t["symbol"]
                try:
                    candles = await self._run_blocking(
                        self.collector.get_candles,
                        symbol=symbol,
                        interval=self.config.signal.primary_interval,
                        start_time=datetime.now(timezone.utc) - timedelta(days=210),
                    )
                    if candles:
                        df = candles_to_dataframe(candles)
                        df = add_all_features(df)
                        candle_data[symbol] = df
                except Exception as e:
                    logger.debug(f"Failed to get candles for {symbol}: {e}")

            self.pair_selector.select_pairs(tickers, candle_data)
        except Exception as e:
            logger.error(f"Failed to update pairs: {e}")

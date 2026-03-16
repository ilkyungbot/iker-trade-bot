"""
Main entry point: SignalBot — 대화형 시그널 봇.

자동 주문 없음. 시그널 발송 + 대화 상태 추적만 수행.
"""

import asyncio
import functools
import logging
import signal
import sys
from datetime import datetime, timezone, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.config import AppConfig
from core.types import ConversationState, SignalAction, SignalQuality
from data.collector import BybitCollector
from data.validator import DataValidator
from data.features import add_all_features, candles_to_dataframe
from strategy.pair_selector import PairSelector
from strategy.trend_following import TrendFollowingStrategy
from strategy.funding_rate import FundingRateStrategy
from conversation.state_machine import ConversationStateMachine
from conversation.signal_tracker import SignalTracker
from execution.circuit_breaker import SignalCooldown
from review.reporter import Reporter, TelegramBotSender
from review.telegram_commands import TelegramCommandHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SignalBot:
    """시그널 봇 오케스트레이터."""

    def __init__(self, config: AppConfig):
        self.config = config

        # Data layer
        self.collector = BybitCollector.from_config(
            config.bybit.api_key, config.bybit.api_secret, config.bybit.testnet,
        )
        self.validator = DataValidator()

        # Strategy layer
        self.pair_selector = PairSelector(
            max_pairs=config.signal.max_pairs,
            rebalance_days=config.signal.pair_rebalance_days,
        )
        self.trend_strategy = TrendFollowingStrategy()
        self.funding_strategy = FundingRateStrategy()

        # Conversation layer
        self.state_machine = ConversationStateMachine()
        self.signal_tracker = SignalTracker()

        # Cooldown & API error tracking
        self.cooldown = SignalCooldown(cooldown_minutes=config.signal.signal_cooldown_minutes)
        self._last_signal_time: datetime | None = None

        # Telegram
        tg_sender = (
            TelegramBotSender(config.telegram.bot_token)
            if config.telegram.bot_token
            else None
        )
        self.reporter = Reporter(sender=tg_sender, chat_id=config.telegram.chat_id)

        self.cmd_handler = TelegramCommandHandler(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id,
        )
        self.cmd_handler.attach_bot(self)

        self._running = False

    async def start(self) -> None:
        """Start the signal bot."""
        logger.info("Starting signal bot")
        self._running = True

        scheduler = AsyncIOScheduler()

        # 시그널 사이클: 매 4시간 (4H 봉 마감 시)
        scheduler.add_job(self.signal_cycle, "cron", hour="0,4,8,12,16,20", minute=1)

        # 포지션 모니터링: 매 15분 (사용자가 포지션 중일 때만)
        scheduler.add_job(self.monitor_position, "cron", minute="*/15")

        # 시그널 만료 체크: 매 10분
        scheduler.add_job(self.check_expiry, "cron", minute="*/10")

        # 시그널 결과 추적: 매 시간
        scheduler.add_job(self.check_signal_outcomes, "cron", minute=30)

        # 일간 리포트: 매일 00:05 UTC
        scheduler.add_job(self.daily_report, "cron", hour=0, minute=5)

        # 주간 리포트: 월요일 00:10 UTC
        scheduler.add_job(self.weekly_report, "cron", day_of_week="mon", hour=0, minute=10)

        scheduler.start()

        # Telegram 커맨드 리스너 시작
        await self.cmd_handler.start()

        logger.info("Scheduler started. Running initial signal cycle...")
        await self.signal_cycle()

        try:
            while self._running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down...")
        finally:
            await self.cmd_handler.stop()
            scheduler.shutdown()
            logger.info("Signal bot stopped.")

    async def signal_cycle(self) -> None:
        """4시간 봉 기반 시그널 생성 사이클."""
        now = datetime.now(timezone.utc)
        logger.info(f"=== Signal cycle: {now.isoformat()} ===")

        # 상태 체크: IDLE이 아니면 스킵
        session = self.state_machine.get_session(self.config.telegram.chat_id)
        if session.state != ConversationState.IDLE:
            logger.info(f"Skipping signal cycle: state is {session.state.value}")
            return

        # 쿨다운 체크
        if not self.cooldown.can_send_signal(self._last_signal_time):
            logger.info("Skipping: cooldown active")
            return

        try:
            # 페어 업데이트
            await self._update_pairs()

            # 최고 품질 시그널 수집
            best_signal = None

            for pair in self.pair_selector._current_pairs:
                signal_msg = await self._generate_signal_for_pair(pair.symbol, now)
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

    async def monitor_position(self) -> None:
        """모니터링 중인 포지션 업데이트."""
        session = self.state_machine.get_session(self.config.telegram.chat_id)
        if session.state not in (ConversationState.MONITORING, ConversationState.EXIT_SIGNAL_SENT):
            return

        if not session.active_signal:
            return

        s = session.active_signal.signal
        try:
            candles = await self._run_sync(
                self.collector.get_candles,
                s.symbol, "1",
                start_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            )
            if not candles:
                return

            current_price = candles[-1].close
            direction = "long" if s.action == SignalAction.ENTER_LONG else "short"

            # TP/SL 도달 체크
            if direction == "long":
                tp_hit = current_price >= s.take_profit
                sl_hit = current_price <= s.stop_loss
            else:
                tp_hit = current_price <= s.take_profit
                sl_hit = current_price >= s.stop_loss

            if (tp_hit or sl_hit) and session.state == ConversationState.MONITORING:
                reason = "목표가 도달! \U0001f389" if tp_hit else "손절가 도달 \u26a0\ufe0f"
                self.state_machine.send_exit_signal(self.config.telegram.chat_id)
                await self.reporter.send_exit_signal(s.symbol, direction, reason)
            elif session.state == ConversationState.MONITORING:
                # 정기 모니터링 업데이트 (15분마다)
                entry = session.user_entry_price or s.entry_price
                await self.reporter.send_monitoring_update(
                    s.symbol, direction, entry, current_price, s.stop_loss, s.take_profit,
                )

        except Exception as e:
            logger.error(f"Monitor error: {e}")

    async def check_expiry(self) -> None:
        """SIGNAL_SENT 상태 타임아웃 체크."""
        expired = self.state_machine.check_expiry(
            self.config.telegram.chat_id,
            self.config.signal.signal_expiry_minutes,
        )
        if expired:
            await self.reporter.send_alert("시그널 응답 시간 초과. 자동으로 패스 처리되었습니다.")
            logger.info("Signal expired, auto-passed")

    async def check_signal_outcomes(self) -> None:
        """과거 시그널 결과 추적."""
        unchecked = self.signal_tracker.get_unchecked_signals()
        for sig_data in unchecked:
            try:
                symbol = sig_data["symbol"]
                signal_time = datetime.fromisoformat(sig_data["signal_time"])
                if signal_time.tzinfo is None:
                    signal_time = signal_time.replace(tzinfo=timezone.utc)

                elapsed = datetime.now(timezone.utc) - signal_time
                hours = elapsed.total_seconds() / 3600

                candles = await self._run_sync(
                    self.collector.get_candles,
                    symbol, "60",
                    start_time=signal_time,
                )
                if not candles:
                    continue

                prices = {c.timestamp: c.close for c in candles}
                sorted_times = sorted(prices.keys())

                price_4h = None
                price_8h = None
                price_24h = None

                for t in sorted_times:
                    h = (t - signal_time).total_seconds() / 3600
                    if h >= 4 and price_4h is None:
                        price_4h = prices[t]
                    if h >= 8 and price_8h is None:
                        price_8h = prices[t]
                    if h >= 24 and price_24h is None:
                        price_24h = prices[t]

                self.signal_tracker.update_outcome(
                    sig_data["id"],
                    price_4h=price_4h,
                    price_8h=price_8h,
                    price_24h=price_24h,
                )
            except Exception as e:
                logger.error(f"Error checking signal outcome: {e}")

    async def daily_report(self) -> None:
        """일간 시그널 리포트."""
        report = self.signal_tracker.weekly_report(days=1)
        if report["total"] > 0:
            await self.reporter.send_weekly_accuracy(report)

    async def weekly_report(self) -> None:
        """주간 시그널 정확도 리포트."""
        report = self.signal_tracker.weekly_report()
        await self.reporter.send_weekly_accuracy(report)

    # --- Private helpers ---

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs),
        )

    async def _update_pairs(self) -> None:
        try:
            tickers = await self._run_sync(self.collector.get_all_usdt_perpetuals)

            candle_data = {}
            for t in tickers[:30]:
                symbol = t["symbol"]
                try:
                    candles = await self._run_sync(
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

    async def _generate_signal_for_pair(self, symbol: str, now: datetime):
        """단일 페어에 대해 시그널 생성."""
        try:
            lookback = timedelta(days=120)
            candles = await self._run_sync(
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
                rates = await self._run_sync(
                    self.collector.get_funding_rates,
                    symbol, start_time=now - timedelta(hours=24),
                )
                if rates:
                    latest_funding = rates[-1].rate
            except Exception:
                pass

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
        """시그널 발송 + 상태 전이 + 추적 기록."""
        chat_id = self.config.telegram.chat_id
        success = self.state_machine.send_signal(chat_id, signal_msg)
        if not success:
            logger.warning("Failed to transition to SIGNAL_SENT")
            return

        await self.reporter.send_signal(signal_msg)

        # 정확도 추적용 기록
        s = signal_msg.signal
        direction = "long" if s.action == SignalAction.ENTER_LONG else "short"
        self.signal_tracker.record_signal(
            symbol=s.symbol,
            direction=direction,
            strategy=s.strategy.value,
            quality=signal_msg.quality.value,
            entry_price=s.entry_price,
            stop_loss=s.stop_loss,
            take_profit=s.take_profit,
            signal_time=s.timestamp,
        )

        self._last_signal_time = datetime.now(timezone.utc)
        logger.info(f"Signal sent: {direction} {s.symbol} (quality={signal_msg.quality.value})")

    @staticmethod
    def _signal_score(msg) -> float:
        """시그널 스코어 (정렬용)."""
        quality_score = {
            SignalQuality.STRONG: 3,
            SignalQuality.MODERATE: 2,
            SignalQuality.WEAK: 1,
        }
        return quality_score.get(msg.quality, 0) + msg.signal.confidence

    def stop(self) -> None:
        self._running = False


def main():
    config = AppConfig.from_env()
    bot = SignalBot(config)

    loop = asyncio.new_event_loop()

    def shutdown_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        bot.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        loop.run_until_complete(bot.start())
    finally:
        loop.close()


if __name__ == "__main__":
    main()

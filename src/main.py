"""
Main entry point: SignalBot — 대화형 시그널 봇.

자동 주문 없음. 시그널 발송 + 대화 상태 추적만 수행.
"""

import asyncio
import functools
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.config import AppConfig
from core.types import ConversationState
from data.collector import BybitCollector
from data.validator import DataValidator
from data.features import add_all_features, candles_to_dataframe
from strategy.pair_selector import PairSelector
from strategy.trend_following import TrendFollowingStrategy
from strategy.funding_rate import FundingRateStrategy
from conversation.state_machine import ConversationStateMachine
from conversation.signal_tracker import SignalTracker
from conversation.position_manager import PositionManager
from strategy.position_monitor import PositionMonitorV2
from strategy.edge_detector import EdgeDetector
from strategy.market_regime import MarketRegimeClassifier
from execution.exit_manager import ExitManager
from execution.portfolio_guard import PortfolioGuard
from execution.risk_calculator import validate_position
from execution.circuit_breaker import SignalCooldown
from review.reporter import Reporter, TelegramBotSender
from review.telegram_commands import TelegramCommandHandler
from review.trading_journal import TradingJournal
from service.coin_analyzer import CoinAnalyzer
from service.briefing_service import BriefingService
from service.signal_generator import SignalGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SignalBot:
    """시그널 봇 오케스트레이터."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._executor = ThreadPoolExecutor()

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
        self.position_manager = PositionManager()
        self.position_monitor_v2 = PositionMonitorV2()
        self.portfolio_guard = PortfolioGuard()
        self.trading_journal = TradingJournal()

        # Cooldown & API error tracking
        self.cooldown = SignalCooldown(cooldown_minutes=config.signal.signal_cooldown_minutes)
        self._last_regime: dict[str, str] = {}  # symbol -> last regime value

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

        # Service layer
        self.coin_analyzer = CoinAnalyzer(
            collector=self.collector,
            config=config,
        )
        self.briefing_service = BriefingService(
            collector=self.collector,
            config=config,
            coin_analyzer=self.coin_analyzer,
        )
        self.signal_generator = SignalGenerator(
            collector=self.collector,
            validator=self.validator,
            config=config,
            pair_selector=self.pair_selector,
            trend_strategy=self.trend_strategy,
            funding_strategy=self.funding_strategy,
            cooldown=self.cooldown,
            reporter=self.reporter,
            signal_tracker=self.signal_tracker,
        )

        self._running = False

    async def start(self) -> None:
        """Start the signal bot."""
        logger.info("Starting signal bot")
        self._running = True

        scheduler = AsyncIOScheduler()

        scheduler.add_job(self.signal_cycle, "cron", hour="0,4,8,12,16,20", minute=1)
        scheduler.add_job(self.hourly_briefing, "cron", minute=0)
        scheduler.add_job(self.check_signal_outcomes, "cron", minute=30)
        scheduler.add_job(self.monitor_manual_positions, "cron", minute="*/5")
        scheduler.add_job(self.daily_report, "cron", hour=0, minute=5)
        scheduler.add_job(self.weekly_report, "cron", day_of_week="mon", hour=0, minute=10)
        scheduler.add_job(self._reset_daily_guard, "cron", hour=0, minute=0)
        scheduler.add_job(self._reset_monthly_guard, "cron", day=1, hour=0, minute=0)

        scheduler.start()
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

    # --- Delegations to services ---

    async def signal_cycle(self) -> None:
        await self.signal_generator.run_cycle()

    async def generate_briefing(self) -> dict:
        briefing = await self.briefing_service.generate_briefing()
        briefing["watched_pairs"] = [p.symbol for p in self.pair_selector._current_pairs]
        return briefing

    async def analyze_coin(self, query: str) -> dict | None:
        return await self.coin_analyzer.analyze_coin(query)

    # --- Kept in orchestrator ---

    async def hourly_briefing(self) -> None:
        try:
            briefing = await self.generate_briefing()
            await self.reporter.send_hourly_briefing(briefing)
        except Exception as e:
            logger.error(f"Error in hourly briefing: {e}", exc_info=True)

    async def check_signal_outcomes(self) -> None:
        unchecked = self.signal_tracker.get_unchecked_signals()
        for sig_data in unchecked:
            try:
                symbol = sig_data["symbol"]
                signal_time = datetime.fromisoformat(sig_data["signal_time"])
                if signal_time.tzinfo is None:
                    signal_time = signal_time.replace(tzinfo=timezone.utc)

                elapsed = datetime.now(timezone.utc) - signal_time
                hours = elapsed.total_seconds() / 3600

                candles = await self._run_blocking(
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

    async def monitor_manual_positions(self) -> None:
        positions = self.position_manager.get_all_active_positions()
        if not positions:
            return

        now = datetime.now(timezone.utc)

        btc_df = None
        try:
            btc_candles = await self._run_blocking(
                self.collector.get_candles,
                "BTCUSDT", self.config.signal.primary_interval,
                start_time=now - timedelta(days=60),
            )
            if btc_candles and len(btc_candles) > 50:
                btc_df = candles_to_dataframe(btc_candles)
                btc_df = add_all_features(btc_df)
        except Exception:
            pass

        for pos in positions:
            try:
                candles = await self._run_blocking(
                    self.collector.get_candles,
                    pos.symbol, self.config.signal.primary_interval,
                    start_time=now - timedelta(days=30),
                )
                if not candles or len(candles) < 10:
                    continue

                df = candles_to_dataframe(candles)
                df = add_all_features(df)

                recent = await self._run_blocking(
                    self.collector.get_candles,
                    pos.symbol, "1",
                    start_time=now - timedelta(minutes=5),
                )
                if not recent:
                    continue
                current_price = recent[-1].close

                atr = float(df.iloc[-1].get("atr", 0))

                funding_rates = []
                try:
                    rates = await self._run_blocking(
                        self.collector.get_funding_rates,
                        pos.symbol, start_time=now - timedelta(hours=24),
                    )
                    if rates:
                        funding_rates = rates
                except Exception:
                    pass

                oi_data = []
                try:
                    oi_list = await self._run_blocking(
                        self.collector.get_open_interest_history,
                        pos.symbol, "1h", start_time=now - timedelta(hours=4),
                    )
                    if oi_list:
                        oi_data = oi_list
                except Exception:
                    pass

                result = self.position_monitor_v2.check_position(
                    pos, df, current_price, atr,
                    funding_rates=funding_rates,
                    oi_data=oi_data,
                    btc_df=btc_df,
                )

                regime = result.get("regime")
                if regime:
                    last = self._last_regime.get(pos.symbol)
                    if last and last != regime.regime.value:
                        await self.reporter.send_regime_change(regime, position=pos)
                    self._last_regime[pos.symbol] = regime.regime.value

                for sig in result["exit_signals"]:
                    await self.reporter.send_exit_signal_v2(sig, position=pos)

                for edge in result["edge_signals"]:
                    await self.reporter.send_edge_alert(edge, pos)

            except Exception as e:
                logger.error(f"Manual position monitor error for {pos.symbol}: {e}", exc_info=True)

    async def _reset_daily_guard(self) -> None:
        self.portfolio_guard.reset_daily()
        logger.info("Daily portfolio guard reset")

    async def _reset_monthly_guard(self) -> None:
        self.portfolio_guard.reset_monthly()
        logger.info("Monthly portfolio guard reset")

    async def daily_report(self) -> None:
        report = self.signal_tracker.weekly_report(days=1)
        if report["total"] > 0:
            await self.reporter.send_weekly_accuracy(report)

    async def weekly_report(self) -> None:
        report = self.signal_tracker.weekly_report()
        await self.reporter.send_weekly_accuracy(report)

    async def _run_blocking(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs),
        )

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

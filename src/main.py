"""
Main entry point: wires all layers together and runs the trading loop.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.config import AppConfig
from core.types import TradingMode, SignalAction
from core.safety import REQUIRE_EXCHANGE_STOP_LOSS
from data.collector import BybitCollector
from data.validator import DataValidator
from data.features import add_all_features, candles_to_dataframe, add_ema
from data.storage import Storage
from strategy.pair_selector import PairSelector
from strategy.trend_following import TrendFollowingStrategy
from strategy.funding_rate import FundingRateStrategy
from sizing.kelly import kelly_fraction, calculate_position_size
from sizing.adjustments import apply_all_adjustments
from execution.order_manager import OrderManager
from execution.position_tracker import PositionTracker
from execution.circuit_breaker import CircuitBreaker
from review.trade_logger import TradeLogger
from review.performance import calculate_metrics, calculate_strategy_attribution
from review.reporter import Reporter
from review.retrainer import Retrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: AppConfig, initial_capital: float = 1_000_000.0):
        self.config = config

        # Data layer
        self.collector = BybitCollector.from_config(
            config.bybit.api_key, config.bybit.api_secret, config.bybit.testnet,
        )
        self.validator = DataValidator()
        self.storage = Storage(config.database.url)

        # Strategy layer
        self.pair_selector = PairSelector(
            max_pairs=config.trading.max_pairs,
            rebalance_days=config.trading.pair_rebalance_days,
        )
        self.trend_strategy = TrendFollowingStrategy()
        self.funding_strategy = FundingRateStrategy()

        # Execution layer
        self.order_manager = OrderManager(mode=config.trading.mode)
        self.position_tracker = PositionTracker(initial_capital)
        self.circuit_breaker = CircuitBreaker()

        # Review layer
        self.trade_logger = TradeLogger(self.storage)
        self.reporter = Reporter(chat_id=config.telegram.chat_id)
        self.retrainer = Retrainer()

        # State
        self._running = False
        self._last_retrain: datetime | None = None
        self._last_param_review: datetime | None = None
        self._candle_cache: dict[str, dict[str, list]] = {}  # symbol -> interval -> candles

    async def start(self) -> None:
        """Start the trading bot with scheduled jobs."""
        logger.info(f"Starting trading bot in {self.config.trading.mode.value} mode")
        self._running = True

        scheduler = AsyncIOScheduler()

        # Main trading loop: every hour on the 1-minute mark
        scheduler.add_job(self.trading_cycle, "cron", minute=1, second=0)

        # Daily reset and report: midnight UTC
        scheduler.add_job(self.daily_tasks, "cron", hour=0, minute=5)

        # Weekly report: Monday midnight UTC
        scheduler.add_job(self.weekly_tasks, "cron", day_of_week="mon", hour=0, minute=10)

        # Health check heartbeat: every hour
        scheduler.add_job(self.heartbeat, "cron", minute=30)

        scheduler.start()

        logger.info("Scheduler started. Waiting for trading cycles...")

        # Run initial trading cycle
        await self.trading_cycle()

        # Keep running
        try:
            while self._running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down...")
            scheduler.shutdown()

    async def trading_cycle(self) -> None:
        """Main hourly trading cycle."""
        now = datetime.now(timezone.utc)
        logger.info(f"=== Trading cycle: {now.isoformat()} ===")

        # Check circuit breaker
        cb_state = self.circuit_breaker.check(self.position_tracker.state)
        if self.circuit_breaker.is_halted:
            logger.warning(f"Trading halted: {cb_state.value}")
            return

        try:
            # 1. Update pair selection
            await self._update_pairs()

            # 2. For each selected pair, fetch data and check signals
            for pair in self.pair_selector._current_pairs:
                await self._process_pair(pair.symbol, now)

            # 3. Check stop-losses for all positions
            await self._check_stops()

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            if self.circuit_breaker.record_api_error():
                await self.reporter.send_alert(f"API error threshold breached: {e}")

    async def _update_pairs(self) -> None:
        """Update pair selection if rebalance is due."""
        try:
            tickers = self.collector.get_all_usdt_perpetuals()

            # Fetch candle data for top candidates
            candle_data = {}
            for t in tickers[:30]:  # top 30 by volume
                symbol = t["symbol"]
                try:
                    candles = self.collector.get_candles(
                        symbol=symbol,
                        interval=self.config.trading.primary_interval,
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

    async def _process_pair(self, symbol: str, now: datetime) -> None:
        """Process a single pair: fetch data, generate signals, execute."""
        try:
            # Fetch 1H and 4H candles
            lookback_1h = timedelta(days=60)
            lookback_4h = timedelta(days=120)

            candles_1h = self.collector.get_candles(
                symbol, self.config.trading.primary_interval,
                start_time=now - lookback_1h,
            )
            candles_4h = self.collector.get_candles(
                symbol, self.config.trading.trend_interval,
                start_time=now - lookback_4h,
            )

            if not candles_1h or not candles_4h:
                return

            # Validate
            validation = self.validator.validate_candles(candles_1h, self.config.trading.primary_interval, now)
            if not validation.is_valid:
                logger.warning(f"Invalid data for {symbol}, skipping")
                return

            # Compute features
            df_1h = candles_to_dataframe(candles_1h)
            df_1h = add_all_features(df_1h)
            df_4h = candles_to_dataframe(candles_4h)
            df_4h = add_ema(df_4h, 50)

            # Save candles
            self.storage.save_candles(candles_1h)

            # Check current position
            current_side = None
            for pos in self.position_tracker.state.positions:
                if pos.symbol == symbol:
                    current_side = pos.side.value
                    # Update unrealized PnL
                    current_price = df_1h["close"].iloc[-1]
                    self.position_tracker.update_unrealized_pnl(symbol, current_price)
                    # Update trailing stop
                    atr = df_1h["atr"].iloc[-1]
                    if not __import__("math").isnan(atr):
                        self.position_tracker.update_trailing_stop(symbol, atr)
                    break

            # Get latest funding rate
            latest_funding = None
            try:
                rates = self.collector.get_funding_rates(
                    symbol, start_time=now - timedelta(hours=24),
                )
                if rates:
                    latest_funding = rates[-1].rate
            except Exception:
                pass

            # Generate signals from both strategies
            # Strategy A: Trend Following (70% allocation)
            signal_a = self.trend_strategy.generate_signal(
                df_1h, df_4h, symbol, current_side,
            )

            # Strategy B: Funding Rate (30% allocation)
            signal_b = self.funding_strategy.generate_signal(
                df_1h, df_4h, symbol, current_side,
                latest_funding_rate=latest_funding,
            )

            # Process signals
            for signal, allocation in [
                (signal_a, self.config.trading.strategy_a_allocation),
                (signal_b, self.config.trading.strategy_b_allocation),
            ]:
                if signal is None:
                    continue

                if signal.action == SignalAction.EXIT:
                    await self._execute_exit(symbol, df_1h["close"].iloc[-1])
                elif signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
                    await self._execute_entry(signal, allocation, df_1h)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    async def _execute_entry(
        self, signal, allocation: float, df_1h,
    ) -> None:
        """Execute an entry signal."""
        if not self.position_tracker.can_open_position():
            return

        state = self.position_tracker.state

        # Calculate position size
        trades = self.storage.get_trades(
            start=datetime.now(timezone.utc) - timedelta(days=60),
        )
        metrics = calculate_metrics(trades)

        risk_fraction = kelly_fraction(
            metrics.win_rate if metrics.win_rate > 0 else 0.4,
            metrics.avg_win if metrics.avg_win > 0 else 0.02,
            metrics.avg_loss if metrics.avg_loss > 0 else 0.01,
        )

        # Get adjustments
        atr_pct = df_1h["atr_pct"].iloc[-1] if "atr_pct" in df_1h.columns else 0.02
        existing_sides = [p.side.value for p in state.positions]
        new_side = "long" if signal.action == SignalAction.ENTER_LONG else "short"

        capital_for_strategy = state.total_capital * allocation
        base_quantity = calculate_position_size(
            capital=capital_for_strategy,
            risk_fraction=risk_fraction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            leverage=self.config.trading.default_leverage,
        )

        # Apply adjustments
        adjusted_quantity = apply_all_adjustments(
            base_size=base_quantity,
            current_atr_pct=float(atr_pct) if not __import__("math").isnan(atr_pct) else 0.02,
            existing_position_sides=existing_sides,
            new_signal_side=new_side,
            correlation_to_existing=0.7,  # conservative default
            current_mdd=state.current_mdd,
            consecutive_losses=state.consecutive_losses,
            ml_confidence=signal.confidence,
        )

        # Apply circuit breaker size multiplier
        adjusted_quantity *= self.circuit_breaker.size_multiplier

        if adjusted_quantity <= 0:
            return

        # Place order
        order = self.order_manager.place_entry_order(
            signal=signal,
            quantity=adjusted_quantity,
            leverage=self.config.trading.default_leverage,
            pair_volume_24h=50_000_000,  # TODO: get actual volume
        )

        if order and order.status.value == "filled":
            # Set stop-loss
            sl_set = self.order_manager.set_stop_loss(signal.symbol, signal.stop_loss)
            if REQUIRE_EXCHANGE_STOP_LOSS and not sl_set:
                logger.error(f"Failed to set SL for {signal.symbol}, closing position")
                positions = self.order_manager.get_paper_positions()
                if signal.symbol in positions:
                    self.order_manager.close_position(positions[signal.symbol])
                return

            # Track position
            from core.types import Position, Side
            position = Position(
                symbol=signal.symbol,
                side=Side.LONG if signal.action == SignalAction.ENTER_LONG else Side.SHORT,
                entry_price=signal.entry_price,
                quantity=adjusted_quantity,
                leverage=self.config.trading.default_leverage,
                stop_loss=signal.stop_loss,
                trailing_stop=signal.take_profit,
                strategy=signal.strategy,
                entry_time=datetime.now(timezone.utc),
            )
            self.position_tracker.add_position(position)

            await self.reporter.send_signal_alert(signal)
            logger.info(f"Entered {new_side} {signal.symbol}: qty={adjusted_quantity:.6f}")

    async def _execute_exit(self, symbol: str, current_price: float) -> None:
        """Execute an exit signal."""
        trade = self.position_tracker.close_position(
            symbol, current_price, fees=current_price * 0.0004,  # ~0.04% round trip
        )
        if trade:
            self.trade_logger.log_trade(trade)
            self.order_manager.close_position(
                next(
                    (p for p in self.position_tracker.state.positions if p.symbol == symbol),
                    None,
                ) or trade  # fallback
            )
            await self.reporter.send_trade_alert(trade)

    async def _check_stops(self) -> None:
        """Check stop-losses for all positions."""
        prices = {}
        for pos in self.position_tracker.state.positions:
            try:
                candles = self.collector.get_candles(
                    pos.symbol, "1", datetime.now(timezone.utc) - timedelta(minutes=5),
                )
                if candles:
                    prices[pos.symbol] = candles[-1].close
            except Exception:
                pass

        to_close = self.position_tracker.check_stop_losses(prices)
        for symbol in to_close:
            if symbol in prices:
                await self._execute_exit(symbol, prices[symbol])

    async def daily_tasks(self) -> None:
        """Daily maintenance tasks."""
        logger.info("Running daily tasks...")

        trades = self.storage.get_trades()
        metrics = calculate_metrics(trades)

        await self.reporter.send_daily_report(
            self.position_tracker.state, metrics, datetime.now(timezone.utc),
        )

        # Save performance snapshot
        state = self.position_tracker.state
        self.storage.save_performance(
            date=datetime.now(timezone.utc),
            total_capital=state.total_capital,
            daily_pnl=state.daily_pnl,
            daily_pnl_percent=state.daily_pnl / max(state.total_capital, 1) * 100,
            cumulative_pnl=state.total_capital - self.position_tracker.initial_capital,
            max_drawdown=state.current_mdd,
            open_positions=state.position_count,
            trades_today=0,
            win_rate=metrics.win_rate,
        )

        self.position_tracker.reset_daily_pnl()

    async def weekly_tasks(self) -> None:
        """Weekly maintenance tasks."""
        logger.info("Running weekly tasks...")

        trades = self.storage.get_trades()
        metrics = calculate_metrics(trades)
        strategy_attrs = calculate_strategy_attribution(trades)

        await self.reporter.send_weekly_report(
            self.position_tracker.state, metrics, strategy_attrs,
        )

        # Check if retrain is needed
        if self.retrainer.should_retrain(self._last_retrain):
            logger.info("Monthly retrain due — would trigger ML retrain here")
            self._last_retrain = datetime.now(timezone.utc)

        # Check allocation adjustment
        if self.retrainer.should_review_params(self._last_param_review):
            adj = self.retrainer.calculate_allocation_adjustment(trades)
            logger.info(
                f"Allocation review: A={adj.strategy_a_weight:.0%}, "
                f"B={adj.strategy_b_weight:.0%} — {adj.reason}"
            )
            self._last_param_review = datetime.now(timezone.utc)

        self.position_tracker.reset_weekly_pnl()

    async def heartbeat(self) -> None:
        """Send periodic health check."""
        state = self.position_tracker.state
        msg = (
            f"Heartbeat: Capital={state.total_capital:,.0f} "
            f"Positions={state.position_count} "
            f"MDD={state.current_mdd:.1%} "
            f"CB={state.circuit_breaker_state.value}"
        )
        logger.info(msg)

    def stop(self) -> None:
        self._running = False


def main():
    config = AppConfig.from_env()
    initial_capital = float(__import__("os").getenv("INITIAL_CAPITAL", "1000000"))
    bot = TradingBot(config, initial_capital)

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

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

import numpy as np
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler


def pandas_isna(val) -> bool:
    """Null check that handles both pandas NA and numpy NaN."""
    if val is None:
        return True
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False

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

        # 1시간 브리핑: 매시 정각
        scheduler.add_job(self.hourly_briefing, "cron", minute=0)

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

    async def hourly_briefing(self) -> None:
        """1시간 브리핑 발송."""
        try:
            briefing = await self.generate_briefing()
            await self.reporter.send_hourly_briefing(briefing)
        except Exception as e:
            logger.error(f"Error in hourly briefing: {e}", exc_info=True)

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
        tickers = await self._run_sync(self.collector.get_all_usdt_perpetuals)
        top_tickers = tickers[:20]

        # 2. 주요 코인 현황 (상위 10개)
        top_coins = []
        for t in top_tickers[:10]:
            symbol = t["symbol"]
            try:
                candles = await self._run_sync(
                    self.collector.get_candles,
                    symbol, "240",
                    start_time=now - timedelta(hours=8),
                )
                change_4h = 0.0
                if candles and len(candles) >= 2:
                    prev_close = candles[-2].close
                    curr_close = candles[-1].close
                    change_4h = (curr_close - prev_close) / prev_close * 100 if prev_close > 0 else 0

                top_coins.append({
                    "symbol": symbol,
                    "price": t["last_price"],
                    "change_4h": round(change_4h, 2),
                    "volume_24h": t["volume_24h"],
                })
            except Exception as e:
                logger.debug(f"Briefing ticker error {symbol}: {e}")

        briefing["market_summary"]["top_coins"] = top_coins

        # 3. 스코어링 스캔 (상위 20개 코인)
        scored_coins = []
        for t in top_tickers:
            symbol = t["symbol"]
            try:
                candles = await self._run_sync(
                    self.collector.get_candles,
                    symbol, self.config.signal.primary_interval,
                    start_time=now - timedelta(days=120),
                )
                if not candles or len(candles) < 55:
                    continue

                df = candles_to_dataframe(candles)
                df = add_all_features(df)

                # Trend Following 스코어 계산 (시그널 안 나와도 점수 추출)
                result = self._score_pair(df, symbol)
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
                rates = await self._run_sync(
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
            except Exception:
                pass

        funding_alerts.sort(key=lambda x: abs(x["rate"]), reverse=True)
        briefing["funding_alerts"] = funding_alerts

        # 5. 현재 관찰 페어
        briefing["watched_pairs"] = [p.symbol for p in self.pair_selector._current_pairs]

        return briefing

    def _score_pair(self, df, symbol: str) -> dict | None:
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

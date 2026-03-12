"""
Telegram command handler — responds to user commands in Korean.

Commands: /상태, /포지션, /성과, /오늘, /중지, /재개, /페어
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

from core.types import CircuitBreakerState
from review.reporter import _strategy_kr, _cb_state_kr, _side_kr, _sign

logger = logging.getLogger(__name__)


class TelegramCommandHandler:
    """Listens for Telegram commands and responds with bot state."""

    def __init__(self, bot_token: str, chat_id: str):
        if not HAS_TELEGRAM:
            raise ImportError("python-telegram-bot is required")
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._bot_ref = None  # set via attach_bot()
        self._app: Application | None = None

    def attach_bot(self, trading_bot) -> None:
        """Attach reference to TradingBot for querying state."""
        self._bot_ref = trading_bot

    async def start(self) -> None:
        """Start polling for Telegram commands (non-blocking)."""
        if not self.bot_token:
            logger.info("Telegram command handler disabled (no bot token)")
            return

        self._app = Application.builder().token(self.bot_token).build()

        self._app.add_handler(CommandHandler("상태", self._cmd_status))
        self._app.add_handler(CommandHandler("포지션", self._cmd_positions))
        self._app.add_handler(CommandHandler("성과", self._cmd_performance))
        self._app.add_handler(CommandHandler("오늘", self._cmd_today))
        self._app.add_handler(CommandHandler("중지", self._cmd_stop))
        self._app.add_handler(CommandHandler("재개", self._cmd_resume))
        self._app.add_handler(CommandHandler("페어", self._cmd_pairs))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("start", self._cmd_help))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram command handler started")

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    def _check_auth(self, update: Update) -> bool:
        """Only allow commands from the configured chat_id."""
        if not self.chat_id:
            return True
        return str(update.effective_chat.id) == self.chat_id

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        msg = (
            "<b>\U0001f916 트레이딩봇 명령어</b>\n\n"
            "/상태 — 현재 자본금, 포지션, MDD\n"
            "/포지션 — 보유 포지션 상세\n"
            "/성과 — 누적 거래 성과\n"
            "/오늘 — 오늘 손익 및 거래 내역\n"
            "/페어 — 현재 트레이딩 페어\n"
            "/중지 — 트레이딩 일시정지\n"
            "/재개 — 트레이딩 재개"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 아직 초기화되지 않았습니다.")
            return

        state = bot.position_tracker.state
        cb = _cb_state_kr(state.circuit_breaker_state)
        s = _sign(state.daily_pnl)
        msg = (
            f"<b>\U0001f4ca 현재 상태</b>\n\n"
            f"자본금: {state.total_capital:,.2f} USDT\n"
            f"가용자본: {state.available_capital:,.2f} USDT\n"
            f"보유 포지션: {state.position_count}개\n"
            f"오늘 손익: {s}{state.daily_pnl:,.2f} USDT\n"
            f"최대낙폭: {state.current_mdd:.1%}\n"
            f"서킷브레이커: {cb}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 아직 초기화되지 않았습니다.")
            return

        positions = bot.position_tracker.state.positions
        if not positions:
            await update.message.reply_text("현재 보유 중인 포지션이 없습니다.")
            return

        msg = f"<b>\U0001f4bc 보유 포지션 ({len(positions)}개)</b>\n\n"
        for p in positions:
            s = _sign(p.unrealized_pnl)
            msg += (
                f"<b>{p.symbol}</b> {_side_kr(p.side)} x{p.leverage:.0f}\n"
                f"  진입가: {p.entry_price:,.4f}\n"
                f"  수량: {p.quantity:.6f}\n"
                f"  손절가: {p.stop_loss:,.4f}\n"
                f"  미실현손익: {s}{p.unrealized_pnl:,.2f} USDT\n"
                f"  전략: {_strategy_kr(p.strategy.value)}\n\n"
            )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 아직 초기화되지 않았습니다.")
            return

        from review.performance import calculate_metrics
        trades = bot.storage.get_trades()
        metrics = calculate_metrics(trades)

        msg = (
            f"<b>\U0001f4c8 누적 성과</b>\n\n"
            f"총 거래: {metrics.total_trades}회\n"
            f"승: {metrics.winning_trades}회 / 패: {metrics.losing_trades}회\n"
            f"승률: {metrics.win_rate:.0%}\n"
            f"평균 수익: {metrics.avg_win:,.2f} USDT\n"
            f"평균 손실: {metrics.avg_loss:,.2f} USDT\n"
            f"손익비: {metrics.profit_factor:.2f}\n"
            f"샤프지수: {metrics.sharpe_ratio:.2f}\n"
            f"순손익: {_sign(metrics.net_pnl)}{metrics.net_pnl:,.2f} USDT\n"
            f"최대낙폭: {metrics.max_drawdown:,.2f} USDT"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 아직 초기화되지 않았습니다.")
            return

        from review.performance import calculate_metrics
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        trades = bot.storage.get_trades(start=today_start)
        state = bot.position_tracker.state

        s = _sign(state.daily_pnl)
        msg = (
            f"<b>\U0001f4c5 오늘 현황</b>\n\n"
            f"오늘 손익: {s}{state.daily_pnl:,.2f} USDT\n"
            f"오늘 거래: {len(trades)}회\n\n"
        )

        if trades:
            for t in trades:
                ts = _sign(t.pnl)
                msg += (
                    f"\u25b8 {t.symbol} {_side_kr(t.side)} "
                    f"| {ts}{t.pnl:,.2f} USDT ({ts}{t.pnl_percent:.1f}%)\n"
                )
        else:
            msg += "오늘 완료된 거래가 없습니다."

        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 아직 초기화되지 않았습니다.")
            return

        bot.circuit_breaker._full_stop = True
        bot.position_tracker.state.circuit_breaker_state = CircuitBreakerState.FULL_STOP
        logger.warning("Trading manually halted via Telegram command")
        await update.message.reply_text(
            "\u26d4 <b>트레이딩이 중지되었습니다.</b>\n"
            "재개하려면 /재개 명령을 보내세요.",
            parse_mode="HTML",
        )

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 아직 초기화되지 않았습니다.")
            return

        bot.circuit_breaker.manual_resume()
        bot.position_tracker.state.circuit_breaker_state = CircuitBreakerState.NORMAL
        logger.info("Trading manually resumed via Telegram command")
        await update.message.reply_text(
            "\u2705 <b>트레이딩이 재개되었습니다.</b>",
            parse_mode="HTML",
        )

    async def _cmd_pairs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 아직 초기화되지 않았습니다.")
            return

        pairs = bot.pair_selector._current_pairs
        if not pairs:
            await update.message.reply_text("현재 선정된 트레이딩 페어가 없습니다.")
            return

        msg = f"<b>\U0001f4b1 트레이딩 페어 ({len(pairs)}개)</b>\n\n"
        for i, p in enumerate(pairs, 1):
            msg += (
                f"{i}. <b>{p.symbol}</b>\n"
                f"   24h 거래량: ${p.volume_24h:,.0f}\n"
                f"   변동성(ATR%): {p.atr_percent:.2%}\n"
                f"   BTC 상관계수: {p.correlation_to_btc:.2f}\n\n"
            )
        await update.message.reply_text(msg, parse_mode="HTML")

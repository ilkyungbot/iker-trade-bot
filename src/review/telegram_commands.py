"""
Telegram command handler — 대화형 시그널 봇.

한글 텍스트 커맨드 + 인라인 버튼 콜백 처리.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

try:
    from telegram import Update, CallbackQuery
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters,
    )
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

from core.types import ConversationState

logger = logging.getLogger(__name__)

# Korean text → handler method name mapping
_KR_COMMANDS: dict[str, str] = {
    "잡았다": "_cmd_entered",
    "팔았다": "_cmd_exited",
    "패스": "_cmd_pass",
    "홀딩": "_cmd_hold",
    "상태": "_cmd_status",
    "성과": "_cmd_performance",
    "도움말": "_cmd_help",
}


class TelegramCommandHandler:
    """대화형 시그널 봇 텔레그램 핸들러."""

    def __init__(self, bot_token: str, chat_id: str):
        if not HAS_TELEGRAM:
            raise ImportError("python-telegram-bot is required")
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._bot_ref = None  # set via attach_bot()
        self._app: Application | None = None

    def attach_bot(self, signal_bot) -> None:
        """SignalBot 참조 연결."""
        self._bot_ref = signal_bot

    async def start(self) -> None:
        if not self.bot_token:
            logger.info("Telegram command handler disabled (no bot token)")
            return

        self._app = Application.builder().token(self.bot_token).build()

        # 슬래시 커맨드
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("performance", self._cmd_performance))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("start", self._cmd_help))

        # 인라인 버튼 콜백
        self._app.add_handler(CallbackQueryHandler(self._callback_handler))

        # 한글 텍스트 라우팅
        self._app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self._route_korean,
        ))

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
        if not self.chat_id:
            return True
        chat = update.effective_chat
        if chat is None:
            return False
        return str(chat.id) == self.chat_id

    async def _route_korean(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        text = (update.message.text or "").strip()
        method_name = _KR_COMMANDS.get(text)
        if method_name:
            handler = getattr(self, method_name)
            await handler(update, context)

    async def _callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """인라인 버튼 콜백 처리."""
        query: CallbackQuery = update.callback_query
        if not query:
            return
        await query.answer()

        if not self._check_auth(update):
            return

        data = query.data
        if data == "entered":
            await self._handle_entered(query)
        elif data == "pass":
            await self._handle_pass(query)
        elif data == "exited":
            await self._handle_exited(query)
        elif data == "hold":
            await self._handle_hold(query)

    async def _handle_entered(self, query: CallbackQuery) -> None:
        bot = self._bot_ref
        if not bot:
            return
        success = bot.state_machine.user_entered(self.chat_id)
        if success:
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(
                "\u2705 <b>진입 확인!</b>\n포지션 모니터링을 시작합니다.\n매도 시 '팔았다' 또는 버튼을 눌러주세요.",
                parse_mode="HTML",
            )
        else:
            await query.message.reply_text("현재 상태에서 진입 확인이 불가합니다.")

    async def _handle_pass(self, query: CallbackQuery) -> None:
        bot = self._bot_ref
        if not bot:
            return
        success = bot.state_machine.user_passed(self.chat_id)
        if success:
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("\u274c 시그널을 패스했습니다. 다음 시그널을 기다립니다.")
        else:
            await query.message.reply_text("현재 상태에서 패스가 불가합니다.")

    async def _handle_exited(self, query: CallbackQuery) -> None:
        bot = self._bot_ref
        if not bot:
            return
        success = bot.state_machine.user_exited(self.chat_id)
        if success:
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("\u2705 <b>청산 확인!</b>\n다음 시그널을 기다립니다.", parse_mode="HTML")
        else:
            await query.message.reply_text("현재 상태에서 청산 확인이 불가합니다.")

    async def _handle_hold(self, query: CallbackQuery) -> None:
        bot = self._bot_ref
        if not bot:
            return
        success = bot.state_machine.user_hold(self.chat_id)
        if success:
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("\u23f3 홀딩 계속! 모니터링을 유지합니다.")
        else:
            await query.message.reply_text("현재 상태에서 홀딩이 불가합니다.")

    # --- 텍스트 커맨드 핸들러 ---

    async def _cmd_entered(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 초기화되지 않았습니다.")
            return
        success = bot.state_machine.user_entered(self.chat_id)
        if success:
            await update.message.reply_text(
                "\u2705 <b>진입 확인!</b>\n포지션 모니터링을 시작합니다.",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text("현재 활성 시그널이 없거나 이미 진입 상태입니다.")

    async def _cmd_exited(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 초기화되지 않았습니다.")
            return
        success = bot.state_machine.user_exited(self.chat_id)
        if success:
            await update.message.reply_text(
                "\u2705 <b>청산 확인!</b>\n다음 시그널을 기다립니다.",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text("현재 모니터링 중인 포지션이 없습니다.")

    async def _cmd_pass(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 초기화되지 않았습니다.")
            return
        success = bot.state_machine.user_passed(self.chat_id)
        if success:
            await update.message.reply_text("\u274c 시그널을 패스했습니다.")
        else:
            await update.message.reply_text("현재 대기 중인 시그널이 없습니다.")

    async def _cmd_hold(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 초기화되지 않았습니다.")
            return
        success = bot.state_machine.user_hold(self.chat_id)
        if success:
            await update.message.reply_text("\u23f3 홀딩 계속! 모니터링을 유지합니다.")
        else:
            await update.message.reply_text("현재 청산 시그널 대기 상태가 아닙니다.")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 초기화되지 않았습니다.")
            return

        session = bot.state_machine.get_session(self.chat_id)
        state_kr = {
            ConversationState.IDLE: "\U0001f7e2 대기 중",
            ConversationState.SIGNAL_SENT: "\U0001f7e1 시그널 발송됨 (응답 대기)",
            ConversationState.MONITORING: "\U0001f4ca 모니터링 중",
            ConversationState.EXIT_SIGNAL_SENT: "\U0001f534 청산 시그널 대기",
        }

        msg = f"<b>\U0001f916 봇 상태</b>\n\n상태: {state_kr.get(session.state, session.state.value)}\n"

        if session.active_signal:
            s = session.active_signal.signal
            msg += (
                f"\n<b>활성 시그널</b>\n"
                f"종목: {s.symbol}\n"
                f"방향: {'롱' if s.action.value == 'enter_long' else '숏'}\n"
                f"진입가: {s.entry_price:,.0f}\n"
                f"손절가: {s.stop_loss:,.0f}\n"
                f"목표가: {s.take_profit:,.0f}\n"
            )
            if session.user_entry_price:
                msg += f"사용자 진입가: {session.user_entry_price:,.0f}\n"

        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 초기화되지 않았습니다.")
            return

        report = bot.signal_tracker.weekly_report()
        msg = bot.reporter.format_weekly_accuracy(report)
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        msg = (
            "<b>\U0001f916 시그널봇 명령어</b>\n\n"
            "<b>한글 입력</b>\n"
            "잡았다 \u2014 시그널 진입 확인\n"
            "팔았다 \u2014 포지션 청산 확인\n"
            "패스 \u2014 시그널 스킵\n"
            "홀딩 \u2014 계속 보유\n"
            "상태 \u2014 현재 봇 상태\n"
            "성과 \u2014 시그널 정확도\n"
            "도움말 \u2014 이 메시지\n\n"
            "<b>슬래시 명령어</b>\n"
            "/status /performance /help"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

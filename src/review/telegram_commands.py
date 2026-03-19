"""
Telegram command handler — 대화형 시그널 봇.

한글 텍스트 커맨드 + 인라인 버튼 콜백 처리.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

try:
    from telegram import Update, CallbackQuery
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters,
    )
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

from core.types import ConversationState, Side

logger = logging.getLogger(__name__)

# Korean text → handler method name mapping
_KR_COMMANDS: dict[str, str] = {
    "상태": "_cmd_status",
    "현황": "_cmd_briefing",
    "성과": "_cmd_performance",
    "도움말": "_cmd_help",
    "신규 포지션": "_cmd_new_position",
    "청산": "_cmd_close_position",
}

_POSITION_FLOW_STEPS = {
    "ask_symbol": "어떤 코인인가요? (예: BTC, ETH, SOL)",
    "ask_side": "롱인가요, 숏인가요? (롱/숏)",
    "ask_entry": "평단가를 입력해주세요. (숫자만)",
    "ask_leverage": "레버리지 배율을 입력해주세요. (예: 5)",
    "ask_margin": "투입 마진을 입력해주세요. (USDT, 예: 500)",
    "ask_stop_loss": "손절가를 입력해주세요.",  # Bot will show ATR guide
    "ask_take_profit": "익절가를 입력해주세요.",  # Bot will show R:R guide
    "ask_reason": "진입 논리를 한줄로 입력해주세요. (예: 200EMA 반등 + 펀딩 음전환)",
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

        # 대화 흐름 중이면 흐름 처리 우선 (단, 명령어나 "취소"면 흐름 탈출)
        flow_step = context.user_data.get("position_flow")
        if flow_step:
            if text == "취소":
                context.user_data.pop("position_flow", None)
                context.user_data.pop("position_data", None)
                await update.message.reply_text("포지션 등록을 취소했습니다.")
                return
            # 기존 명령어면 흐름 중단 후 명령 실행
            if text in _KR_COMMANDS:
                context.user_data.pop("position_flow", None)
                context.user_data.pop("position_data", None)
                handler = getattr(self, _KR_COMMANDS[text])
                await handler(update, context)
                return
            await self._handle_position_flow(update, context, text)
            return

        method_name = _KR_COMMANDS.get(text)
        if method_name:
            handler = getattr(self, method_name)
            await handler(update, context)
            return

        # 티커 심볼 입력 감지 (영문 1~10자, 예: SOL, BTC, ETH)
        upper = text.upper()
        if upper.isalpha() and 1 <= len(upper) <= 10:
            await self._cmd_analyze_coin(update, context, upper)

    async def _callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """인라인 버튼 콜백 처리 (레거시 버튼은 무시)."""
        query: CallbackQuery = update.callback_query
        if not query:
            return
        await query.answer()

        if not self._check_auth(update):
            return

    async def _cmd_analyze_coin(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str = "") -> None:
        """티커 심볼 입력 → 코인 심층 분석."""
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 초기화되지 않았습니다.")
            return

        if not symbol:
            await update.message.reply_text("심볼을 입력해주세요. (예: SOL, BTC, ETH)")
            return

        await update.message.reply_text(f"\U0001f50d {symbol} 분석 중...")
        try:
            analysis = await bot.analyze_coin(symbol)
            if analysis is None:
                await update.message.reply_text(f"'{symbol}' 코인을 찾을 수 없거나 데이터가 부족합니다.")
                return
            text = bot.reporter.format_coin_analysis(analysis)
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Coin analysis error: {e}", exc_info=True)
            await update.message.reply_text(f"분석 실패: {e}")

    async def _cmd_briefing(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """현황 — 포지션 대시보드 (없으면 시장 브리핑)."""
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot:
            await update.message.reply_text("봇이 초기화되지 않았습니다.")
            return

        # Position dashboard first
        if hasattr(bot, "position_manager"):
            positions = bot.position_manager.get_active_positions(self.chat_id)
            if positions:
                price_data = {}
                for pos in positions:
                    try:
                        now = datetime.now(timezone.utc)
                        candles = await bot._run_sync(
                            bot.collector.get_candles, pos.symbol, "1",
                            start_time=now - timedelta(minutes=5),
                        )
                        if candles:
                            cp = candles[-1].close
                            pnl_pct = (cp - pos.entry_price) / pos.entry_price * 100
                            if pos.side == Side.SHORT:
                                pnl_pct = -pnl_pct
                            pnl_pct *= pos.leverage
                            pnl_usdt = (pos.margin_usdt * pnl_pct / 100) if pos.margin_usdt else None
                            price_data[pos.symbol] = {"current_price": cp, "pnl_pct": pnl_pct, "pnl_usdt": pnl_usdt}
                    except Exception:
                        pass

                dashboard = bot.reporter.format_position_dashboard(positions, price_data)
                await update.message.reply_text(dashboard, parse_mode="HTML")
                return

        # If no positions, show market briefing
        await update.message.reply_text("\U0001f50d 시장 스캔 중...")
        try:
            briefing = await bot.generate_briefing()
            text = bot.reporter.format_hourly_briefing(briefing)
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"브리핑 생성 실패: {e}")

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
            ConversationState.MONITORING: "\U0001f4ca 모니터링 중",
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

        if hasattr(bot, "position_manager"):
            positions = bot.position_manager.get_active_positions(self.chat_id)
            if positions:
                msg += "\n<b>\U0001f4cb 수동 포지션</b>\n"
                for p in positions:
                    side_kr = "롱" if p.side == Side.LONG else "숏"
                    msg += f"• {p.symbol} ({side_kr} {p.leverage}x) 평단 {p.entry_price:,.2f}\n"

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
            "<b>포지션 관리</b>\n"
            "신규 포지션 \u2014 수동 포지션 등록\n"
            "청산 \u2014 포지션 청산\n"
            "취소 \u2014 등록 취소\n\n"
            "<b>조회</b>\n"
            "현황 \u2014 내 포지션 대시보드\n"
            "상태 \u2014 봇 시스템 상태\n"
            "성과 \u2014 트레이딩 리포트\n"
            "SOL, BTC 등 \u2014 코인 심층분석\n"
            "도움말 \u2014 이 메시지\n\n"
            "<b>슬래시 명령어</b>\n"
            "/status /performance /help"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_new_position(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """신규 포지션 등록 시작."""
        if not self._check_auth(update):
            return
        context.user_data["position_flow"] = "ask_symbol"
        context.user_data["position_data"] = {}
        await update.message.reply_text(
            "\U0001f4dd <b>신규 포지션 등록</b>\n\n" + _POSITION_FLOW_STEPS["ask_symbol"],
            parse_mode="HTML",
        )

    async def _cmd_close_position(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """포지션 청산."""
        if not self._check_auth(update):
            return
        bot = self._bot_ref
        if not bot or not hasattr(bot, "position_manager"):
            await update.message.reply_text("봇이 초기화되지 않았습니다.")
            return

        positions = bot.position_manager.get_active_positions(self.chat_id)
        if not positions:
            await update.message.reply_text("활성 포지션이 없습니다.")
            return

        if len(positions) == 1:
            pos = positions[0]
            final_pnl = None
            try:
                now = datetime.now(timezone.utc)
                candles = await bot._run_sync(
                    bot.collector.get_candles, pos.symbol, "1",
                    start_time=now - timedelta(minutes=5),
                )
                if candles:
                    current_price = candles[-1].close
                    price_change = (current_price - pos.entry_price) / pos.entry_price * 100
                    if pos.side == Side.SHORT:
                        price_change = -price_change
                    final_pnl = price_change * pos.leverage
            except Exception:
                pass

            bot.position_manager.close_position(pos.id, self.chat_id)
            if hasattr(bot, "position_monitor"):
                bot.position_monitor.clear_position(pos.id)
            text = bot.reporter.format_position_closed(pos, final_pnl)
            await update.message.reply_text(text, parse_mode="HTML")
        else:
            context.user_data["position_flow"] = "ask_close_which"
            lines = ["\U0001f4dd <b>어떤 포지션을 청산할까요?</b>\n"]
            for p in positions:
                side_kr = "롱" if p.side == Side.LONG else "숏"
                lines.append(f"• {p.symbol} ({side_kr} {p.leverage}x) — 평단 {p.entry_price:,.2f}")
            lines.append("\n코인명을 입력해주세요. (예: BTC)")
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _handle_position_flow(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
        """신규 포지션 대화 흐름 단계별 처리."""
        step = context.user_data.get("position_flow")
        data = context.user_data.get("position_data", {})
        bot = self._bot_ref

        if step == "ask_symbol":
            symbol = text.strip().upper()
            if not symbol or not symbol.replace("USDT", "").isalpha():
                await update.message.reply_text("올바른 코인명을 입력해주세요. (예: BTC, ETH, SOL)")
                return
            if not symbol.endswith("USDT"):
                symbol += "USDT"
            data["symbol"] = symbol
            context.user_data["position_data"] = data
            context.user_data["position_flow"] = "ask_side"
            await update.message.reply_text(_POSITION_FLOW_STEPS["ask_side"])

        elif step == "ask_side":
            text_lower = text.strip().lower()
            if text_lower in ("롱", "long", "매수"):
                data["side"] = Side.LONG
            elif text_lower in ("숏", "short", "매도"):
                data["side"] = Side.SHORT
            else:
                await update.message.reply_text("'롱' 또는 '숏'으로 입력해주세요.")
                return
            context.user_data["position_data"] = data
            context.user_data["position_flow"] = "ask_entry"
            await update.message.reply_text(_POSITION_FLOW_STEPS["ask_entry"])

        elif step == "ask_entry":
            try:
                entry_price = float(text.strip().replace(",", ""))
                if entry_price <= 0:
                    raise ValueError
            except ValueError:
                await update.message.reply_text("올바른 숫자를 입력해주세요.")
                return
            data["entry_price"] = entry_price
            context.user_data["position_data"] = data
            context.user_data["position_flow"] = "ask_leverage"
            await update.message.reply_text(_POSITION_FLOW_STEPS["ask_leverage"])

        elif step == "ask_leverage":
            try:
                leverage = float(text.strip().replace("배", "").replace("x", ""))
                if leverage <= 0 or leverage > 125:
                    raise ValueError
            except ValueError:
                await update.message.reply_text("1~125 사이 숫자를 입력해주세요.")
                return
            data["leverage"] = leverage
            context.user_data["position_data"] = data
            context.user_data["position_flow"] = "ask_margin"
            await update.message.reply_text(_POSITION_FLOW_STEPS["ask_margin"])

        elif step == "ask_margin":
            try:
                margin = float(text.strip().replace(",", ""))
                if margin <= 0:
                    raise ValueError
            except ValueError:
                await update.message.reply_text("올바른 숫자를 입력해주세요. (예: 500)")
                return
            data["margin_usdt"] = margin
            context.user_data["position_data"] = data
            context.user_data["position_flow"] = "ask_stop_loss"

            # Show ATR-based SL suggestion
            sl_guide = ""
            if bot:
                try:
                    from execution.risk_calculator import suggest_stop_loss
                    now = datetime.now(timezone.utc)
                    candles = await bot._run_sync(
                        bot.collector.get_candles, data["symbol"], "240",
                        start_time=now - timedelta(days=30),
                    )
                    if candles and len(candles) > 14:
                        from data.features import candles_to_dataframe, add_all_features
                        df = candles_to_dataframe(candles)
                        df = add_all_features(df)
                        atr = float(df.iloc[-1].get("atr", 0))
                        if atr > 0:
                            suggested_sl = suggest_stop_loss(data["entry_price"], data["side"], atr, data["leverage"])
                            from review.reporter import _format_price
                            leveraged_loss = abs(data["entry_price"] - suggested_sl) / data["entry_price"] * 100 * data["leverage"]
                            liq_price = (
                                data["entry_price"] * (1 - 1 / data["leverage"])
                                if data["side"].value == "long"
                                else data["entry_price"] * (1 + 1 / data["leverage"])
                            )
                            sl_guide = (
                                f"\n\U0001f4cc <b>참고 가이드:</b>"
                                f"\n\u2022 ATR 기반 추천 손절: {_format_price(suggested_sl)} (레버리지 반영 -{leveraged_loss:.1f}%)"
                                f"\n\u2022 {data['leverage']}x 청산가: {_format_price(liq_price)}"
                            )
                except Exception:
                    pass

            await update.message.reply_text(
                _POSITION_FLOW_STEPS["ask_stop_loss"] + sl_guide,
                parse_mode="HTML",
            )

        elif step == "ask_stop_loss":
            try:
                sl = float(text.strip().replace(",", ""))
                if sl <= 0:
                    raise ValueError
                if data["side"] == Side.LONG and sl >= data["entry_price"]:
                    await update.message.reply_text("롱 포지션의 손절가는 평단가보다 낮아야 합니다.")
                    return
                if data["side"] == Side.SHORT and sl <= data["entry_price"]:
                    await update.message.reply_text("숏 포지션의 손절가는 평단가보다 높아야 합니다.")
                    return
            except ValueError:
                await update.message.reply_text("올바른 숫자를 입력해주세요.")
                return
            data["stop_loss"] = sl
            context.user_data["position_data"] = data
            context.user_data["position_flow"] = "ask_take_profit"

            # Show R:R guide
            from execution.risk_calculator import suggest_take_profit
            from review.reporter import _format_price
            suggested_tp = suggest_take_profit(data["entry_price"], data["side"], sl)
            tp_guide = f"\n\U0001f4cc <b>참고:</b>\n\u2022 R:R 2:1 기준 익절가: {_format_price(suggested_tp)}"

            await update.message.reply_text(
                _POSITION_FLOW_STEPS["ask_take_profit"] + tp_guide,
                parse_mode="HTML",
            )

        elif step == "ask_take_profit":
            try:
                tp = float(text.strip().replace(",", ""))
                if tp <= 0:
                    raise ValueError
                if data["side"] == Side.LONG and tp <= data["entry_price"]:
                    await update.message.reply_text("롱 포지션의 익절가는 평단가보다 높아야 합니다.")
                    return
                if data["side"] == Side.SHORT and tp >= data["entry_price"]:
                    await update.message.reply_text("숏 포지션의 익절가는 평단가보다 낮아야 합니다.")
                    return
            except ValueError:
                await update.message.reply_text("올바른 숫자를 입력해주세요.")
                return
            data["take_profit"] = tp
            context.user_data["position_data"] = data
            context.user_data["position_flow"] = "ask_reason"

            # Show R:R validation
            from execution.risk_calculator import calculate_rr_ratio, validate_position
            rr = calculate_rr_ratio(data["entry_price"], data["stop_loss"], tp, data["side"])
            validation = validate_position(
                data["entry_price"], data["stop_loss"], tp,
                data["leverage"], data["margin_usdt"], data["side"],
            )

            rr_msg = f"\nR:R = 1:{rr:.1f}"
            if rr < 1.5:
                rr_msg += " \u26a0\ufe0f (1.5 미만 \u2014 권장하지 않음)"
            else:
                rr_msg += " \u2705"

            if validation.get("max_loss_usdt"):
                rr_msg += f"\n최대 손실: {validation['max_loss_usdt']:.1f} USDT ({validation['max_loss_pct']:.1f}%)"

            await update.message.reply_text(
                rr_msg + "\n\n" + _POSITION_FLOW_STEPS["ask_reason"],
                parse_mode="HTML",
            )

        elif step == "ask_reason":
            reason = text.strip()
            if not reason:
                reason = "(미입력)"
            data["entry_reason"] = reason

            # Register position
            if bot and hasattr(bot, "position_manager"):
                pos = bot.position_manager.open_position(
                    chat_id=self.chat_id,
                    symbol=data["symbol"],
                    side=data["side"],
                    entry_price=data["entry_price"],
                    leverage=data["leverage"],
                    stop_loss=data.get("stop_loss"),
                    take_profit=data.get("take_profit"),
                    margin_usdt=data.get("margin_usdt"),
                    entry_reason=data.get("entry_reason", ""),
                )

                # Record in trading journal
                if hasattr(bot, "trading_journal"):
                    bot.trading_journal.record_entry(pos)

                # Check portfolio guard
                if hasattr(bot, "portfolio_guard"):
                    positions = bot.position_manager.get_active_positions(self.chat_id)
                    result = bot.portfolio_guard.check_can_open(positions[:-1], data["symbol"])
                    if not result.allowed:
                        bot.position_manager.close_position(pos.id, self.chat_id)
                        await update.message.reply_text(f"\u274c 진입 거부: {result.reason}")
                        context.user_data.pop("position_flow", None)
                        context.user_data.pop("position_data", None)
                        return

                text_msg = bot.reporter.format_position_registered(pos)
                await update.message.reply_text(text_msg, parse_mode="HTML")
            else:
                await update.message.reply_text("포지션 매니저가 초기화되지 않았습니다.")

            context.user_data.pop("position_flow", None)
            context.user_data.pop("position_data", None)

        elif step == "ask_close_which":
            symbol = text.strip().upper()
            if not symbol.endswith("USDT"):
                symbol += "USDT"
            if bot and hasattr(bot, "position_manager"):
                positions = bot.position_manager.get_active_positions(self.chat_id)
                target = next((p for p in positions if p.symbol == symbol), None)
                if target:
                    # PnL 계산
                    final_pnl = None
                    try:
                        now = datetime.now(timezone.utc)
                        candles = await bot._run_sync(
                            bot.collector.get_candles, target.symbol, "1",
                            start_time=now - timedelta(minutes=5),
                        )
                        if candles:
                            current_price = candles[-1].close
                            price_change = (current_price - target.entry_price) / target.entry_price * 100
                            if target.side == Side.SHORT:
                                price_change = -price_change
                            final_pnl = price_change * target.leverage
                    except Exception:
                        pass

                    bot.position_manager.close_position(target.id, self.chat_id)
                    if hasattr(bot, "position_monitor"):
                        bot.position_monitor.clear_position(target.id)
                    text_msg = bot.reporter.format_position_closed(target, final_pnl)
                    await update.message.reply_text(text_msg, parse_mode="HTML")
                else:
                    await update.message.reply_text(f"{symbol} 포지션을 찾을 수 없습니다. 다시 입력해주세요.")
                    return  # Don't clear flow — let user retry
            context.user_data.pop("position_flow", None)
            context.user_data.pop("position_data", None)

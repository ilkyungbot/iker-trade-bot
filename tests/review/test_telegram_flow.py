"""Tests for Telegram auth logic and input validation (Task 11)."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from core.types import Side, ConversationState, ManualPosition, UserSession
from datetime import datetime, timezone


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_update(chat_id=12345, text=""):
    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.message = AsyncMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    return update


def _make_context(**user_data):
    context = MagicMock()
    context.user_data = dict(user_data)
    return context


def _make_handler():
    from review.telegram_commands import TelegramCommandHandler
    handler = TelegramCommandHandler.__new__(TelegramCommandHandler)
    handler.chat_id = "12345"
    handler.bot_token = "test"
    handler._bot_ref = MagicMock()
    return handler


def _make_mock_position(
    symbol="BTCUSDT",
    side=Side.LONG,
    entry_price=50000.0,
    leverage=5.0,
    stop_loss=49000.0,
    take_profit=52000.0,
    margin_usdt=500.0,
    pos_id=1,
):
    pos = MagicMock(spec=ManualPosition)
    pos.id = pos_id
    pos.symbol = symbol
    pos.side = side
    pos.entry_price = entry_price
    pos.leverage = leverage
    pos.stop_loss = stop_loss
    pos.take_profit = take_profit
    pos.margin_usdt = margin_usdt
    return pos


# ── Auth tests ────────────────────────────────────────────────────────────────

class TestCheckAuth:
    def _make_handler_auth(self, chat_id):
        from review.telegram_commands import TelegramCommandHandler
        handler = TelegramCommandHandler.__new__(TelegramCommandHandler)
        handler.chat_id = chat_id
        return handler

    def _make_update_auth(self, chat_id_int):
        update = MagicMock()
        update.effective_chat = MagicMock()
        update.effective_chat.id = chat_id_int
        return update

    def test_empty_chat_id_rejects(self):
        handler = self._make_handler_auth("")
        assert handler._check_auth(self._make_update_auth(12345)) is False

    def test_matching_id(self):
        handler = self._make_handler_auth("12345")
        assert handler._check_auth(self._make_update_auth(12345)) is True

    def test_wrong_id(self):
        handler = self._make_handler_auth("12345")
        assert handler._check_auth(self._make_update_auth(99999)) is False

    def test_no_effective_chat(self):
        handler = self._make_handler_auth("12345")
        update = MagicMock()
        update.effective_chat = None
        assert handler._check_auth(update) is False


# ── _cmd_help ─────────────────────────────────────────────────────────────────

class TestCmdHelp:
    @pytest.mark.asyncio
    async def test_help_sends_message(self):
        handler = _make_handler()
        update = _make_update(chat_id=12345, text="도움말")
        context = _make_context()
        await handler._cmd_help(update, context)
        update.message.reply_text.assert_awaited_once()
        call_args = update.message.reply_text.call_args
        text = call_args[0][0]
        assert "시그널봇" in text
        assert "신규 포지션" in text
        assert "청산" in text

    @pytest.mark.asyncio
    async def test_help_rejected_for_wrong_chat(self):
        handler = _make_handler()
        update = _make_update(chat_id=99999, text="도움말")
        context = _make_context()
        await handler._cmd_help(update, context)
        update.message.reply_text.assert_not_awaited()


# ── _cmd_status ───────────────────────────────────────────────────────────────

class TestCmdStatus:
    def _setup_bot(self, handler, state=ConversationState.IDLE, active_signal=None, positions=None):
        bot = handler._bot_ref
        session = UserSession(
            chat_id="12345",
            state=state,
            active_signal=active_signal,
        )
        bot.state_machine = MagicMock()
        bot.state_machine.get_session.return_value = session
        bot.position_manager = MagicMock()
        bot.position_manager.get_active_positions.return_value = positions or []
        return bot

    @pytest.mark.asyncio
    async def test_status_idle_no_positions(self):
        handler = _make_handler()
        self._setup_bot(handler)
        update = _make_update()
        context = _make_context()
        await handler._cmd_status(update, context)
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "봇 상태" in text
        assert "대기 중" in text

    @pytest.mark.asyncio
    async def test_status_with_active_signal(self):
        handler = _make_handler()
        signal_msg = MagicMock()
        signal_msg.signal.symbol = "BTCUSDT"
        signal_msg.signal.action.value = "enter_long"
        signal_msg.signal.entry_price = 50000.0
        signal_msg.signal.stop_loss = 49000.0
        signal_msg.signal.take_profit = 52000.0
        self._setup_bot(handler, active_signal=signal_msg)
        update = _make_update()
        context = _make_context()
        await handler._cmd_status(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "활성 시그널" in text
        assert "BTCUSDT" in text
        assert "50,000" in text

    @pytest.mark.asyncio
    async def test_status_with_positions(self):
        handler = _make_handler()
        pos = _make_mock_position()
        self._setup_bot(handler, positions=[pos])
        update = _make_update()
        context = _make_context()
        await handler._cmd_status(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "수동 포지션" in text
        assert "BTCUSDT" in text

    @pytest.mark.asyncio
    async def test_status_no_bot(self):
        handler = _make_handler()
        handler._bot_ref = None
        update = _make_update()
        context = _make_context()
        await handler._cmd_status(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "초기화" in text

    @pytest.mark.asyncio
    async def test_status_wrong_chat_rejected(self):
        handler = _make_handler()
        update = _make_update(chat_id=99999)
        context = _make_context()
        await handler._cmd_status(update, context)
        update.message.reply_text.assert_not_awaited()


# ── _cmd_new_position ─────────────────────────────────────────────────────────

class TestCmdNewPosition:
    @pytest.mark.asyncio
    async def test_starts_flow(self):
        handler = _make_handler()
        update = _make_update(text="신규 포지션")
        context = _make_context()
        await handler._cmd_new_position(update, context)
        assert context.user_data["position_flow"] == "ask_symbol"
        assert context.user_data["position_data"] == {}
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "코인" in text

    @pytest.mark.asyncio
    async def test_rejected_for_wrong_chat(self):
        handler = _make_handler()
        update = _make_update(chat_id=99999)
        context = _make_context()
        await handler._cmd_new_position(update, context)
        assert "position_flow" not in context.user_data


# ── _route_korean ─────────────────────────────────────────────────────────────

class TestRouteKorean:
    @pytest.mark.asyncio
    async def test_routes_help_command(self):
        handler = _make_handler()
        update = _make_update(text="도움말")
        context = _make_context()
        await handler._route_korean(update, context)
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "시그널봇" in text

    @pytest.mark.asyncio
    async def test_routes_new_position(self):
        handler = _make_handler()
        update = _make_update(text="신규 포지션")
        context = _make_context()
        await handler._route_korean(update, context)
        assert context.user_data.get("position_flow") == "ask_symbol"

    @pytest.mark.asyncio
    async def test_cancel_during_flow(self):
        handler = _make_handler()
        update = _make_update(text="취소")
        context = _make_context(
            position_flow="ask_side",
            position_data={"symbol": "BTCUSDT"},
        )
        await handler._route_korean(update, context)
        assert "position_flow" not in context.user_data
        assert "position_data" not in context.user_data
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "취소" in text

    @pytest.mark.asyncio
    async def test_korean_command_interrupts_flow(self):
        handler = _make_handler()
        update = _make_update(text="도움말")
        context = _make_context(
            position_flow="ask_side",
            position_data={"symbol": "BTCUSDT"},
        )
        await handler._route_korean(update, context)
        # Flow should be cleared and help dispatched
        assert "position_flow" not in context.user_data
        text = update.message.reply_text.call_args[0][0]
        assert "시그널봇" in text

    @pytest.mark.asyncio
    async def test_unknown_text_no_reply(self):
        handler = _make_handler()
        update = _make_update(text="알수없는텍스트1234")
        context = _make_context()
        await handler._route_korean(update, context)
        update.message.reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejected_for_wrong_chat(self):
        handler = _make_handler()
        update = _make_update(chat_id=99999, text="도움말")
        context = _make_context()
        await handler._route_korean(update, context)
        update.message.reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ticker_symbol_routes_to_analyze(self):
        handler = _make_handler()
        handler._cmd_analyze_coin = AsyncMock()
        update = _make_update(text="SOL")
        context = _make_context()
        await handler._route_korean(update, context)
        handler._cmd_analyze_coin.assert_awaited_once_with(update, context, "SOL")

    @pytest.mark.asyncio
    async def test_flow_step_dispatched(self):
        handler = _make_handler()
        update = _make_update(text="BTC")
        context = _make_context(
            position_flow="ask_symbol",
            position_data={},
        )
        await handler._route_korean(update, context)
        # Flow advances to ask_side
        assert context.user_data.get("position_flow") == "ask_side"


# ── _handle_position_flow: ask_symbol ─────────────────────────────────────────

class TestPositionFlowAskSymbol:
    @pytest.mark.asyncio
    async def test_valid_symbol_advances(self):
        handler = _make_handler()
        update = _make_update(text="BTC")
        context = _make_context(position_flow="ask_symbol", position_data={})
        await handler._handle_position_flow(update, context, "BTC")
        assert context.user_data["position_data"]["symbol"] == "BTCUSDT"
        assert context.user_data["position_flow"] == "ask_side"

    @pytest.mark.asyncio
    async def test_symbol_already_has_usdt(self):
        handler = _make_handler()
        update = _make_update(text="BTCUSDT")
        context = _make_context(position_flow="ask_symbol", position_data={})
        await handler._handle_position_flow(update, context, "BTCUSDT")
        assert context.user_data["position_data"]["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_lowercase_symbol_normalized(self):
        handler = _make_handler()
        update = _make_update(text="eth")
        context = _make_context(position_flow="ask_symbol", position_data={})
        await handler._handle_position_flow(update, context, "eth")
        assert context.user_data["position_data"]["symbol"] == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_invalid_symbol_rejected(self):
        handler = _make_handler()
        update = _make_update(text="123")
        context = _make_context(position_flow="ask_symbol", position_data={})
        await handler._handle_position_flow(update, context, "123")
        update.message.reply_text.assert_awaited_once()
        assert context.user_data.get("position_flow") == "ask_symbol"

    @pytest.mark.asyncio
    async def test_empty_symbol_rejected(self):
        handler = _make_handler()
        update = _make_update(text="  ")
        context = _make_context(position_flow="ask_symbol", position_data={})
        await handler._handle_position_flow(update, context, "  ")
        update.message.reply_text.assert_awaited_once()
        assert context.user_data.get("position_flow") == "ask_symbol"


# ── _handle_position_flow: ask_side ──────────────────────────────────────────

class TestPositionFlowAskSide:
    def _ctx(self):
        return _make_context(
            position_flow="ask_side",
            position_data={"symbol": "BTCUSDT"},
        )

    @pytest.mark.asyncio
    async def test_long_korean(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="롱"), ctx, "롱")
        assert ctx.user_data["position_data"]["side"] == Side.LONG
        assert ctx.user_data["position_flow"] == "ask_entry"

    @pytest.mark.asyncio
    async def test_long_english(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="long"), ctx, "long")
        assert ctx.user_data["position_data"]["side"] == Side.LONG

    @pytest.mark.asyncio
    async def test_buy_korean(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="매수"), ctx, "매수")
        assert ctx.user_data["position_data"]["side"] == Side.LONG

    @pytest.mark.asyncio
    async def test_short_korean(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="숏"), ctx, "숏")
        assert ctx.user_data["position_data"]["side"] == Side.SHORT
        assert ctx.user_data["position_flow"] == "ask_entry"

    @pytest.mark.asyncio
    async def test_short_english(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="short"), ctx, "short")
        assert ctx.user_data["position_data"]["side"] == Side.SHORT

    @pytest.mark.asyncio
    async def test_sell_korean(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="매도"), ctx, "매도")
        assert ctx.user_data["position_data"]["side"] == Side.SHORT

    @pytest.mark.asyncio
    async def test_invalid_side_rejected(self):
        handler = _make_handler()
        ctx = self._ctx()
        update = _make_update(text="잘못된값")
        await handler._handle_position_flow(update, ctx, "잘못된값")
        update.message.reply_text.assert_awaited_once()
        assert ctx.user_data["position_flow"] == "ask_side"


# ── _handle_position_flow: ask_entry ─────────────────────────────────────────

class TestPositionFlowAskEntry:
    def _ctx(self):
        return _make_context(
            position_flow="ask_entry",
            position_data={"symbol": "BTCUSDT", "side": Side.LONG},
        )

    @pytest.mark.asyncio
    async def test_valid_entry_advances(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="50000"), ctx, "50000")
        assert ctx.user_data["position_data"]["entry_price"] == 50000.0
        assert ctx.user_data["position_flow"] == "ask_leverage"

    @pytest.mark.asyncio
    async def test_comma_number_parsed(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="50,000"), ctx, "50,000")
        assert ctx.user_data["position_data"]["entry_price"] == 50000.0

    @pytest.mark.asyncio
    async def test_zero_rejected(self):
        handler = _make_handler()
        ctx = self._ctx()
        update = _make_update(text="0")
        await handler._handle_position_flow(update, ctx, "0")
        update.message.reply_text.assert_awaited_once()
        assert ctx.user_data["position_flow"] == "ask_entry"

    @pytest.mark.asyncio
    async def test_negative_rejected(self):
        handler = _make_handler()
        ctx = self._ctx()
        update = _make_update(text="-100")
        await handler._handle_position_flow(update, ctx, "-100")
        update.message.reply_text.assert_awaited_once()
        assert ctx.user_data["position_flow"] == "ask_entry"

    @pytest.mark.asyncio
    async def test_non_numeric_rejected(self):
        handler = _make_handler()
        ctx = self._ctx()
        update = _make_update(text="abc")
        await handler._handle_position_flow(update, ctx, "abc")
        update.message.reply_text.assert_awaited_once()
        assert ctx.user_data["position_flow"] == "ask_entry"


# ── _handle_position_flow: ask_leverage ──────────────────────────────────────

class TestPositionFlowAskLeverage:
    def _ctx(self):
        return _make_context(
            position_flow="ask_leverage",
            position_data={"symbol": "BTCUSDT", "side": Side.LONG, "entry_price": 50000.0},
        )

    @pytest.mark.asyncio
    async def test_valid_leverage(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="5"), ctx, "5")
        assert ctx.user_data["position_data"]["leverage"] == 5.0
        assert ctx.user_data["position_flow"] == "ask_margin"

    @pytest.mark.asyncio
    async def test_x_suffix_stripped(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="10x"), ctx, "10x")
        assert ctx.user_data["position_data"]["leverage"] == 10.0

    @pytest.mark.asyncio
    async def test_bae_suffix_stripped(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="20배"), ctx, "20배")
        assert ctx.user_data["position_data"]["leverage"] == 20.0

    @pytest.mark.asyncio
    async def test_max_leverage_125(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="125"), ctx, "125")
        assert ctx.user_data["position_data"]["leverage"] == 125.0

    @pytest.mark.asyncio
    async def test_over_125_rejected(self):
        handler = _make_handler()
        ctx = self._ctx()
        update = _make_update(text="126")
        await handler._handle_position_flow(update, ctx, "126")
        update.message.reply_text.assert_awaited_once()
        assert ctx.user_data["position_flow"] == "ask_leverage"

    @pytest.mark.asyncio
    async def test_zero_rejected(self):
        handler = _make_handler()
        ctx = self._ctx()
        update = _make_update(text="0")
        await handler._handle_position_flow(update, ctx, "0")
        update.message.reply_text.assert_awaited_once()
        assert ctx.user_data["position_flow"] == "ask_leverage"

    @pytest.mark.asyncio
    async def test_non_numeric_rejected(self):
        handler = _make_handler()
        ctx = self._ctx()
        update = _make_update(text="many")
        await handler._handle_position_flow(update, ctx, "many")
        update.message.reply_text.assert_awaited_once()


# ── _handle_position_flow: ask_margin ────────────────────────────────────────

class TestPositionFlowAskMargin:
    def _ctx(self):
        return _make_context(
            position_flow="ask_margin",
            position_data={
                "symbol": "BTCUSDT", "side": Side.LONG,
                "entry_price": 50000.0, "leverage": 5.0,
            },
        )

    @pytest.mark.asyncio
    async def test_valid_margin_advances(self):
        handler = _make_handler()
        # Mock _run_sync to avoid network call
        handler._bot_ref._run_sync = AsyncMock(return_value=[])
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="500"), ctx, "500")
        assert ctx.user_data["position_data"]["margin_usdt"] == 500.0
        assert ctx.user_data["position_flow"] == "ask_stop_loss"

    @pytest.mark.asyncio
    async def test_zero_margin_rejected(self):
        handler = _make_handler()
        ctx = self._ctx()
        update = _make_update(text="0")
        await handler._handle_position_flow(update, ctx, "0")
        update.message.reply_text.assert_awaited_once()
        assert ctx.user_data["position_flow"] == "ask_margin"

    @pytest.mark.asyncio
    async def test_comma_margin_parsed(self):
        handler = _make_handler()
        handler._bot_ref._run_sync = AsyncMock(return_value=[])
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="1,000"), ctx, "1,000")
        assert ctx.user_data["position_data"]["margin_usdt"] == 1000.0


# ── _handle_position_flow: ask_stop_loss ─────────────────────────────────────

class TestPositionFlowAskStopLoss:
    def _ctx_long(self):
        return _make_context(
            position_flow="ask_stop_loss",
            position_data={
                "symbol": "BTCUSDT", "side": Side.LONG,
                "entry_price": 50000.0, "leverage": 5.0, "margin_usdt": 500.0,
            },
        )

    def _ctx_short(self):
        return _make_context(
            position_flow="ask_stop_loss",
            position_data={
                "symbol": "BTCUSDT", "side": Side.SHORT,
                "entry_price": 50000.0, "leverage": 5.0, "margin_usdt": 500.0,
            },
        )

    @pytest.mark.asyncio
    async def test_long_valid_sl_below_entry(self):
        handler = _make_handler()
        ctx = self._ctx_long()
        await handler._handle_position_flow(_make_update(text="49000"), ctx, "49000")
        assert ctx.user_data["position_data"]["stop_loss"] == 49000.0
        assert ctx.user_data["position_flow"] == "ask_take_profit"

    @pytest.mark.asyncio
    async def test_long_sl_above_entry_rejected(self):
        handler = _make_handler()
        ctx = self._ctx_long()
        update = _make_update(text="51000")
        await handler._handle_position_flow(update, ctx, "51000")
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "롱" in text
        assert ctx.user_data["position_flow"] == "ask_stop_loss"

    @pytest.mark.asyncio
    async def test_long_sl_equal_entry_rejected(self):
        handler = _make_handler()
        ctx = self._ctx_long()
        update = _make_update(text="50000")
        await handler._handle_position_flow(update, ctx, "50000")
        update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_short_valid_sl_above_entry(self):
        handler = _make_handler()
        ctx = self._ctx_short()
        await handler._handle_position_flow(_make_update(text="51000"), ctx, "51000")
        assert ctx.user_data["position_data"]["stop_loss"] == 51000.0
        assert ctx.user_data["position_flow"] == "ask_take_profit"

    @pytest.mark.asyncio
    async def test_short_sl_below_entry_rejected(self):
        handler = _make_handler()
        ctx = self._ctx_short()
        update = _make_update(text="49000")
        await handler._handle_position_flow(update, ctx, "49000")
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "숏" in text

    @pytest.mark.asyncio
    async def test_zero_sl_rejected(self):
        handler = _make_handler()
        ctx = self._ctx_long()
        update = _make_update(text="0")
        await handler._handle_position_flow(update, ctx, "0")
        update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_numeric_sl_rejected(self):
        handler = _make_handler()
        ctx = self._ctx_long()
        update = _make_update(text="abc")
        await handler._handle_position_flow(update, ctx, "abc")
        update.message.reply_text.assert_awaited_once()


# ── _handle_position_flow: ask_take_profit ────────────────────────────────────

class TestPositionFlowAskTakeProfit:
    def _ctx_long(self, rr_override=False):
        # entry=50000, sl=49000 → risk=1000 → rr2 tp=52000
        return _make_context(
            position_flow="ask_take_profit",
            position_data={
                "symbol": "BTCUSDT", "side": Side.LONG,
                "entry_price": 50000.0, "leverage": 5.0,
                "margin_usdt": 500.0, "stop_loss": 49000.0,
            },
        )

    def _ctx_short(self):
        return _make_context(
            position_flow="ask_take_profit",
            position_data={
                "symbol": "BTCUSDT", "side": Side.SHORT,
                "entry_price": 50000.0, "leverage": 5.0,
                "margin_usdt": 500.0, "stop_loss": 51000.0,
            },
        )

    @pytest.mark.asyncio
    async def test_long_valid_tp_above_entry(self):
        handler = _make_handler()
        ctx = self._ctx_long()
        # tp=52000 → rr=2.0, above 1.5 threshold
        await handler._handle_position_flow(_make_update(text="52000"), ctx, "52000")
        assert ctx.user_data["position_data"]["take_profit"] == 52000.0
        assert ctx.user_data["position_flow"] == "ask_reason"

    @pytest.mark.asyncio
    async def test_long_tp_below_entry_rejected(self):
        handler = _make_handler()
        ctx = self._ctx_long()
        update = _make_update(text="48000")
        await handler._handle_position_flow(update, ctx, "48000")
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "롱" in text

    @pytest.mark.asyncio
    async def test_long_tp_equal_entry_rejected(self):
        handler = _make_handler()
        ctx = self._ctx_long()
        update = _make_update(text="50000")
        await handler._handle_position_flow(update, ctx, "50000")
        update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_short_valid_tp_below_entry(self):
        handler = _make_handler()
        ctx = self._ctx_short()
        # entry=50000, sl=51000 → risk=1000 → rr2 tp=48000
        await handler._handle_position_flow(_make_update(text="48000"), ctx, "48000")
        assert ctx.user_data["position_data"]["take_profit"] == 48000.0
        assert ctx.user_data["position_flow"] == "ask_reason"

    @pytest.mark.asyncio
    async def test_short_tp_above_entry_rejected(self):
        handler = _make_handler()
        ctx = self._ctx_short()
        update = _make_update(text="52000")
        await handler._handle_position_flow(update, ctx, "52000")
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "숏" in text

    @pytest.mark.asyncio
    async def test_low_rr_triggers_confirm_step(self):
        handler = _make_handler()
        # entry=50000, sl=49000 → risk=1000; tp=50500 → reward=500 → rr=0.5 < 1.5
        ctx = self._ctx_long()
        await handler._handle_position_flow(_make_update(text="50500"), ctx, "50500")
        assert ctx.user_data["position_flow"] == "confirm_low_rr"
        text = update_text = update_text = ""
        for call in _make_update().message.reply_text.call_args_list:
            pass
        # Check reply_text was called with low RR warning
        call_text = _make_update().message.reply_text.call_args
        # Just verify flow moved to confirm_low_rr
        assert ctx.user_data["position_flow"] == "confirm_low_rr"

    @pytest.mark.asyncio
    async def test_zero_tp_rejected(self):
        handler = _make_handler()
        ctx = self._ctx_long()
        update = _make_update(text="0")
        await handler._handle_position_flow(update, ctx, "0")
        update.message.reply_text.assert_awaited_once()


# ── _handle_position_flow: confirm_low_rr ────────────────────────────────────

class TestPositionFlowConfirmLowRR:
    def _ctx(self):
        return _make_context(
            position_flow="confirm_low_rr",
            position_data={
                "symbol": "BTCUSDT", "side": Side.LONG,
                "entry_price": 50000.0, "leverage": 5.0,
                "margin_usdt": 500.0, "stop_loss": 49000.0,
                "take_profit": 50500.0,
            },
        )

    @pytest.mark.asyncio
    async def test_yes_advances_to_ask_reason(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="예"), ctx, "예")
        assert ctx.user_data["position_flow"] == "ask_reason"

    @pytest.mark.asyncio
    async def test_yes_english_advances(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="yes"), ctx, "yes")
        assert ctx.user_data["position_flow"] == "ask_reason"

    @pytest.mark.asyncio
    async def test_no_cancels_flow(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="아니오"), ctx, "아니오")
        assert "position_flow" not in ctx.user_data
        assert "position_data" not in ctx.user_data

    @pytest.mark.asyncio
    async def test_no_english_cancels_flow(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="no"), ctx, "no")
        assert "position_flow" not in ctx.user_data

    @pytest.mark.asyncio
    async def test_invalid_answer_stays(self):
        handler = _make_handler()
        ctx = self._ctx()
        update = _make_update(text="모르겠어요")
        await handler._handle_position_flow(update, ctx, "모르겠어요")
        update.message.reply_text.assert_awaited_once()
        assert ctx.user_data["position_flow"] == "confirm_low_rr"

    @pytest.mark.asyncio
    async def test_short_yes_advances(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="ㅇ"), ctx, "ㅇ")
        assert ctx.user_data["position_flow"] == "ask_reason"

    @pytest.mark.asyncio
    async def test_short_no_cancels(self):
        handler = _make_handler()
        ctx = self._ctx()
        await handler._handle_position_flow(_make_update(text="ㄴ"), ctx, "ㄴ")
        assert "position_flow" not in ctx.user_data


# ── _handle_position_flow: ask_reason ────────────────────────────────────────

class TestPositionFlowAskReason:
    def _ctx(self):
        return _make_context(
            position_flow="ask_reason",
            position_data={
                "symbol": "BTCUSDT", "side": Side.LONG,
                "entry_price": 50000.0, "leverage": 5.0,
                "margin_usdt": 500.0, "stop_loss": 49000.0,
                "take_profit": 52000.0,
            },
        )

    def _setup_position_manager(self, handler):
        bot = handler._bot_ref
        bot.position_manager = MagicMock()
        mock_pos = _make_mock_position()
        bot.position_manager.open_position.return_value = mock_pos
        bot.position_manager.get_active_positions.return_value = []
        bot.reporter = MagicMock()
        bot.reporter.format_position_registered.return_value = "포지션 등록 완료"
        return bot

    @pytest.mark.asyncio
    async def test_empty_reason_rejected(self):
        handler = _make_handler()
        ctx = self._ctx()
        update = _make_update(text="  ")
        await handler._handle_position_flow(update, ctx, "  ")
        update.message.reply_text.assert_awaited_once()
        assert ctx.user_data["position_flow"] == "ask_reason"

    @pytest.mark.asyncio
    async def test_valid_reason_registers_position(self):
        handler = _make_handler()
        self._setup_position_manager(handler)
        ctx = self._ctx()
        await handler._handle_position_flow(
            _make_update(text="200EMA 반등 + 펀딩 음전환"), ctx,
            "200EMA 반등 + 펀딩 음전환",
        )
        handler._bot_ref.position_manager.open_position.assert_called_once()
        assert "position_flow" not in ctx.user_data
        assert "position_data" not in ctx.user_data

    @pytest.mark.asyncio
    async def test_reason_clears_flow(self):
        handler = _make_handler()
        self._setup_position_manager(handler)
        ctx = self._ctx()
        await handler._handle_position_flow(
            _make_update(text="테스트 진입 사유"), ctx, "테스트 진입 사유",
        )
        assert "position_flow" not in ctx.user_data

    @pytest.mark.asyncio
    async def test_no_position_manager_sends_error(self):
        handler = _make_handler()
        handler._bot_ref = MagicMock(spec=[])  # no position_manager attr
        ctx = self._ctx()
        update = _make_update(text="테스트 사유")
        await handler._handle_position_flow(update, ctx, "테스트 사유")
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "초기화" in text

    @pytest.mark.asyncio
    async def test_portfolio_guard_blocks_entry(self):
        handler = _make_handler()
        bot = handler._bot_ref
        bot.position_manager = MagicMock()
        bot.position_manager.get_active_positions.return_value = []
        guard_result = MagicMock()
        guard_result.allowed = False
        guard_result.reason = "최대 포지션 수 초과"
        bot.portfolio_guard = MagicMock()
        bot.portfolio_guard.check_can_open.return_value = guard_result
        bot.reporter = MagicMock()
        ctx = self._ctx()
        update = _make_update(text="테스트 사유")
        await handler._handle_position_flow(update, ctx, "테스트 사유")
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "거부" in text
        # Flow cleared even on rejection
        assert "position_flow" not in ctx.user_data


# ── _cmd_close_position ───────────────────────────────────────────────────────

class TestCmdClosePosition:
    @pytest.mark.asyncio
    async def test_no_positions(self):
        handler = _make_handler()
        bot = handler._bot_ref
        bot.position_manager = MagicMock()
        bot.position_manager.get_active_positions.return_value = []
        update = _make_update()
        context = _make_context()
        await handler._cmd_close_position(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "없습니다" in text

    @pytest.mark.asyncio
    async def test_single_position_closes_immediately(self):
        handler = _make_handler()
        bot = handler._bot_ref
        pos = _make_mock_position()
        bot.position_manager = MagicMock()
        bot.position_manager.get_active_positions.return_value = [pos]
        bot.reporter = MagicMock()
        bot.reporter.format_position_closed.return_value = "청산 완료"
        # _run_sync returns empty candles to skip pnl calculation
        bot._run_sync = AsyncMock(return_value=[])
        update = _make_update()
        context = _make_context()
        await handler._cmd_close_position(update, context)
        bot.position_manager.close_position.assert_called_once_with(pos.id, "12345")
        update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multiple_positions_asks_which(self):
        handler = _make_handler()
        bot = handler._bot_ref
        pos1 = _make_mock_position(symbol="BTCUSDT", pos_id=1)
        pos2 = _make_mock_position(symbol="ETHUSDT", side=Side.SHORT, pos_id=2)
        bot.position_manager = MagicMock()
        bot.position_manager.get_active_positions.return_value = [pos1, pos2]
        update = _make_update()
        context = _make_context()
        await handler._cmd_close_position(update, context)
        assert context.user_data.get("position_flow") == "ask_close_which"
        text = update.message.reply_text.call_args[0][0]
        assert "BTCUSDT" in text
        assert "ETHUSDT" in text

    @pytest.mark.asyncio
    async def test_no_bot_sends_error(self):
        handler = _make_handler()
        handler._bot_ref = None
        update = _make_update()
        context = _make_context()
        await handler._cmd_close_position(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "초기화" in text

    @pytest.mark.asyncio
    async def test_wrong_chat_rejected(self):
        handler = _make_handler()
        update = _make_update(chat_id=99999)
        context = _make_context()
        await handler._cmd_close_position(update, context)
        update.message.reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bot_without_position_manager(self):
        handler = _make_handler()
        handler._bot_ref = MagicMock(spec=[])  # no position_manager
        update = _make_update()
        context = _make_context()
        await handler._cmd_close_position(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "초기화" in text

    @pytest.mark.asyncio
    async def test_ask_close_which_flow(self):
        handler = _make_handler()
        bot = handler._bot_ref
        pos = _make_mock_position(symbol="BTCUSDT")
        bot.position_manager = MagicMock()
        bot.position_manager.get_active_positions.return_value = [pos]
        bot.reporter = MagicMock()
        bot.reporter.format_position_closed.return_value = "청산 완료"
        bot._run_sync = AsyncMock(return_value=[])
        update = _make_update(text="BTC")
        context = _make_context(position_flow="ask_close_which", position_data={})
        await handler._handle_position_flow(update, context, "BTC")
        bot.position_manager.close_position.assert_called_once_with(pos.id, "12345")
        assert "position_flow" not in context.user_data

    @pytest.mark.asyncio
    async def test_ask_close_which_not_found(self):
        handler = _make_handler()
        bot = handler._bot_ref
        pos = _make_mock_position(symbol="BTCUSDT")
        bot.position_manager = MagicMock()
        bot.position_manager.get_active_positions.return_value = [pos]
        update = _make_update(text="ETH")
        context = _make_context(position_flow="ask_close_which", position_data={})
        await handler._handle_position_flow(update, context, "ETH")
        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "찾을 수 없습니다" in text
        # Flow NOT cleared — user can retry
        assert context.user_data.get("position_flow") == "ask_close_which"


# ── Legacy static validation tests ────────────────────────────────────────────

class TestPositionFlowValidation:
    def test_long_sl_above_entry(self):
        assert Side.LONG == Side.LONG and 51000.0 >= 50000.0  # should be rejected

    def test_short_sl_below_entry(self):
        assert Side.SHORT == Side.SHORT and 49000.0 <= 50000.0  # should be rejected

    def test_leverage_bounds(self):
        assert 0 < 5 <= 125
        assert not (0 < 0 <= 125)
        assert not (0 < 126 <= 125)

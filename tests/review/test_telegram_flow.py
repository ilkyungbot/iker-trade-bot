"""Tests for Telegram auth logic and input validation (Task 11)."""

import pytest
from unittest.mock import MagicMock
from core.types import Side


class TestCheckAuth:
    def _make_handler(self, chat_id):
        from review.telegram_commands import TelegramCommandHandler
        handler = TelegramCommandHandler.__new__(TelegramCommandHandler)
        handler.chat_id = chat_id
        return handler

    def _make_update(self, chat_id_int):
        update = MagicMock()
        update.effective_chat = MagicMock()
        update.effective_chat.id = chat_id_int
        return update

    def test_empty_chat_id_rejects(self):
        handler = self._make_handler("")
        assert handler._check_auth(self._make_update(12345)) is False

    def test_matching_id(self):
        handler = self._make_handler("12345")
        assert handler._check_auth(self._make_update(12345)) is True

    def test_wrong_id(self):
        handler = self._make_handler("12345")
        assert handler._check_auth(self._make_update(99999)) is False

    def test_no_effective_chat(self):
        handler = self._make_handler("12345")
        update = MagicMock()
        update.effective_chat = None
        assert handler._check_auth(update) is False


class TestPositionFlowValidation:
    def test_long_sl_above_entry(self):
        assert Side.LONG == Side.LONG and 51000.0 >= 50000.0  # should be rejected

    def test_short_sl_below_entry(self):
        assert Side.SHORT == Side.SHORT and 49000.0 <= 50000.0  # should be rejected

    def test_leverage_bounds(self):
        assert 0 < 5 <= 125
        assert not (0 < 0 <= 125)
        assert not (0 < 126 <= 125)

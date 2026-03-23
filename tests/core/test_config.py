"""Tests for config loading."""

import os
import pytest
from unittest.mock import patch

from core.config import AppConfig


_REQUIRED_ENV = {
    "TELEGRAM_BOT_TOKEN": "test-token",
    "TELEGRAM_CHAT_ID": "123456",
}


class TestConfigLoading:
    def test_default_config_loads(self, monkeypatch):
        """기본 환경변수 없이도 로드 가능."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
        config = AppConfig.from_env()
        assert config.signal.primary_interval == "240"
        assert config.signal.signal_cooldown_minutes == 30

    def test_testnet_defaults_to_false(self, monkeypatch):
        """testnet 기본값이 false."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
        monkeypatch.delenv("BYBIT_TESTNET", raising=False)
        config = AppConfig.from_env()
        assert config.bybit.testnet is False

    def test_signal_config_from_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
        monkeypatch.setenv("SIGNAL_COOLDOWN_MINUTES", "45")
        monkeypatch.setenv("MIN_SIGNAL_QUALITY", "strong")
        config = AppConfig.from_env()
        assert config.signal.signal_cooldown_minutes == 45
        assert config.signal.min_signal_quality == "strong"


def test_validate_raises_on_missing_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        AppConfig.from_env()


def test_validate_raises_on_missing_chat_id(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token123")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(ValueError, match="TELEGRAM_CHAT_ID"):
        AppConfig.from_env()

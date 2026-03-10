"""Tests for config validation logic in AppConfig.from_env()."""

import os
import pytest
from unittest.mock import patch

from core.config import AppConfig
from core.types import TradingMode


class TestConfigValidation:
    """Verify that live-mode credential checks and paper-mode defaults work."""

    def test_live_mode_raises_without_api_key(self):
        """PAPER_TRADING=false with no BYBIT_API_KEY must raise ValueError."""
        env = {
            "PAPER_TRADING": "false",
            "BYBIT_API_KEY": "",
            "BYBIT_API_SECRET": "some_secret",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="BYBIT_API_KEY"):
                AppConfig.from_env()

    def test_live_mode_raises_without_api_secret(self):
        """PAPER_TRADING=false with no BYBIT_API_SECRET must raise ValueError."""
        env = {
            "PAPER_TRADING": "false",
            "BYBIT_API_KEY": "some_key",
            "BYBIT_API_SECRET": "",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="BYBIT_API_SECRET"):
                AppConfig.from_env()

    def test_paper_mode_allows_empty_keys(self):
        """In paper mode, empty API credentials should not raise."""
        env = {
            "PAPER_TRADING": "true",
            "BYBIT_API_KEY": "",
            "BYBIT_API_SECRET": "",
        }
        with patch.dict(os.environ, env, clear=False):
            config = AppConfig.from_env()
            assert config.trading.mode == TradingMode.PAPER
            assert config.bybit.api_key == ""
            assert config.bybit.api_secret == ""

    def test_default_is_paper_mode(self):
        """With no env vars set, trading mode must default to PAPER."""
        # Clear the env vars that influence mode selection
        env = {
            "PAPER_TRADING": "",  # not "true", but os.getenv default handles it
        }
        # Remove PAPER_TRADING entirely so the default "true" kicks in
        with patch.dict(os.environ, {}, clear=False):
            # Ensure PAPER_TRADING is not set so default applies
            os.environ.pop("PAPER_TRADING", None)
            config = AppConfig.from_env()
            assert config.trading.mode == TradingMode.PAPER

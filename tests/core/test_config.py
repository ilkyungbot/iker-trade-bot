"""Tests for config loading."""

import os
import pytest
from unittest.mock import patch

from core.config import AppConfig


class TestConfigLoading:
    def test_default_config_loads(self):
        """기본 환경변수 없이도 로드 가능."""
        config = AppConfig.from_env()
        assert config.signal.primary_interval == "240"
        assert config.signal.signal_cooldown_minutes == 30

    def test_testnet_defaults_to_false(self):
        """testnet 기본값이 false."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BYBIT_TESTNET", None)
            config = AppConfig.from_env()
            assert config.bybit.testnet is False

    def test_signal_config_from_env(self):
        env = {
            "SIGNAL_COOLDOWN_MINUTES": "45",
            "MIN_SIGNAL_QUALITY": "strong",
        }
        with patch.dict(os.environ, env, clear=False):
            config = AppConfig.from_env()
            assert config.signal.signal_cooldown_minutes == 45
            assert config.signal.min_signal_quality == "strong"

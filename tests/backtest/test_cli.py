"""Tests for the backtest CLI argument parsing and main entry point."""

import sys
import pytest
from unittest.mock import patch, MagicMock

from backtest.cli import parse_args, main


class TestParseArgs:
    """Verify default and custom argument parsing."""

    def test_parse_args_defaults(self):
        """No arguments should produce BTCUSDT, 90 days, 100000 capital."""
        args = parse_args([])
        assert args.symbol == "BTCUSDT"
        assert args.days == 90
        assert args.capital == 100_000.0

    def test_parse_args_custom(self):
        """Custom --symbol, --days, --capital should be parsed correctly."""
        args = parse_args(["--symbol", "ETHUSDT", "--days", "30", "--capital", "50000"])
        assert args.symbol == "ETHUSDT"
        assert args.days == 30
        assert args.capital == 50_000.0


class TestMainExitOnNoData:
    """Verify that main() exits when the collector returns no candle data."""

    @patch("backtest.cli.BybitCollector")
    def test_main_exits_on_no_data(self, mock_collector_cls):
        """When get_candles returns an empty list, main should call sys.exit(1)."""
        # Set up mock collector that returns no data
        mock_collector = MagicMock()
        mock_collector.get_candles.return_value = []
        mock_collector_cls.from_config.return_value = mock_collector

        with pytest.raises(SystemExit) as exc_info:
            main(["--symbol", "BTCUSDT", "--days", "10"])

        assert exc_info.value.code == 1

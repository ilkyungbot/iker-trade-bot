"""Integration tests for SignalBot orchestrator."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone, timedelta


def _make_bot():
    """Helper: create SignalBot with mocked external dependencies."""
    with patch("main.TelegramBotSender"), \
         patch("main.TelegramCommandHandler") as mock_cmd, \
         patch("main.BybitCollector") as mock_collector:
        mock_cmd.return_value = MagicMock()
        mock_cmd.return_value.attach_bot = MagicMock()
        mock_collector.from_config.return_value = MagicMock()

        from core.config import AppConfig
        import os
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test_token")
        os.environ.setdefault("TELEGRAM_CHAT_ID", "12345")
        config = AppConfig.from_env()

        from main import SignalBot
        bot = SignalBot(config)
        return bot


@pytest.fixture
def bot():
    with patch("main.TelegramBotSender"), \
         patch("main.TelegramCommandHandler") as mock_cmd, \
         patch("main.BybitCollector") as mock_collector:
        mock_cmd.return_value = MagicMock()
        mock_cmd.return_value.attach_bot = MagicMock()
        mock_collector.from_config.return_value = MagicMock()

        import os
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test_token")
        os.environ.setdefault("TELEGRAM_CHAT_ID", "12345")

        from core.config import AppConfig
        config = AppConfig.from_env()

        from main import SignalBot
        yield SignalBot(config)


class TestSignalBotInit:
    """Test that SignalBot creates all required service components."""

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    def test_creates_services(self):
        """SignalBot should create all service instances."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)

            assert bot.coin_analyzer is not None
            assert bot.briefing_service is not None
            assert bot.signal_generator is not None
            assert bot.position_manager is not None
            assert bot.portfolio_guard is not None

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    def test_creates_strategy_components(self):
        """SignalBot should create all strategy components."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)

            assert bot.pair_selector is not None
            assert bot.trend_strategy is not None
            assert bot.funding_strategy is not None
            assert bot.cooldown is not None
            assert bot.signal_tracker is not None
            assert bot.reporter is not None

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    def test_initial_state(self):
        """Bot starts with _running=False."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)

            assert bot._running is False

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    def test_stop_sets_running_false(self):
        """stop() sets _running to False."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            bot._running = True
            bot.stop()
            assert bot._running is False


class TestMonitorManualPositions:
    """Test the monitor loop handles edge cases."""

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_no_active_positions(self):
        """When no positions, monitor returns immediately."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector_instance = MagicMock()
            mock_collector.from_config.return_value = mock_collector_instance

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            bot.position_manager.get_all_active_positions = MagicMock(return_value=[])

            await bot.monitor_manual_positions()
            # Should return early without calling collector.get_candles
            mock_collector_instance.get_candles.assert_not_called()

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_with_positions_calls_get_candles(self):
        """When positions exist, calls get_candles for each."""
        from core.types import Side, ManualPosition

        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector_instance = MagicMock()
            mock_collector_instance.get_candles = MagicMock(return_value=None)
            mock_collector.from_config.return_value = mock_collector_instance

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)

            pos = ManualPosition(
                id=1,
                chat_id="123",
                symbol="BTCUSDT",
                side=Side.LONG,
                entry_price=50000.0,
                leverage=5.0,
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
            bot.position_manager.get_all_active_positions = MagicMock(return_value=[pos])

            # get_candles returns None → position loop skips
            await bot.monitor_manual_positions()
            # Should have been called (at least for BTCUSDT market data)
            assert mock_collector_instance.get_candles.call_count >= 1


class TestDailyWeeklyReports:
    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_daily_report_no_signals(self):
        """When no signals today, send_weekly_accuracy is NOT called."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            bot.signal_tracker.weekly_report = MagicMock(return_value={"total": 0})
            bot.reporter.send_weekly_accuracy = AsyncMock()

            await bot.daily_report()
            bot.reporter.send_weekly_accuracy.assert_not_called()

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_daily_report_with_signals(self):
        """When signals exist today, send_weekly_accuracy IS called."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            report = {"total": 3, "tp_rate": 0.67, "sl_rate": 0.1, "by_quality": {}}
            bot.signal_tracker.weekly_report = MagicMock(return_value=report)
            bot.reporter.send_weekly_accuracy = AsyncMock()

            await bot.daily_report()
            bot.reporter.send_weekly_accuracy.assert_called_once_with(report)

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_weekly_report(self):
        """weekly_report always calls send_weekly_accuracy."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            report = {"total": 5, "tp_rate": 0.6, "sl_rate": 0.2, "by_quality": {}}
            bot.signal_tracker.weekly_report = MagicMock(return_value=report)
            bot.reporter.send_weekly_accuracy = AsyncMock()

            await bot.weekly_report()
            bot.reporter.send_weekly_accuracy.assert_called_once()

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_weekly_report_empty(self):
        """weekly_report with 0 signals still calls send_weekly_accuracy."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            bot.signal_tracker.weekly_report = MagicMock(return_value={"total": 0})
            bot.reporter.send_weekly_accuracy = AsyncMock()

            await bot.weekly_report()
            bot.reporter.send_weekly_accuracy.assert_called_once()


class TestHourlyBriefing:
    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_hourly_briefing_calls_reporter(self):
        """hourly_briefing fetches briefing and calls reporter."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)

            briefing_data = {
                "time": "01/01 00:00 UTC",
                "market_summary": {"top_coins": []},
                "scored_coins": [],
                "funding_alerts": [],
                "watched_pairs": [],
            }
            bot.briefing_service.generate_briefing = AsyncMock(return_value=briefing_data)
            bot.pair_selector._current_pairs = []
            bot.reporter.send_hourly_briefing = AsyncMock()

            await bot.hourly_briefing()
            bot.reporter.send_hourly_briefing.assert_called_once()

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_hourly_briefing_error_handled(self):
        """hourly_briefing errors are caught and don't propagate."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            bot.briefing_service.generate_briefing = AsyncMock(
                side_effect=Exception("network error")
            )

            # Should not raise
            await bot.hourly_briefing()


class TestPortfolioGuardReset:
    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_reset_daily_guard(self):
        """_reset_daily_guard calls portfolio_guard.reset_daily."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            bot.portfolio_guard.reset_daily = MagicMock()

            await bot._reset_daily_guard()
            bot.portfolio_guard.reset_daily.assert_called_once()

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_reset_monthly_guard(self):
        """_reset_monthly_guard calls portfolio_guard.reset_monthly."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            bot.portfolio_guard.reset_monthly = MagicMock()

            await bot._reset_monthly_guard()
            bot.portfolio_guard.reset_monthly.assert_called_once()


class TestGenerateBriefingDelegation:
    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_generate_briefing_adds_watched_pairs(self):
        """generate_briefing injects watched_pairs from pair_selector."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)

            base_briefing = {
                "time": "01/01 00:00 UTC",
                "market_summary": {"top_coins": []},
                "scored_coins": [],
                "funding_alerts": [],
                "watched_pairs": [],
            }
            bot.briefing_service.generate_briefing = AsyncMock(return_value=base_briefing)

            pair_mock = MagicMock()
            pair_mock.symbol = "BTCUSDT"
            bot.pair_selector._current_pairs = [pair_mock]

            result = await bot.generate_briefing()
            assert result["watched_pairs"] == ["BTCUSDT"]

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_analyze_coin_delegates(self):
        """analyze_coin delegates to coin_analyzer."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)

            expected = {"symbol": "BTCUSDT", "verdict": "관망"}
            bot.coin_analyzer.analyze_coin = AsyncMock(return_value=expected)

            result = await bot.analyze_coin("BTC")
            assert result == expected
            bot.coin_analyzer.analyze_coin.assert_called_once_with("BTC")


class TestTryAutoResume:
    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_auto_resume_triggers_alert(self):
        """When circuit breaker auto-resumes, sends alert."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            bot.signal_generator.cooldown.try_auto_resume = MagicMock(return_value=True)
            bot.reporter.send_alert = AsyncMock()

            await bot._try_auto_resume()
            bot.reporter.send_alert.assert_called_once()

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @pytest.mark.anyio
    async def test_auto_resume_no_alert_when_not_resumed(self):
        """When circuit breaker does NOT auto-resume, no alert sent."""
        with patch("main.TelegramBotSender"), \
             patch("main.TelegramCommandHandler") as mock_cmd, \
             patch("main.BybitCollector") as mock_collector:
            mock_cmd.return_value = MagicMock()
            mock_cmd.return_value.attach_bot = MagicMock()
            mock_collector.from_config.return_value = MagicMock()

            from core.config import AppConfig
            config = AppConfig.from_env()
            from main import SignalBot
            bot = SignalBot(config)
            bot.signal_generator.cooldown.try_auto_resume = MagicMock(return_value=False)
            bot.reporter.send_alert = AsyncMock()

            await bot._try_auto_resume()
            bot.reporter.send_alert.assert_not_called()

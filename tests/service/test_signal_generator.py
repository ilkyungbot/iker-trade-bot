# tests/service/test_signal_generator.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone, timedelta
from service.signal_generator import SignalGenerator
from core.types import SignalAction, SignalQuality, SignalMessage, Signal, StrategyName


def _make_signal_msg(quality=SignalQuality.STRONG, confidence=0.8):
    signal = Signal(
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        action=SignalAction.ENTER_LONG,
        strategy=StrategyName.TREND_FOLLOWING,
        entry_price=50000,
        stop_loss=48000,
        take_profit=55000,
        confidence=confidence,
    )
    return SignalMessage(
        signal=signal,
        quality=quality,
        explanation=["EMA 골든크로스"],
        indicators={"rsi": 55},
        risk_reward_ratio=1.5,
    )


class TestSignalGenerator:
    def setup_method(self):
        self.collector = MagicMock()
        self.validator = MagicMock()
        self.config = MagicMock()
        self.config.signal.signal_cooldown_minutes = 30
        self.config.signal.primary_interval = "240"
        self.config.signal.max_pairs = 5
        self.config.signal.pair_rebalance_days = 14
        self.config.signal.min_signal_quality = "moderate"

        self.pair_selector = MagicMock()
        self.pair_selector._current_pairs = []

        self.trend_strategy = MagicMock()
        self.funding_strategy = MagicMock()
        self.cooldown = MagicMock()
        self.reporter = MagicMock()
        self.reporter.send_signal = AsyncMock()
        self.reporter.send_alert = AsyncMock()
        self.signal_tracker = MagicMock()

        self.gen = SignalGenerator(
            collector=self.collector,
            validator=self.validator,
            config=self.config,
            pair_selector=self.pair_selector,
            trend_strategy=self.trend_strategy,
            funding_strategy=self.funding_strategy,
            cooldown=self.cooldown,
            reporter=self.reporter,
            signal_tracker=self.signal_tracker,
        )

    @pytest.mark.anyio
    async def test_run_cycle_skips_on_cooldown(self):
        """When cooldown active, cycle does nothing."""
        self.cooldown.can_send_signal.return_value = False
        await self.gen.run_cycle()
        self.collector.get_all_usdt_perpetuals.assert_not_called()

    @pytest.mark.anyio
    async def test_run_cycle_no_pairs(self):
        """When no pairs, no signals generated."""
        self.cooldown.can_send_signal.return_value = True
        self.pair_selector._current_pairs = []
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[])
        self.collector.get_candles = MagicMock(return_value=[])
        await self.gen.run_cycle()
        self.reporter.send_signal.assert_not_called()

    @pytest.mark.anyio
    async def test_generate_for_pair_no_candles(self):
        """When no candles available, return None."""
        self.collector.get_candles = MagicMock(return_value=None)
        result = await self.gen.generate_for_pair("BTCUSDT", datetime.now(timezone.utc))
        assert result is None

    @pytest.mark.anyio
    async def test_generate_for_pair_empty_candles(self):
        """When empty candles list returned, return None."""
        self.collector.get_candles = MagicMock(return_value=[])
        result = await self.gen.generate_for_pair("BTCUSDT", datetime.now(timezone.utc))
        assert result is None

    @pytest.mark.anyio
    async def test_generate_for_pair_invalid_data(self):
        """When validation fails, return None."""
        self.collector.get_candles = MagicMock(return_value=[MagicMock()])
        validation_result = MagicMock()
        validation_result.is_valid = False
        self.validator.validate_candles.return_value = validation_result
        result = await self.gen.generate_for_pair("BTCUSDT", datetime.now(timezone.utc))
        assert result is None

    @pytest.mark.anyio
    async def test_generate_for_pair_returns_best_signal(self):
        """When both strategies return signals, best is chosen."""
        from core.types import Candle

        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles = [
            Candle(
                timestamp=base_time + timedelta(hours=4 * i),
                open=50000.0 + i,
                high=50100.0 + i,
                low=49900.0 + i,
                close=50050.0 + i,
                volume=1000.0,
                symbol="BTCUSDT",
                interval="240",
            )
            for i in range(70)
        ]
        self.collector.get_candles = MagicMock(return_value=candles)
        self.collector.get_funding_rates = MagicMock(return_value=[])

        validation_result = MagicMock()
        validation_result.is_valid = True
        self.validator.validate_candles.return_value = validation_result

        strong_msg = _make_signal_msg(quality=SignalQuality.STRONG, confidence=0.9)
        weak_msg = _make_signal_msg(quality=SignalQuality.WEAK, confidence=0.3)

        self.trend_strategy.generate_signal.return_value = strong_msg
        self.funding_strategy.generate_signal.return_value = weak_msg

        result = await self.gen.generate_for_pair("BTCUSDT", datetime.now(timezone.utc))
        assert result is not None
        assert result.quality == SignalQuality.STRONG

    @pytest.mark.anyio
    async def test_generate_for_pair_no_strategy_signals(self):
        """When both strategies return None, return None."""
        from core.types import Candle

        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles = [
            Candle(
                timestamp=base_time + timedelta(hours=4 * i),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=1000.0,
                symbol="BTCUSDT",
                interval="240",
            )
            for i in range(70)
        ]
        self.collector.get_candles = MagicMock(return_value=candles)
        self.collector.get_funding_rates = MagicMock(return_value=[])

        validation_result = MagicMock()
        validation_result.is_valid = True
        self.validator.validate_candles.return_value = validation_result

        self.trend_strategy.generate_signal.return_value = None
        self.funding_strategy.generate_signal.return_value = None

        result = await self.gen.generate_for_pair("BTCUSDT", datetime.now(timezone.utc))
        assert result is None

    @pytest.mark.anyio
    async def test_run_cycle_sends_best_signal(self):
        """run_cycle sends the highest-scored signal."""
        from core.types import PairInfo

        self.cooldown.can_send_signal.return_value = True

        pair = MagicMock()
        pair.symbol = "BTCUSDT"
        self.pair_selector._current_pairs = [pair]

        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[])
        self.collector.get_candles = MagicMock(return_value=[])

        strong_msg = _make_signal_msg(quality=SignalQuality.STRONG, confidence=0.9)

        # Patch generate_for_pair to return our signal
        async def fake_generate(symbol, now):
            return strong_msg

        self.gen.generate_for_pair = fake_generate
        self.signal_tracker.record_signal = MagicMock()

        await self.gen.run_cycle()
        self.reporter.send_signal.assert_called_once_with(strong_msg)

    @pytest.mark.anyio
    async def test_run_cycle_filters_by_min_quality_strong(self):
        """When min_signal_quality=strong, moderate signals are filtered out."""
        self.config.signal.min_signal_quality = "strong"
        self.cooldown.can_send_signal.return_value = True

        pair = MagicMock()
        pair.symbol = "BTCUSDT"
        self.pair_selector._current_pairs = [pair]

        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[])
        self.collector.get_candles = MagicMock(return_value=[])

        moderate_msg = _make_signal_msg(quality=SignalQuality.MODERATE)

        async def fake_generate(symbol, now):
            return moderate_msg

        self.gen.generate_for_pair = fake_generate

        await self.gen.run_cycle()
        self.reporter.send_signal.assert_not_called()

    @pytest.mark.anyio
    async def test_run_cycle_api_error_sends_alert(self):
        """API errors exceeding threshold trigger alert."""
        self.cooldown.can_send_signal.return_value = True
        self.cooldown.record_api_error.return_value = True

        # Patch _update_pairs to succeed, but generate_for_pair to raise
        async def fake_update_pairs():
            pass

        pair = MagicMock()
        pair.symbol = "BTCUSDT"
        self.pair_selector._current_pairs = [pair]

        async def fake_generate_raises(symbol, now):
            raise Exception("API down")

        self.gen._update_pairs = fake_update_pairs
        self.gen.generate_for_pair = fake_generate_raises

        await self.gen.run_cycle()
        self.reporter.send_alert.assert_called_once()

    @pytest.mark.anyio
    async def test_run_cycle_api_error_below_threshold(self):
        """API errors below threshold do NOT trigger alert."""
        self.cooldown.can_send_signal.return_value = True
        self.cooldown.record_api_error.return_value = False

        pair = MagicMock()
        pair.symbol = "BTCUSDT"
        self.pair_selector._current_pairs = [pair]

        async def fake_update_pairs():
            pass

        async def fake_generate_raises(symbol, now):
            raise Exception("API down")

        self.gen._update_pairs = fake_update_pairs
        self.gen.generate_for_pair = fake_generate_raises

        await self.gen.run_cycle()
        self.reporter.send_alert.assert_not_called()

    def test_signal_score_strong(self):
        """STRONG quality = 3 + confidence."""
        msg = _make_signal_msg(quality=SignalQuality.STRONG, confidence=0.8)
        score = SignalGenerator._signal_score(msg)
        assert score == 3 + 0.8

    def test_signal_score_moderate(self):
        """MODERATE quality = 2 + confidence."""
        msg = _make_signal_msg(quality=SignalQuality.MODERATE, confidence=0.5)
        score = SignalGenerator._signal_score(msg)
        assert score == 2 + 0.5

    def test_signal_score_weak(self):
        """WEAK quality = 1 + confidence."""
        msg = _make_signal_msg(quality=SignalQuality.WEAK, confidence=0.3)
        score = SignalGenerator._signal_score(msg)
        assert score == 1 + 0.3

    def test_signal_score_ordering(self):
        """Strong > moderate > weak regardless of confidence within reason."""
        strong = _make_signal_msg(quality=SignalQuality.STRONG, confidence=0.0)
        moderate = _make_signal_msg(quality=SignalQuality.MODERATE, confidence=0.9)
        weak = _make_signal_msg(quality=SignalQuality.WEAK, confidence=0.9)
        assert SignalGenerator._signal_score(strong) > SignalGenerator._signal_score(weak)
        assert SignalGenerator._signal_score(strong) > SignalGenerator._signal_score(moderate)

    @pytest.mark.anyio
    async def test_generate_for_pair_funding_error_silenced(self):
        """Funding rate fetch error is silenced and signal still generated."""
        from core.types import Candle

        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles = [
            Candle(
                timestamp=base_time + timedelta(hours=4 * i),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=1000.0,
                symbol="BTCUSDT",
                interval="240",
            )
            for i in range(70)
        ]
        self.collector.get_candles = MagicMock(return_value=candles)
        self.collector.get_funding_rates = MagicMock(side_effect=Exception("rate error"))

        validation_result = MagicMock()
        validation_result.is_valid = True
        self.validator.validate_candles.return_value = validation_result

        strong_msg = _make_signal_msg(quality=SignalQuality.STRONG)
        self.trend_strategy.generate_signal.return_value = strong_msg
        self.funding_strategy.generate_signal.return_value = None

        result = await self.gen.generate_for_pair("BTCUSDT", datetime.now(timezone.utc))
        # Should still return trend signal even if funding failed
        assert result is not None
        assert result.quality == SignalQuality.STRONG

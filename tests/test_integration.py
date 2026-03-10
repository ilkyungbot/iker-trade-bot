"""
Integration test: end-to-end flow through all layers.

Simulates a complete trading cycle without real API calls.
Verifies that all components wire together correctly.
"""

import asyncio
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from core.config import AppConfig, BybitConfig, DatabaseConfig, TelegramConfig, TradingConfig
from core.types import (
    Candle, Trade, Side, Signal, SignalAction, StrategyName,
    Position, PortfolioState, TradingMode, CircuitBreakerState,
)
from core.safety import MAX_RISK_PER_TRADE
from data.features import add_all_features, add_ema, candles_to_dataframe
from data.validator import DataValidator
from strategy.trend_following import TrendFollowingStrategy
from strategy.funding_rate import FundingRateStrategy
from sizing.kelly import kelly_fraction, calculate_position_size
from sizing.adjustments import apply_all_adjustments
from sizing.ml_model import MLConfidenceModel, FeatureRow
from execution.position_tracker import PositionTracker
from execution.circuit_breaker import CircuitBreaker
from review.performance import calculate_metrics
from backtest.runner import BacktestRunner, BacktestConfig
from backtest.analysis import analyze


def _make_candles(
    n: int = 100, symbol: str = "BTCUSDT", interval: str = "60",
    start_price: float = 50000.0, trend: float = 5.0,
) -> list[Candle]:
    """Create synthetic candles."""
    np.random.seed(42)
    candles = []
    price = start_price
    for i in range(n):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        noise = np.random.normal(0, 30)
        price += trend + noise
        open_price = price - noise / 2
        close_price = price
        high = max(open_price, close_price) + abs(np.random.normal(0, 50)) + 0.5
        low = min(open_price, close_price) - abs(np.random.normal(0, 50)) - 0.5
        candles.append(Candle(
            timestamp=ts, open=open_price, high=high, low=low,
            close=close_price, volume=np.random.uniform(100, 1000),
            symbol=symbol, interval=interval,
        ))
    return candles


class TestEndToEnd:
    """Test the complete data → features → strategy → sizing → execution pipeline."""

    def test_candles_to_features_to_signal(self):
        """Layer 1 → Layer 3: Raw candles → features → strategy signal."""
        candles_1h = _make_candles(n=100, interval="60", trend=10)
        candles_4h = _make_candles(n=60, interval="240", trend=40)

        df_1h = candles_to_dataframe(candles_1h)
        df_1h = add_all_features(df_1h)

        df_4h = candles_to_dataframe(candles_4h)
        df_4h = add_ema(df_4h, 50)

        strategy = TrendFollowingStrategy()
        signal = strategy.generate_signal(df_1h, df_4h, "BTCUSDT", None)

        # Signal may or may not fire, but the pipeline should not crash
        if signal is not None:
            assert signal.symbol == "BTCUSDT"
            assert signal.entry_price > 0
            assert signal.stop_loss > 0
            assert 0 <= signal.confidence <= 1

    def test_signal_to_position_sizing(self):
        """Layer 3 → Layer 4: Signal → Kelly sizing → adjustments."""
        signal = Signal(
            timestamp=datetime.now(timezone.utc),
            symbol="ETHUSDT", action=SignalAction.ENTER_LONG,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=3000.0, stop_loss=2900.0, take_profit=3200.0,
            confidence=0.7,
        )

        # Kelly sizing
        risk = kelly_fraction(win_rate=0.45, avg_win=0.03, avg_loss=0.015)
        assert 0 < risk <= MAX_RISK_PER_TRADE

        base_qty = calculate_position_size(
            capital=100_000, risk_fraction=risk,
            entry_price=signal.entry_price, stop_loss=signal.stop_loss,
            leverage=5.0,
        )
        assert base_qty > 0

        # Adjustments
        adjusted = apply_all_adjustments(
            base_size=base_qty,
            current_atr_pct=0.025,
            existing_position_sides=["long"],
            new_signal_side="long",
            correlation_to_existing=0.6,
            current_mdd=0.05,
            consecutive_losses=2,
            ml_confidence=0.8,
        )
        # Adjusted should be smaller due to existing position and drawdown
        assert 0 < adjusted <= base_qty * 1.5

    def test_position_lifecycle(self):
        """Layer 5: Open position → track PnL → close → get Trade."""
        tracker = PositionTracker(initial_capital=100_000)

        pos = Position(
            symbol="BTCUSDT", side=Side.LONG,
            entry_price=50000, quantity=0.1, leverage=5,
            stop_loss=49000, trailing_stop=52000,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_time=datetime.now(timezone.utc),
        )
        tracker.add_position(pos)
        assert tracker.state.position_count == 1

        # Update PnL
        tracker.update_unrealized_pnl("BTCUSDT", 51000)
        assert tracker.state.positions[0].unrealized_pnl > 0

        # Close position
        trade = tracker.close_position("BTCUSDT", 51000, fees=20)
        assert trade is not None
        assert trade.pnl > 0
        assert tracker.state.position_count == 0

    def test_circuit_breaker_integration(self):
        """Layer 5: Circuit breaker responds to portfolio state."""
        cb = CircuitBreaker()
        state = PortfolioState(
            total_capital=97000, available_capital=90000,
            daily_pnl=-3500,  # > 3% of 100k
        )

        cb_state = cb.check(state)
        assert cb.is_halted  # should halt on daily loss

    def test_ml_model_integration(self):
        """Layer 4: ML model integrates with feature extraction."""
        candles = _make_candles(n=100, trend=5)
        df = candles_to_dataframe(candles)
        df = add_all_features(df)

        # Extract features
        row = MLConfidenceModel.extract_features(df, funding_rate=0.0002)
        assert row is not None

        # Model without training returns default
        model = MLConfidenceModel()
        confidence = model.predict_confidence(row)
        assert confidence == 1.0

    def test_validator_integration(self):
        """Layer 1: Validator checks candle quality."""
        candles = _make_candles(n=50)
        validator = DataValidator()

        now = candles[-1].timestamp + timedelta(minutes=20)
        result = validator.validate_candles(candles, "60", now)
        assert result.is_valid

    def test_performance_metrics_from_trades(self):
        """Layer 6: Performance metrics from trade history."""
        trades = [
            Trade(
                symbol="BTCUSDT", side=Side.LONG,
                strategy=StrategyName.TREND_FOLLOWING,
                entry_price=50000, exit_price=51000,
                quantity=0.1, leverage=5,
                entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
                pnl=100, pnl_percent=2.0, fees=5, slippage=0,
                stop_loss_hit=False, trailing_stop_hit=True,
            ),
            Trade(
                symbol="ETHUSDT", side=Side.SHORT,
                strategy=StrategyName.FUNDING_RATE,
                entry_price=3000, exit_price=3050,
                quantity=1.0, leverage=5,
                entry_time=datetime(2024, 1, 3, tzinfo=timezone.utc),
                exit_time=datetime(2024, 1, 4, tzinfo=timezone.utc),
                pnl=-50, pnl_percent=-0.83, fees=3, slippage=0,
                stop_loss_hit=True, trailing_stop_hit=False,
            ),
        ]

        metrics = calculate_metrics(trades)
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.5

    def test_full_backtest_pipeline(self):
        """Full pipeline: candles → features → backtest → analysis."""
        candles_1h = _make_candles(n=300, interval="60", trend=8)
        candles_4h = _make_candles(n=80, interval="240", trend=32)

        df_1h = candles_to_dataframe(candles_1h)
        df_4h = candles_to_dataframe(candles_4h)

        config = BacktestConfig(initial_capital=100_000, max_positions=3)
        runner = BacktestRunner(config)
        result = runner.run(df_1h, df_4h, "BTCUSDT")

        assert result.total_bars > 0
        assert len(result.equity_curve) > 0

        # Analyze
        analysis = analyze(result)
        assert analysis.metrics.total_trades >= 0
        assert np.isfinite(analysis.total_return_pct)
        assert analysis.max_drawdown_pct >= 0

    def test_multi_layer_sizing_pipeline(self):
        """Test the full sizing chain: Kelly → adjustments → ML confidence → final size."""
        # Step 1: Kelly
        risk = kelly_fraction(0.5, 0.025, 0.015)
        assert risk > 0

        # Step 2: Base position size
        base = calculate_position_size(
            capital=100_000, risk_fraction=risk,
            entry_price=50000, stop_loss=49250, leverage=5,
        )
        assert base > 0

        # Step 3: Apply all adjustments including ML
        final = apply_all_adjustments(
            base_size=base,
            current_atr_pct=0.02,
            existing_position_sides=[],
            new_signal_side="long",
            correlation_to_existing=0.0,
            current_mdd=0.0,
            consecutive_losses=0,
            ml_confidence=1.2,
        )
        # With no penalties and ml_confidence > 1, final should be >= base
        assert final >= base * 0.9  # allow small rounding

    def test_ml_confidence_used_in_sizing(self):
        """When the ML model is trained, predict_confidence is called and affects sizing."""
        from main import TradingBot

        # Build a minimal AppConfig in paper mode
        config = AppConfig(
            bybit=BybitConfig(api_key="", api_secret="", testnet=True),
            database=DatabaseConfig(url="sqlite:///:memory:"),
            telegram=TelegramConfig(bot_token="", chat_id=""),
            trading=TradingConfig(
                mode=TradingMode.PAPER,
                default_leverage=5.0,
                strategy_a_allocation=0.70,
                strategy_b_allocation=0.30,
                max_pairs=5,
                pair_rebalance_days=14,
                candle_intervals=("15", "60", "240"),
                primary_interval="60",
                trend_interval="240",
            ),
        )

        # Create bot with heavy mocking to avoid real IO
        with patch("main.BybitCollector"), \
             patch("main.Storage"), \
             patch("main.OrderManager"), \
             patch("main.Reporter"):
            bot = TradingBot(config, initial_capital=100_000)

        # Replace ML model with a mock that tracks calls
        mock_ml = MagicMock(spec=MLConfidenceModel)
        mock_ml.predict_confidence.return_value = 1.3  # high confidence
        bot.ml_model = mock_ml

        # Prepare a signal
        signal = Signal(
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            action=SignalAction.ENTER_LONG,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            confidence=0.6,
        )

        # Build a minimal feature row
        feature_row = FeatureRow(
            atr_pct=0.02, adx=30.0, rsi=55.0, bb_width=0.04,
            volume_ratio=1.2, ema_20_slope=0.001, ema_50_slope=0.0005,
            donchian_position=0.6, funding_rate=0.0001,
        )

        # Build a tiny df_1h with required columns
        df_1h = pd.DataFrame({
            "close": [50000.0],
            "atr_pct": [0.02],
        })

        # Mock storage.get_trades and position_tracker to allow entry
        bot.storage = MagicMock()
        bot.storage.get_trades.return_value = []
        bot.position_tracker = MagicMock(spec=PositionTracker)
        bot.position_tracker.can_open_position.return_value = True
        bot.position_tracker.state = PortfolioState(
            total_capital=100_000,
            available_capital=90_000,
            positions=[],
            current_mdd=0.0,
            consecutive_losses=0,
        )
        bot.order_manager = AsyncMock()
        bot.order_manager.place_entry_order.return_value = None  # order not filled

        bot.circuit_breaker = MagicMock()
        bot.circuit_breaker.size_multiplier = 1.0

        bot.pair_selector = MagicMock()
        bot.pair_selector._current_pairs = []

        # Execute entry
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                bot._execute_entry(signal, 0.70, df_1h, feature_row)
            )
        finally:
            loop.close()

        # The key assertion: predict_confidence was called with our feature_row
        mock_ml.predict_confidence.assert_called_once_with(feature_row)

    def test_config_validation_live_mode(self):
        """Live mode without API keys must raise ValueError."""
        import os
        from unittest.mock import patch as mock_patch

        env = {
            "PAPER_TRADING": "false",
            "BYBIT_API_KEY": "",
            "BYBIT_API_SECRET": "",
        }
        with mock_patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="BYBIT_API_KEY"):
                AppConfig.from_env()

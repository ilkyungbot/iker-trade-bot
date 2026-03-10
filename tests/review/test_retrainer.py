"""Tests for retrainer."""

from datetime import datetime, timezone, timedelta
from review.retrainer import Retrainer, AllocationAdjustment
from core.types import Trade, Side, StrategyName


def _make_trade(
    pnl: float, strategy: StrategyName = StrategyName.TREND_FOLLOWING
) -> Trade:
    return Trade(
        symbol="BTCUSDT", side=Side.LONG, strategy=strategy,
        entry_price=50000, exit_price=50000 + pnl * 10,
        quantity=0.1, leverage=5,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        pnl=pnl, pnl_percent=pnl / 100,
        fees=5, slippage=0,
        stop_loss_hit=pnl < 0, trailing_stop_hit=False,
    )


class TestRetrainSchedule:
    def test_retrain_due_when_never_run(self):
        r = Retrainer()
        assert r.should_retrain(None) is True

    def test_retrain_due_after_30_days(self):
        r = Retrainer()
        last = datetime(2024, 1, 1)
        now = datetime(2024, 2, 1)
        assert r.should_retrain(last, now) is True

    def test_retrain_not_due_within_30_days(self):
        r = Retrainer()
        last = datetime(2024, 1, 1)
        now = datetime(2024, 1, 20)
        assert r.should_retrain(last, now) is False


class TestMLModelEvaluation:
    def test_good_model(self):
        r = Retrainer()
        predictions = [0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7, 0.3]
        actuals = [True, False, True, False, True, False, True, False, True, False]
        accuracy, should_disable = r.evaluate_ml_model(predictions, actuals)
        assert accuracy == 1.0
        assert should_disable is False

    def test_bad_model_disabled(self):
        r = Retrainer()
        # All predictions wrong
        predictions = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        actuals = [False, False, False, False, False, False, False, False, False, False]
        accuracy, should_disable = r.evaluate_ml_model(predictions, actuals)
        assert accuracy == 0.0
        assert should_disable is True

    def test_too_few_predictions(self):
        r = Retrainer()
        accuracy, should_disable = r.evaluate_ml_model([0.5] * 5, [True] * 5)
        assert should_disable is False


class TestAllocationAdjustment:
    def test_a_outperforms(self):
        r = Retrainer()
        # A has varied positive trades, B has varied negative trades
        import random
        random.seed(42)
        trades = (
            [_make_trade(pnl=random.uniform(50, 200), strategy=StrategyName.TREND_FOLLOWING) for _ in range(15)]
            + [_make_trade(pnl=random.uniform(-150, -10), strategy=StrategyName.FUNDING_RATE) for _ in range(15)]
        )
        adj = r.calculate_allocation_adjustment(trades)
        assert adj.strategy_a_weight > 0.7  # should increase from 70%

    def test_insufficient_data(self):
        r = Retrainer()
        trades = [_make_trade(pnl=100)]
        adj = r.calculate_allocation_adjustment(trades)
        assert adj.strategy_a_weight == 0.7  # unchanged

    def test_allocation_bounds(self):
        r = Retrainer()
        trades = (
            [_make_trade(pnl=500, strategy=StrategyName.TREND_FOLLOWING) for _ in range(20)]
            + [_make_trade(pnl=-200, strategy=StrategyName.FUNDING_RATE) for _ in range(20)]
        )
        adj = r.calculate_allocation_adjustment(trades)
        assert adj.strategy_a_weight <= 0.90
        assert adj.strategy_b_weight >= 0.10
        assert abs(adj.strategy_a_weight + adj.strategy_b_weight - 1.0) < 0.01

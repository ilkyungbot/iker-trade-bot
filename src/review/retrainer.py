"""
Layer 6: Model retraining and parameter adjustment.

Handles monthly ML model retrain and quarterly strategy parameter review.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from core.types import Trade, StrategyName
from review.performance import calculate_metrics, calculate_strategy_attribution

logger = logging.getLogger(__name__)


@dataclass
class RetrainResult:
    retrained: bool
    previous_accuracy: float
    new_accuracy: float
    model_disabled: bool
    reason: str


@dataclass
class AllocationAdjustment:
    strategy_a_weight: float  # trend following
    strategy_b_weight: float  # funding rate
    reason: str


class Retrainer:
    """Manages periodic model retraining and parameter adjustments."""

    # Minimum trades needed before retraining
    MIN_TRADES_FOR_RETRAIN = 30
    # If model accuracy is worse than this, disable it
    RANDOM_ACCURACY = 0.5

    def should_retrain(self, last_retrain: datetime | None, now: datetime | None = None) -> bool:
        """Check if monthly retrain is due."""
        if last_retrain is None:
            return True
        now = now or datetime.now()
        return (now - last_retrain).days >= 30

    def should_review_params(self, last_review: datetime | None, now: datetime | None = None) -> bool:
        """Check if quarterly parameter review is due."""
        if last_review is None:
            return True
        now = now or datetime.now()
        return (now - last_review).days >= 90

    def evaluate_ml_model(
        self,
        predictions: list[float],
        actuals: list[bool],
    ) -> tuple[float, bool]:
        """
        Evaluate ML model performance.

        Args:
            predictions: model's predicted probabilities
            actuals: whether each trade was actually profitable

        Returns:
            (accuracy, should_disable)
        """
        if len(predictions) < 10:
            return 0.0, False

        correct = sum(
            1 for pred, actual in zip(predictions, actuals)
            if (pred >= 0.5) == actual
        )
        accuracy = correct / len(predictions)

        # Disable if worse than random over last 2 weeks of trades
        should_disable = accuracy < self.RANDOM_ACCURACY
        if should_disable:
            logger.warning(
                f"ML model accuracy {accuracy:.1%} < {self.RANDOM_ACCURACY:.0%}. "
                "Disabling model."
            )

        return accuracy, should_disable

    def calculate_allocation_adjustment(
        self,
        trades: list[Trade],
        current_a_weight: float = 0.7,
        current_b_weight: float = 0.3,
    ) -> AllocationAdjustment:
        """
        Adjust strategy allocation weights based on recent performance.

        Constraints:
        - Strategy A (trend following) minimum 50%, maximum 90%
        - Strategy B (funding rate) minimum 10%, maximum 50%
        - Changes are gradual: max ±10% per quarter
        """
        attrs = calculate_strategy_attribution(trades)

        a_metrics = attrs.get(StrategyName.TREND_FOLLOWING.value)
        b_metrics = attrs.get(StrategyName.FUNDING_RATE.value)

        if not a_metrics or not b_metrics:
            return AllocationAdjustment(
                strategy_a_weight=current_a_weight,
                strategy_b_weight=current_b_weight,
                reason="Insufficient data for both strategies",
            )

        if a_metrics.total_trades < 10 or b_metrics.total_trades < 10:
            return AllocationAdjustment(
                strategy_a_weight=current_a_weight,
                strategy_b_weight=current_b_weight,
                reason="Not enough trades per strategy for adjustment",
            )

        # Compare Sharpe ratios
        a_sharpe = a_metrics.sharpe_ratio
        b_sharpe = b_metrics.sharpe_ratio

        # Calculate adjustment direction
        if a_sharpe > b_sharpe and a_sharpe > 0:
            # A is performing better → increase A
            shift = min(0.10, (a_sharpe - b_sharpe) * 0.05)
            new_a = min(0.90, current_a_weight + shift)
            new_b = 1.0 - new_a
            reason = f"A outperforming (Sharpe: A={a_sharpe:.2f}, B={b_sharpe:.2f})"
        elif b_sharpe > a_sharpe and b_sharpe > 0:
            shift = min(0.10, (b_sharpe - a_sharpe) * 0.05)
            new_b = min(0.50, current_b_weight + shift)
            new_a = 1.0 - new_b
            reason = f"B outperforming (Sharpe: A={a_sharpe:.2f}, B={b_sharpe:.2f})"
        else:
            new_a = current_a_weight
            new_b = current_b_weight
            reason = "No clear winner, keeping current allocation"

        # Enforce minimums
        new_a = max(0.50, min(0.90, new_a))
        new_b = max(0.10, min(0.50, new_b))
        # Normalize
        total = new_a + new_b
        new_a /= total
        new_b /= total

        return AllocationAdjustment(
            strategy_a_weight=new_a,
            strategy_b_weight=new_b,
            reason=reason,
        )

"""
Layer 4: ML Confidence Model.

XGBoost classifier that predicts trade profitability based on technical features.
Outputs a confidence score (0.5-1.5) used as position size multiplier.
Falls back to 1.0 (neutral) when no model is trained or data is insufficient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("xgboost not installed, ML confidence model disabled")


@dataclass
class FeatureRow:
    """A single row of features for the ML model."""
    atr_pct: float
    adx: float
    rsi: float
    bb_width: float
    volume_ratio: float
    ema_20_slope: float
    ema_50_slope: float
    donchian_position: float  # where price sits in donchian channel (0=low, 1=high)
    funding_rate: float
    profitable: Optional[bool] = None  # label; None at prediction time

    def to_feature_array(self) -> list[float]:
        """Return feature values as a list (no label)."""
        return [
            self.atr_pct, self.adx, self.rsi, self.bb_width,
            self.volume_ratio, self.ema_20_slope, self.ema_50_slope,
            self.donchian_position, self.funding_rate,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "atr_pct", "adx", "rsi", "bb_width",
            "volume_ratio", "ema_20_slope", "ema_50_slope",
            "donchian_position", "funding_rate",
        ]


class MLConfidenceModel:
    """XGBoost-based trade confidence predictor."""

    def __init__(self, min_samples: int = 30, default_confidence: float = 1.0):
        self.min_samples = min_samples
        self.default_confidence = default_confidence
        self._model = None
        self._accuracy: Optional[float] = None
        self._disabled = False

    def train(self, rows: list[FeatureRow]) -> bool:
        """
        Train the model on historical feature rows.

        Returns True if training succeeded, False if insufficient data or error.
        """
        if not HAS_XGBOOST:
            logger.warning("Cannot train: xgboost not installed")
            return False

        labeled = [r for r in rows if r.profitable is not None]
        if len(labeled) < self.min_samples:
            logger.info(f"Insufficient data for training: {len(labeled)} < {self.min_samples}")
            return False

        X = np.array([r.to_feature_array() for r in labeled])
        y = np.array([1 if r.profitable else 0 for r in labeled])

        # Replace NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Check we have both classes
        if len(np.unique(y)) < 2:
            logger.warning("Training data has only one class, skipping")
            return False

        try:
            # Walk-forward: use first 80% to train, last 20% to validate
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(X_train, y_train)

            # Validate — skip evaluation if validation set has only one class
            if len(np.unique(y_val)) < 2:
                logger.warning("Validation set has only one class, skipping evaluation — keeping model")
                self._model = model
                self._accuracy = None
                self._disabled = False
                return True

            val_preds = model.predict(X_val)
            self._accuracy = float(np.mean(val_preds == y_val))

            # Only keep model if accuracy > 50% (better than random)
            if self._accuracy <= 0.5:
                logger.warning(f"Model accuracy {self._accuracy:.1%} <= 50%, discarding")
                self._model = None
                return False

            self._model = model
            self._disabled = False
            logger.info(f"ML model trained: accuracy={self._accuracy:.1%} on {len(X_val)} validation samples")
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._model = None
            return False

    def predict_confidence(self, row: FeatureRow) -> float:
        """
        Predict confidence for a potential trade.

        Returns a multiplier between 0.5 and 1.5:
        - 1.0 = neutral (no model or disabled)
        - >1.0 = model thinks trade is likely profitable
        - <1.0 = model thinks trade is less likely profitable
        """
        if self._model is None or self._disabled:
            return self.default_confidence

        try:
            X = np.array([row.to_feature_array()])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Get probability of being profitable
            proba = self._model.predict_proba(X)[0]
            # proba[1] = probability of class 1 (profitable)
            prob_profitable = float(proba[1])

            # Map [0, 1] probability to [0.5, 1.5] confidence
            # 0.0 prob → 0.5 confidence
            # 0.5 prob → 1.0 confidence (neutral)
            # 1.0 prob → 1.5 confidence
            confidence = 0.5 + prob_profitable

            return max(0.5, min(confidence, 1.5))

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self.default_confidence

    def get_accuracy(self) -> Optional[float]:
        """Return validation accuracy from last training, or None."""
        return self._accuracy

    def disable(self) -> None:
        """Disable the model (revert to default confidence)."""
        self._disabled = True
        logger.info("ML confidence model disabled")

    def enable(self) -> None:
        """Re-enable the model."""
        self._disabled = False

    @property
    def is_active(self) -> bool:
        return self._model is not None and not self._disabled

    @staticmethod
    def extract_features(
        df: pd.DataFrame,
        funding_rate: Optional[float] = None,
        profitable: Optional[bool] = None,
    ) -> Optional[FeatureRow]:
        """
        Extract a FeatureRow from the latest row of a feature DataFrame.

        Returns None if required columns are missing.
        """
        required = ["atr_pct", "adx", "rsi", "bb_width", "volume_ratio",
                     "ema_20_slope", "ema_50_slope", "donchian_high", "donchian_low", "close"]

        if not all(col in df.columns for col in required):
            return None

        if len(df) == 0:
            return None

        last = df.iloc[-1]

        # Donchian position: where close sits in channel (0=at low, 1=at high)
        d_range = last["donchian_high"] - last["donchian_low"]
        if d_range > 0:
            donchian_pos = (last["close"] - last["donchian_low"]) / d_range
        else:
            donchian_pos = 0.5

        return FeatureRow(
            atr_pct=float(last["atr_pct"]) if not np.isnan(last["atr_pct"]) else 0.0,
            adx=float(last["adx"]) if not np.isnan(last["adx"]) else 0.0,
            rsi=float(last["rsi"]) if not np.isnan(last["rsi"]) else 50.0,
            bb_width=float(last["bb_width"]) if not np.isnan(last["bb_width"]) else 0.0,
            volume_ratio=float(last["volume_ratio"]) if not np.isnan(last["volume_ratio"]) else 1.0,
            ema_20_slope=float(last["ema_20_slope"]) if not np.isnan(last["ema_20_slope"]) else 0.0,
            ema_50_slope=float(last["ema_50_slope"]) if not np.isnan(last["ema_50_slope"]) else 0.0,
            donchian_position=float(donchian_pos),
            funding_rate=funding_rate or 0.0,
            profitable=profitable,
        )

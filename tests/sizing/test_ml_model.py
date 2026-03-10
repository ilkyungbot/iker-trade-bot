"""Tests for ML confidence model."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

from sizing.ml_model import MLConfidenceModel, FeatureRow


def _make_feature_rows(n: int = 50, profitable_ratio: float = 0.6) -> list[FeatureRow]:
    """Create synthetic feature rows for testing."""
    np.random.seed(42)
    rows = []
    for i in range(n):
        profitable = i < int(n * profitable_ratio)
        rows.append(FeatureRow(
            atr_pct=np.random.uniform(0.01, 0.05),
            adx=np.random.uniform(15, 50) if profitable else np.random.uniform(10, 25),
            rsi=np.random.uniform(30, 70),
            bb_width=np.random.uniform(0.02, 0.08),
            volume_ratio=np.random.uniform(0.5, 2.0),
            ema_20_slope=np.random.uniform(-5, 5),
            ema_50_slope=np.random.uniform(-3, 3),
            donchian_position=np.random.uniform(0, 1),
            funding_rate=np.random.uniform(-0.001, 0.001),
            profitable=profitable,
        ))
    return rows


class TestFeatureRow:
    def test_to_array(self):
        row = FeatureRow(
            atr_pct=0.02, adx=30, rsi=50, bb_width=0.04,
            volume_ratio=1.2, ema_20_slope=2.0, ema_50_slope=1.0,
            donchian_position=0.5, funding_rate=0.0001, profitable=True,
        )
        arr = row.to_feature_array()
        assert len(arr) == 9  # 9 features, no label
        assert arr[0] == 0.02
        assert arr[1] == 30

    def test_feature_names(self):
        names = FeatureRow.feature_names()
        assert len(names) == 9
        assert "atr_pct" in names
        assert "profitable" not in names


class TestMLConfidenceModel:
    def test_predict_without_training_returns_default(self):
        model = MLConfidenceModel()
        row = _make_feature_rows(1)[0]
        confidence = model.predict_confidence(row)
        assert confidence == 1.0  # default when no model

    def test_train_with_insufficient_data(self):
        model = MLConfidenceModel(min_samples=30)
        rows = _make_feature_rows(10)
        trained = model.train(rows)
        assert trained is False
        assert model._model is None

    def test_train_with_sufficient_data(self):
        model = MLConfidenceModel(min_samples=30)
        rows = _make_feature_rows(100)  # need enough for accuracy > 50%
        trained = model.train(rows)
        assert trained is True
        assert model._model is not None

    def test_predict_after_training(self):
        model = MLConfidenceModel(min_samples=30)
        rows = _make_feature_rows(100)
        model.train(rows)

        row = rows[0]
        confidence = model.predict_confidence(row)
        # Confidence should be between 0.5 and 1.5
        assert 0.5 <= confidence <= 1.5

    def test_confidence_range_clamped(self):
        model = MLConfidenceModel(min_samples=30)
        rows = _make_feature_rows(100)
        model.train(rows)

        for row in rows[:20]:
            conf = model.predict_confidence(row)
            assert 0.5 <= conf <= 1.5

    def test_model_accuracy_check(self):
        model = MLConfidenceModel(min_samples=30)
        rows = _make_feature_rows(100)
        model.train(rows)

        accuracy = model.get_accuracy()
        assert accuracy is not None
        assert 0.0 <= accuracy <= 1.0

    def test_disabled_model_returns_default(self):
        model = MLConfidenceModel(min_samples=30)
        rows = _make_feature_rows(100)
        model.train(rows)
        model.disable()

        row = rows[0]
        confidence = model.predict_confidence(row)
        assert confidence == 1.0

    def test_extract_features_from_dataframe(self):
        """Test extracting FeatureRow from a DataFrame with all indicators."""
        n = 50
        df = pd.DataFrame({
            "close": np.linspace(100, 110, n),
            "atr_pct": np.random.uniform(0.01, 0.05, n),
            "adx": np.random.uniform(15, 50, n),
            "rsi": np.random.uniform(30, 70, n),
            "bb_width": np.random.uniform(0.02, 0.08, n),
            "volume_ratio": np.random.uniform(0.5, 2.0, n),
            "ema_20_slope": np.random.uniform(-5, 5, n),
            "ema_50_slope": np.random.uniform(-3, 3, n),
            "donchian_high": np.linspace(105, 115, n),
            "donchian_low": np.linspace(95, 105, n),
        })

        row = MLConfidenceModel.extract_features(df, funding_rate=0.0005)
        assert isinstance(row, FeatureRow)
        assert row.funding_rate == 0.0005
        assert row.profitable is None  # unknown at prediction time

    def test_extract_features_missing_columns_returns_none(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        row = MLConfidenceModel.extract_features(df)
        assert row is None

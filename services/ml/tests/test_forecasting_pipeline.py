"""
CivicPulse AI — Phase 3, Stage 6: Forecasting Pipeline & Gate Verification Tests
Verifies:
1. Chronological splitting preserves temporal order without future data leakage.
2. Time-series metrics calculate correct mathematical values and guard against zero division.
3. Baseline forecasters (Naive, SMA, Linear Trend) produce expected values.
4. Data readiness gate audits 66 real incidents, resolves 58 BLR / 8 DEL, recognizes 5.8h span,
   and strictly halts with DATA_INSUFFICIENT.
5. BiLSTM deep neural forecaster is strictly gated.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from models.forecasting.chronological_eval import (
    ChronologicalSplitter,
    TimeSeriesMetrics,
    NaiveBaselineForecaster,
    MovingAverageForecaster,
    LinearTrendForecaster,
)
from training.train_forecasting_experimental import (
    ExperimentalForecastingPipeline,
    MIN_DAILY_PERIODS_THRESHOLD,
    MIN_MONTHLY_PERIODS_THRESHOLD,
)


class TestChronologicalEvaluation:
    """Tests chronological splitting and evaluation metrics."""

    def test_chronological_splitting_strictly_temporal(self):
        splitter = ChronologicalSplitter(min_train_size=3, forecast_horizon=2, step_size=1)
        splits = splitter.split(n_samples=8)

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            # Train must strictly precede test
            assert np.max(train_idx) < np.min(test_idx)
            # Test size matches forecast horizon
            assert len(test_idx) == 2
            # Train starts from index 0 (expanding window)
            assert train_idx[0] == 0

    def test_chronological_splitting_empty_when_insufficient(self):
        splitter = ChronologicalSplitter(min_train_size=10, forecast_horizon=2)
        splits = splitter.split(n_samples=5)
        assert len(splits) == 0

    def test_metrics_evaluation_accuracy(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 39.0])

        metrics = TimeSeriesMetrics.evaluate_all(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(2.0, 0.01)
        assert metrics["rmse"] == pytest.approx(2.121, 0.01)
        assert metrics["forecast_bias"] == pytest.approx(0.5, 0.01)

    def test_mape_guards_against_zero_division(self):
        y_true = np.array([0.0, 0.0, 10.0])
        y_pred = np.array([1.0, 2.0, 10.0])

        mape = TimeSeriesMetrics.mean_absolute_percentage_error(y_true, y_pred)
        assert not np.isnan(mape)
        assert not np.isinf(mape)
        assert mape > 0


class TestBaselineForecasters:
    """Tests parametric and statistical forecasting baselines."""

    def test_naive_forecaster(self):
        y_train = np.array([5.0, 8.0, 12.0, 15.0])
        model = NaiveBaselineForecaster().fit(y_train)
        preds = model.predict(horizon=3)

        assert len(preds) == 3
        np.testing.assert_array_equal(preds, [15.0, 15.0, 15.0])

    def test_moving_average_forecaster(self):
        y_train = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        model = MovingAverageForecaster(window=3).fit(y_train)
        preds = model.predict(horizon=2)

        # Mean of [14, 16, 18] = 16
        assert len(preds) == 2
        np.testing.assert_array_almost_equal(preds, [16.0, 16.0])

    def test_linear_trend_forecaster(self):
        # Perfect slope of +2: y = 2*t + 10
        y_train = np.array([10.0, 12.0, 14.0, 16.0])  # t = 0, 1, 2, 3
        model = LinearTrendForecaster().fit(y_train)
        preds = model.predict(horizon=2)  # t = 4, 5 -> 18, 20

        assert len(preds) == 2
        np.testing.assert_array_almost_equal(preds, [18.0, 20.0])

    def test_linear_trend_non_negative_clamp(self):
        # Steep downward slope
        y_train = np.array([10.0, 5.0, 1.0])
        model = LinearTrendForecaster().fit(y_train)
        preds = model.predict(horizon=4)
        assert np.all(preds >= 0.0)


class TestForecastingReadinessGate:
    """Tests longitudinal data audit and governance gates."""

    @pytest.fixture
    def mock_66_real_incidents(self):
        base_time = datetime(2026, 11, 7, 11, 27, 4)
        incidents = []

        # 58 Bengaluru records
        for i in range(58):
            dt = base_time + timedelta(minutes=i * 6)  # spans ~5.8 hours
            incidents.append({
                "id": f"inc-blr-{i}",
                "latitude": 12.9716 + (i * 0.001),
                "longitude": 77.5946 + (i * 0.001),
                "damageType": "pothole",
                "reportedAt": dt.isoformat(),
            })

        # 8 Delhi records
        for i in range(8):
            dt = base_time + timedelta(minutes=i * 40)
            incidents.append({
                "id": f"inc-del-{i}",
                "latitude": 28.6139 + (i * 0.001),
                "longitude": 77.2090 + (i * 0.001),
                "damageType": "alligator_crack",
                "reportedAt": dt.isoformat(),
            })

        return incidents

    def test_audit_identifies_regional_census_and_temporal_span(self, mock_66_real_incidents):
        pipeline = ExperimentalForecastingPipeline()
        audit = pipeline.audit_incident_time_series(mock_66_real_incidents, frequency="daily")

        assert audit["total_incidents"] == 66
        assert audit["bengaluru_metro_count"] == 58
        assert audit["delhi_ncr_count"] == 8
        assert 5.0 <= audit["temporal_span_hours"] <= 6.5
        assert audit["temporal_span_days"] < 1.0
        assert audit["is_sufficient"] is False
        assert audit["status"] == "DATA_INSUFFICIENT"

    def test_training_experiment_halts_with_data_insufficient(self, mock_66_real_incidents):
        pipeline = ExperimentalForecastingPipeline()
        result = pipeline.run_training_experiment(
            mock_66_real_incidents, model_name="neural-forecaster-bilstm", frequency="daily"
        )

        assert result["success"] is False
        assert result["status"] == "DATA_INSUFFICIENT"
        assert result["model_gated"] == "neural-forecaster-bilstm"
        assert result["prohibition_enforced"] is True
        assert "Insufficient longitudinal observation depth" in result["message"]

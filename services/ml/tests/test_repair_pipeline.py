"""
Unit tests for Phase 3 Stage 4 Repair Cost & Duration ML pipeline and readiness gate.
"""

import pytest
import sys
from pathlib import Path

# Add services/ml to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_repair_regression_experimental import (
    ExperimentalRepairRegressionPipeline,
    MIN_COMPLETED_REPAIRS_THRESHOLD,
)


def test_repair_regression_readiness_gate_insufficient():
    pipeline = ExperimentalRepairRegressionPipeline(min_threshold=50)

    # 2 tasks with no actuals
    tasks = [
        {"id": "t1", "taskStatus": "pending", "actualCost": None, "actualDuration": None},
        {"id": "t2", "taskStatus": "in_progress", "actualCost": None, "actualDuration": None},
    ]

    res = pipeline.run_training_experiment(tasks)

    assert res["status"] == "DATA_INSUFFICIENT"
    assert res["success"] is False
    assert res["audit"]["is_sufficient"] is False
    assert res["audit"]["total_completed_repairs_with_actuals"] == 0
    assert "circular self-prediction" in res["audit"]["decision_rationale"]


def test_repair_regression_readiness_gate_sufficient():
    pipeline = ExperimentalRepairRegressionPipeline(min_threshold=50)

    # 52 completed tasks with actual costs and durations
    tasks = [
        {
            "id": f"t{i}",
            "taskStatus": "completed",
            "actualCost": 12500.0 + i * 100,
            "actualDuration": 6 + (i % 4),
        }
        for i in range(52)
    ]

    res = pipeline.run_training_experiment(tasks)

    assert res["status"] == "TRAINED"
    assert res["success"] is True


def test_repair_threshold_constant():
    assert MIN_COMPLETED_REPAIRS_THRESHOLD == 50


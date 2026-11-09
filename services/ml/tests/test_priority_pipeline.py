"""
Unit tests for Phase 3 Stage 3 Maintenance Priority ML pipeline and readiness gate.
"""

import pytest
import sys
from pathlib import Path

# Add services/ml to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_priority_experimental import ExperimentalPriorityPipeline, MIN_COMPLETED_REPAIRS_THRESHOLD


def test_priority_readiness_gate_insufficient():
    pipeline = ExperimentalPriorityPipeline(min_threshold=50)

    # 2 tasks, 0 completed repairs
    tasks = [{"id": "t1"}, {"id": "t2"}]
    completed = []
    reviews = []

    res = pipeline.run_training_experiment(tasks, completed, reviews)

    assert res["status"] == "DATA_INSUFFICIENT"
    assert res["success"] is False
    assert res["audit"]["is_sufficient"] is False
    assert res["audit"]["total_completed_repairs"] == 0
    assert "circular proxy" in res["audit"]["decision_rationale"]


def test_priority_readiness_gate_sufficient():
    pipeline = ExperimentalPriorityPipeline(min_threshold=50)

    # 55 completed repairs
    tasks = [{"id": f"t{i}"} for i in range(55)]
    completed = [{"id": f"t{i}", "taskStatus": "completed"} for i in range(55)]
    reviews = [{"id": f"r{i}"} for i in range(55)]

    res = pipeline.run_training_experiment(tasks, completed, reviews)

    assert res["status"] == "TRAINED"
    assert res["success"] is True


def test_priority_threshold_constant():
    assert MIN_COMPLETED_REPAIRS_THRESHOLD == 50


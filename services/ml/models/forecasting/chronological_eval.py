"""
CivicPulse AI — Phase 3, Stage 6: Chronological Time-Series Evaluation & Baselines
Enforces strict temporal order in splitting and evaluation:
1. No future data leakage.
2. Expanding window (rolling origin) evaluation.
3. Standard time-series metrics (MAE, RMSE, MAPE, Bias).
4. Parametric and statistical baselines (Naive, SMA, Linear Trend).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class ChronologicalSplitter:
    """
    Chronological expanding-window time-series splitter.
    Guarantees that test observations strictly follow training observations chronologically.
    """

    def __init__(self, min_train_size: int = 3, forecast_horizon: int = 1, step_size: int = 1):
        self.min_train_size = min_train_size
        self.forecast_horizon = forecast_horizon
        self.step_size = step_size

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates (train_indices, test_indices) tuples respecting temporal order.
        """
        if n_samples < self.min_train_size + self.forecast_horizon:
            return []

        splits = []
        for t in range(self.min_train_size, n_samples - self.forecast_horizon + 1, self.step_size):
            train_idx = np.arange(0, t)
            test_idx = np.arange(t, t + self.forecast_horizon)
            splits.append((train_idx, test_idx))

        return splits


class TimeSeriesMetrics:
    """
    Evaluates forecasting error metrics with safety guards against zero divisions.
    """

    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
        # Avoid division by zero on zero-incident intervals
        denominator = np.maximum(y_true, epsilon)
        return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)

    @staticmethod
    def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Positive indicates systematic overprediction, negative indicates underprediction."""
        return float(np.mean(y_pred - y_true))

    @classmethod
    def evaluate_all(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        return {
            "mae": round(cls.mean_absolute_error(y_t, y_p), 3),
            "rmse": round(cls.root_mean_squared_error(y_t, y_p), 3),
            "mape": round(cls.mean_absolute_percentage_error(y_t, y_p), 2),
            "forecast_bias": round(cls.forecast_bias(y_t, y_p), 3),
        }


class NaiveBaselineForecaster:
    """Carries forward the last observed historical value."""

    def __init__(self):
        self.last_value: Optional[float] = None

    def fit(self, y_train: np.ndarray) -> "NaiveBaselineForecaster":
        if len(y_train) == 0:
            raise ValueError("Training series cannot be empty.")
        self.last_value = float(y_train[-1])
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.last_value is None:
            raise RuntimeError("Model must be fit before predict.")
        return np.full(horizon, self.last_value)


class MovingAverageForecaster:
    """Computes unweighted rolling mean of the last w observations."""

    def __init__(self, window: int = 3):
        self.window = window
        self.mean_val: Optional[float] = None

    def fit(self, y_train: np.ndarray) -> "MovingAverageForecaster":
        if len(y_train) == 0:
            raise ValueError("Training series cannot be empty.")
        w = min(self.window, len(y_train))
        self.mean_val = float(np.mean(y_train[-w:]))
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.mean_val is None:
            raise RuntimeError("Model must be fit before predict.")
        return np.full(horizon, self.mean_val)


class LinearTrendForecaster:
    """Parametric linear regression OLS fit over sequential time indices."""

    def __init__(self):
        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None
        self.n_train: int = 0

    def fit(self, y_train: np.ndarray) -> "LinearTrendForecaster":
        n = len(y_train)
        if n < 2:
            raise ValueError("At least 2 points required for linear trend.")
        t = np.arange(n)
        t_mean = float(np.mean(t))
        y_mean = float(np.mean(y_train))

        num = float(np.sum((t - t_mean) * (y_train - y_mean)))
        den = float(np.sum((t - t_mean) ** 2))

        self.beta = num / den if den > 0 else 0.0
        self.alpha = y_mean - self.beta * t_mean
        self.n_train = n
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.alpha is None or self.beta is None:
            raise RuntimeError("Model must be fit before predict.")
        future_t = np.arange(self.n_train, self.n_train + horizon)
        preds = self.alpha + self.beta * future_t
        return np.maximum(0.0, preds)


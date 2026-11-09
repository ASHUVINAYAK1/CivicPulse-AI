"""
CivicPulse AI - Tabular Preprocessing Pipeline

Encodes structured features for classical ML models (Logistic Regression,
Random Forest, Gradient Boosting) predicting repair urgency, priority, and cost.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class StructuredIncidentTransformer(BaseEstimator, TransformerMixin):
    """Encodes and scales tabular road incident features."""

    DAMAGE_TYPES = ["D00", "D10", "D20", "D40", "other"]
    SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols = ["latitude", "longitude", "incident_age_days", "nearby_incident_count"]
        self.is_fitted = False

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives engineered attributes without target leakage."""
        data = df.copy()

        # Handle missing incident age or compute from reported timestamp
        if "incident_age_days" not in data.columns:
            data["incident_age_days"] = 0.0

        if "nearby_incident_count" not in data.columns:
            data["nearby_incident_count"] = 0

        # One-hot encode damage types
        for dt in self.DAMAGE_TYPES:
            col_name = f"damage_type_{dt}"
            data[col_name] = (data["damage_type"] == dt).astype(float)

        # Map ordinal verified severity if present
        severity_map = {"low": 1.0, "medium": 2.0, "high": 3.0, "critical": 4.0}
        data["severity_score"] = data["severity"].map(severity_map).fillna(1.0)

        return data

    def fit(self, X: pd.DataFrame, y=None):
        """Fits scalers on numerical training features."""
        transformed = self._extract_features(X)
        self.scaler.fit(transformed[self.numeric_cols])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame into model-ready feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("StructuredIncidentTransformer must be fitted before transforming.")

        transformed = self._extract_features(X)
        scaled_numerics = self.scaler.transform(transformed[self.numeric_cols])

        # Concatenate scaled numerics with encoded damage categories and severity score
        encoded_damage = transformed[[f"damage_type_{dt}" for dt in self.DAMAGE_TYPES]].values
        severity_col = transformed[["severity_score"]].values

        return np.hstack([scaled_numerics, encoded_damage, severity_col])


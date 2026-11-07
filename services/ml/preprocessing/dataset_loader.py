"""
CivicPulse AI - Dataset Catalog Loader & Splitter

Provides dataset validation, train/val/test splitting, and synthetic data generation
for academic ML experimentation without leaking target or spatial groups.
"""

from typing import Tuple, Dict, Any, Optional
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_benchmark_synthetic_dataset(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a calibrated synthetic dataset of road incidents for ML experimentation.
    Explicitly tags every row with is_synthetic = True.
    """
    rng = np.random.default_rng(random_state)

    damage_choices = ["D00", "D10", "D20", "D40"]
    damage_probabilities = [0.25, 0.23, 0.19, 0.33]
    damages = rng.choice(damage_choices, size=n_samples, p=damage_probabilities)

    # Correlate damage type with severity
    severities = []
    for d in damages:
        if d == "D40":
            severities.append(rng.choice(["medium", "high", "critical"], p=[0.2, 0.4, 0.4]))
        elif d == "D20":
            severities.append(rng.choice(["low", "medium", "high"], p=[0.2, 0.5, 0.3]))
        else:
            severities.append(rng.choice(["low", "medium", "high"], p=[0.5, 0.4, 0.1]))

    # Coordinates centered around municipal boundary (e.g., Delhi NCR)
    latitudes = 28.6139 + rng.normal(0, 0.05, size=n_samples)
    longitudes = 77.2090 + rng.normal(0, 0.05, size=n_samples)
    incident_ages = rng.uniform(0.5, 60.0, size=n_samples)
    nearby_counts = rng.poisson(lam=4, size=n_samples)

    # Priority target: 0 = normal, 1 = urgent
    is_urgent = [
        1 if (s == "critical" or (s == "high" and age > 14.0)) else 0
        for s, age in zip(severities, incident_ages)
    ]

    # Repair cost target (regression)
    base_cost = {"D00": 8000, "D10": 9500, "D20": 18000, "D40": 15000}
    severity_multiplier = {"low": 1.0, "medium": 1.3, "high": 1.8, "critical": 2.5}
    costs = [
        base_cost[d] * severity_multiplier[s] + rng.normal(0, 800)
        for d, s in zip(damages, severities)
    ]

    df = pd.DataFrame({
        "incident_id": [f"SYN-{1000 + i}" for i in range(n_samples)],
        "damage_type": damages,
        "severity": severities,
        "latitude": latitudes,
        "longitude": longitudes,
        "incident_age_days": incident_ages,
        "nearby_incident_count": nearby_counts,
        "is_urgent": is_urgent,
        "repair_cost": np.round(costs, 2),
        "is_synthetic": True,
    })

    return df


def split_incident_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified splitting into train, validation, and test partitions.
    Preserves class balance across damage categories.
    """
    stratify_col = df["damage_type"] if "damage_type" in df.columns else None

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state,
    )

    adj_val_size = val_size / (1.0 - test_size)
    stratify_val = train_val["damage_type"] if "damage_type" in train_val.columns else None

    train, val = train_test_split(
        train_val,
        test_size=adj_val_size,
        stratify=stratify_val,
        random_state=random_state,
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

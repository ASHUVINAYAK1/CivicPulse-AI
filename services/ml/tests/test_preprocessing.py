"""
Tests for preprocessing and evaluation metrics.
"""

import numpy as np
import pandas as pd
from PIL import Image

from preprocessing.image_pipeline import ImagePreprocessor
from preprocessing.tabular_pipeline import StructuredIncidentTransformer
from preprocessing.dataset_loader import generate_benchmark_synthetic_dataset, split_incident_dataset
from evaluation.metrics import (
    evaluate_regression,
    evaluate_binary_classification,
    evaluate_multiclass_classification,
    evaluate_clustering,
    evaluate_forecasting,
)


def test_image_preprocessor():
    preprocessor = ImagePreprocessor(target_size=(224, 224), normalize=True)
    img = Image.new("RGB", (300, 200), color=(100, 150, 200))
    tensor = preprocessor.preprocess_image(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32


def test_tabular_transformer():
    df = generate_benchmark_synthetic_dataset(n_samples=50, random_state=42)
    transformer = StructuredIncidentTransformer()
    X = transformer.fit_transform(df)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 50
    assert X.shape[1] > 5


def test_dataset_split():
    df = generate_benchmark_synthetic_dataset(n_samples=100, random_state=42)
    train, val, test = split_incident_dataset(df, test_size=0.2, val_size=0.1)
    assert len(train) + len(val) + len(test) == 100
    assert len(test) == 20


def test_evaluation_metrics():
    # Regression
    reg = evaluate_regression(np.array([10.0, 20.0]), np.array([12.0, 19.0]))
    assert "mae" in reg and "rmse" in reg

    # Binary Classification
    cls_b = evaluate_binary_classification(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert "accuracy" in cls_b and "f1_score" in cls_b

    # Multiclass Classification
    cls_m = evaluate_multiclass_classification(np.array([0, 1, 2]), np.array([0, 1, 1]))
    assert "macro_f1" in cls_m

    # Clustering
    X = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 5.0], [5.1, 5.2]])
    labels = np.array([0, 0, 1, 1])
    cl = evaluate_clustering(X, labels)
    assert "silhouette_score" in cl

    # Forecasting
    fc = evaluate_forecasting(np.array([100.0, 120.0]), np.array([105.0, 118.0]))
    assert "mape" in fc

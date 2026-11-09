"""
CivicPulse AI — Geographical Hotspot Clustering Unit Tests
Tests Spherical K-Means and Haversine DBSCAN implementations.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.clustering.geo_kmeans import (
    validate_coordinates,
    haversine_distance,
    latlon_to_cartesian_unit,
    cartesian_unit_to_latlon,
    SphericalKMeans,
    compute_geographic_silhouette,
    compare_with_sklearn,
)
from models.clustering.geo_dbscan import GeographicDBSCAN


def test_haversine_known_distances():
    # India Gate to Qutub Minar (~10.74 km)
    d = haversine_distance(28.6129, 77.2295, 28.5244, 77.1855)
    assert 10500 <= d <= 11000

    # Delhi to Mumbai (~1148 km)
    d_mumbai = haversine_distance(28.6139, 77.2090, 19.0760, 72.8777)
    assert 1140000 <= d_mumbai <= 1160000

    # Distance to self is 0
    assert haversine_distance(28.6, 77.2, 28.6, 77.2) == 0.0


def test_unit_sphere_cartesian_roundtrip():
    coords = np.array([
        [28.6139, 77.2090],
        [19.0760, 72.8777],
        [0.0, 0.0],
        [89.5, 45.0],
        [-45.0, -120.0],
    ])

    cart = latlon_to_cartesian_unit(coords)
    norms = np.linalg.norm(cart, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    inverted = cartesian_unit_to_latlon(cart)
    assert np.allclose(coords, inverted, atol=1e-4)


def test_coordinate_validation():
    assert validate_coordinates(28.6, 77.2)[0] is True
    assert validate_coordinates(95.0, 77.2)[0] is False
    assert validate_coordinates(-95.0, 77.2)[0] is False
    assert validate_coordinates(28.6, 195.0)[0] is False
    assert validate_coordinates(None, 77.2)[0] is False
    assert validate_coordinates(np.nan, 77.2)[0] is False


def test_spherical_kmeans_convergence():
    rng = np.random.default_rng(42)
    # 3 artificial clusters around Delhi
    c1 = np.column_stack([28.61 + rng.normal(0, 0.01, 20), 77.20 + rng.normal(0, 0.01, 20)])
    c2 = np.column_stack([28.70 + rng.normal(0, 0.01, 20), 77.10 + rng.normal(0, 0.01, 20)])
    c3 = np.column_stack([28.50 + rng.normal(0, 0.01, 20), 77.30 + rng.normal(0, 0.01, 20)])
    coords = np.vstack([c1, c2, c3])

    model = SphericalKMeans(k=3, random_state=42).fit(coords)
    assert model.converged_ is True
    assert model.n_iter_ < 20
    assert len(model.centroids_latlon) == 3
    assert model.inertia_ > 0

    sil = compute_geographic_silhouette(coords, model.labels)
    assert sil > 0.70

    # Prediction
    pred = model.predict(coords[:5])
    assert len(pred) == 5


def test_geographic_dbscan():
    rng = np.random.default_rng(42)
    # Dense cluster of 15 points within ~200m
    c1 = np.column_stack([28.6139 + rng.normal(0, 0.001, 15), 77.2090 + rng.normal(0, 0.001, 15)])
    # 2 isolated noise points 10 km away
    noise = np.array([[28.75, 77.40], [28.40, 77.05]])
    coords = np.vstack([c1, noise])

    db = GeographicDBSCAN(eps_meters=500.0, min_samples=3).fit(coords)
    assert db.n_clusters_ >= 1
    assert db.n_noise_ == 2
    assert db.labels[-1] == -1
    assert db.labels[-2] == -1
    assert db.point_types[-1] == "noise"
    assert "Noise / isolated incident" in db.diagnostics[-1]["classification_reason"]


def test_sklearn_comparison():
    coords = np.array([
        [28.61, 77.20],
        [28.62, 77.21],
        [28.70, 77.10],
        [28.71, 77.11],
    ])
    res = compare_with_sklearn(coords, k=2, random_state=42)
    assert "spherical_kmeans_from_scratch" in res
    assert "sklearn_cartesian_3d" in res
    assert "sklearn_euclidean_degrees" in res


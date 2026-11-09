"""
CivicPulse AI — Spherical K-Means from Scratch
Implements spherical k-means on WGS84 geographic coordinates using unit-sphere
projections and Haversine distance metrics.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans

EARTH_RADIUS_METERS = 6371000.0  # Mean radius of Earth in WGS84


def validate_coordinates(lat: float, lon: float) -> Tuple[bool, Optional[str]]:
    """Validates WGS84 latitude and longitude ranges."""
    if lat is None or lon is None:
        return False, "Coordinate values cannot be null"
    if not (isinstance(lat, (int, float, np.floating)) and isinstance(lon, (int, float, np.floating))):
        return False, "Coordinate values must be numerical"
    if np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon):
        return False, "Coordinate values cannot be NaN or Infinite"
    if not (-90.0 <= lat <= 90.0):
        return False, f"Latitude {lat} out of valid range [-90.0, 90.0]"
    if not (-180.0 <= lon <= 180.0):
        return False, f"Longitude {lon} out of valid range [-180.0, 180.0]"
    return True, None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes exact great-circle distance (meters) between two WGS84 points
    using the numerically stable Haversine formula.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return float(EARTH_RADIUS_METERS * c)


def pairwise_haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Computes NxN pairwise Haversine distance matrix in meters.
    coords is an Nx2 array of [latitude, longitude].
    """
    n = len(coords)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    lats = np.radians(coords[:, 0])
    lons = np.radians(coords[:, 1])

    for i in range(n):
        delta_phi = lats - lats[i]
        delta_lambda = lons - lons[i]
        a = (np.sin(delta_phi / 2.0) ** 2 +
             np.cos(lats[i]) * np.cos(lats) * (np.sin(delta_lambda / 2.0) ** 2))
        c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
        dist_matrix[i, :] = EARTH_RADIUS_METERS * c

    return dist_matrix


def latlon_to_cartesian_unit(coords: np.ndarray) -> np.ndarray:
    """
    Converts Nx2 (lat, lon) in degrees to Nx3 Cartesian unit-sphere coordinates:
    x = cos(lat) * cos(lon)
    y = cos(lat) * sin(lon)
    z = sin(lat)
    """
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack([x, y, z])


def cartesian_unit_to_latlon(cart: np.ndarray) -> np.ndarray:
    """
    Converts Nx3 Cartesian unit vectors back to Nx2 (lat, lon) degrees:
    lat = arcsin(z)
    lon = arctan2(y, x)
    """
    norms = np.linalg.norm(cart, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = cart / norms

    x = normalized[:, 0]
    y = normalized[:, 1]
    z = np.clip(normalized[:, 2], -1.0, 1.0)

    lat_rad = np.arcsin(z)
    lon_rad = np.arctan2(y, x)

    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    return np.column_stack([lat_deg, lon_deg])


def compute_geographic_silhouette(coords: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the silhouette score [-1.0, 1.0] using exact pairwise Haversine
    distances (meters). Handles single-cluster or noise edge cases.
    """
    n = len(coords)
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    if len(unique_labels) <= 1 or len(unique_labels) >= n:
        return 0.0

    dist_matrix = pairwise_haversine_matrix(coords)
    silhouette_vals = np.zeros(n)

    for i in range(n):
        own_label = labels[i]
        if own_label < 0:
            continue

        own_mask = (labels == own_label)
        own_count = np.sum(own_mask)
        if own_count <= 1:
            silhouette_vals[i] = 0.0
            continue

        # Mean intra-cluster distance a(i)
        a_i = np.sum(dist_matrix[i, own_mask]) / (own_count - 1)

        # Mean nearest-cluster distance b(i)
        b_i = np.inf
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_mask = (labels == other_label)
            other_dist = np.mean(dist_matrix[i, other_mask])
            if other_dist < b_i:
                b_i = other_dist

        max_ab = max(a_i, b_i)
        if max_ab > 0:
            silhouette_vals[i] = (b_i - a_i) / max_ab
        else:
            silhouette_vals[i] = 0.0

    valid_mask = labels >= 0
    if np.sum(valid_mask) == 0:
        return 0.0
    return float(np.mean(silhouette_vals[valid_mask]))


class SphericalKMeans:
    """
    K-Means clustering implemented from scratch on the unit sphere for WGS84 coordinates.
    Avoids Euclidean distortion on degree coordinates.
    """

    def __init__(
        self,
        k: int = 3,
        max_iter: int = 100,
        tol_meters: float = 1.0,
        random_state: int = 42,
        init_method: str = "k-means++",
    ):
        self.k = k
        self.max_iter = max_iter
        self.tol_meters = tol_meters
        self.random_state = random_state
        self.init_method = init_method

        self.centroids_latlon: Optional[np.ndarray] = None
        self.centroids_cart: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0
        self.converged_: bool = False

    def _init_centroids(self, coords: np.ndarray, cart: np.ndarray) -> np.ndarray:
        """Initializes K centroids using k-means++ or random sampling on sphere."""
        rng = np.random.default_rng(self.random_state)
        n = len(coords)

        if self.init_method == "random":
            chosen_indices = rng.choice(n, size=self.k, replace=False)
            return cart[chosen_indices].copy()

        # Spherical k-means++ initialization
        first_idx = rng.integers(0, n)
        chosen_indices = [first_idx]

        for _ in range(1, self.k):
            min_dist_sq = []
            for i in range(n):
                dists = [
                    haversine_distance(coords[i, 0], coords[i, 1], coords[c_idx, 0], coords[c_idx, 1])
                    for c_idx in chosen_indices
                ]
                min_dist_sq.append(min(dists) ** 2)

            min_dist_sq = np.array(min_dist_sq, dtype=np.float64)
            total_sum = np.sum(min_dist_sq)
            if total_sum > 0:
                probs = min_dist_sq / total_sum
                next_idx = rng.choice(n, p=probs)
            else:
                remaining = [idx for idx in range(n) if idx not in chosen_indices]
                next_idx = rng.choice(remaining) if remaining else 0
            chosen_indices.append(next_idx)

        return cart[chosen_indices].copy()

    def fit(self, coords: np.ndarray) -> "SphericalKMeans":
        """
        Fits spherical K-Means on Nx2 [latitude, longitude] array.
        """
        coords = np.asarray(coords, dtype=np.float64)
        n = len(coords)
        if n < self.k:
            raise ValueError(f"Number of samples ({n}) cannot be less than k ({self.k})")

        # 1. Convert to 3D Cartesian unit vectors
        cart = latlon_to_cartesian_unit(coords)

        # 2. Initialize centroids
        centroids_cart = self._init_centroids(coords, cart)
        centroids_latlon = cartesian_unit_to_latlon(centroids_cart)

        labels = np.zeros(n, dtype=int)
        converged = False
        iteration = 0

        for it in range(self.max_iter):
            iteration = it + 1
            # Assignment step using Haversine distance
            dist_to_centroids = np.zeros((n, self.k), dtype=np.float64)
            for k_idx in range(self.k):
                c_lat = centroids_latlon[k_idx, 0]
                c_lon = centroids_latlon[k_idx, 1]
                for i in range(n):
                    dist_to_centroids[i, k_idx] = haversine_distance(
                        coords[i, 0], coords[i, 1], c_lat, c_lon
                    )

            new_labels = np.argmin(dist_to_centroids, axis=1)

            # Update centroids by averaging unit vectors on sphere
            new_centroids_cart = np.zeros((self.k, 3), dtype=np.float64)
            for k_idx in range(self.k):
                cluster_members = cart[new_labels == k_idx]
                if len(cluster_members) > 0:
                    mean_vector = np.mean(cluster_members, axis=0)
                    norm = np.linalg.norm(mean_vector)
                    if norm > 1e-9:
                        new_centroids_cart[k_idx] = mean_vector / norm
                    else:
                        new_centroids_cart[k_idx] = cart[np.random.randint(0, n)]
                else:
                    # Empty cluster handling: assign to furthest point from any centroid
                    max_dist_pt = np.argmax(np.min(dist_to_centroids, axis=1))
                    new_centroids_cart[k_idx] = cart[max_dist_pt].copy()

            new_centroids_latlon = cartesian_unit_to_latlon(new_centroids_cart)

            # Check convergence via maximum centroid shift in meters
            shifts = [
                haversine_distance(
                    centroids_latlon[k_idx, 0], centroids_latlon[k_idx, 1],
                    new_centroids_latlon[k_idx, 0], new_centroids_latlon[k_idx, 1]
                )
                for k_idx in range(self.k)
            ]
            max_shift = max(shifts)

            labels = new_labels
            centroids_cart = new_centroids_cart
            centroids_latlon = new_centroids_latlon

            if max_shift < self.tol_meters:
                converged = True
                break

        # Calculate final inertia (sum of squared Haversine distances in meters^2)
        total_inertia = 0.0
        for i in range(n):
            c_idx = labels[i]
            d = haversine_distance(
                coords[i, 0], coords[i, 1],
                centroids_latlon[c_idx, 0], centroids_latlon[c_idx, 1]
            )
            total_inertia += (d ** 2)

        self.centroids_latlon = centroids_latlon
        self.centroids_cart = centroids_cart
        self.labels = labels
        self.inertia_ = float(total_inertia)
        self.n_iter_ = iteration
        self.converged_ = converged
        return self

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Assigns new coordinates to nearest learned spherical centroid."""
        if self.centroids_latlon is None:
            raise RuntimeError("Model must be fitted before predict.")
        coords = np.asarray(coords, dtype=np.float64)
        n = len(coords)
        assignments = np.zeros(n, dtype=int)
        for i in range(n):
            dists = [
                haversine_distance(coords[i, 0], coords[i, 1], self.centroids_latlon[k, 0], self.centroids_latlon[k, 1])
                for k in range(self.k)
            ]
            assignments[i] = int(np.argmin(dists))
        return assignments

    def get_cluster_summaries(self, coords: np.ndarray) -> List[Dict[str, Any]]:
        """Returns statistical metrics per cluster."""
        summaries = []
        for k_idx in range(self.k):
            mask = (self.labels == k_idx)
            members = coords[mask]
            count = int(np.sum(mask))
            c_lat = float(self.centroids_latlon[k_idx, 0])
            c_lon = float(self.centroids_latlon[k_idx, 1])

            if count > 0:
                dists = [
                    haversine_distance(m[0], m[1], c_lat, c_lon)
                    for m in members
                ]
                avg_dist = float(np.mean(dists))
                max_dist = float(np.max(dists))
            else:
                avg_dist = 0.0
                max_dist = 0.0

            summaries.append({
                "cluster_index": k_idx,
                "centroid_latitude": round(c_lat, 6),
                "centroid_longitude": round(c_lon, 6),
                "member_count": count,
                "average_distance_meters": round(avg_dist, 2),
                "max_distance_meters": round(max_dist, 2),
            })
        return summaries


def compare_with_sklearn(coords: np.ndarray, k: int = 3, random_state: int = 42) -> Dict[str, Any]:
    """
    Compares SphericalKMeans from scratch against Scikit-Learn KMeans:
    1. Spherical K-Means (our implementation on unit sphere with Haversine)
    2. Sklearn on 3D Cartesian coordinates
    3. Sklearn on unprojected degrees (Euclidean degrees)
    """
    # 1. Spherical K-Means
    spherical_model = SphericalKMeans(k=k, random_state=random_state).fit(coords)
    spherical_sil = compute_geographic_silhouette(coords, spherical_model.labels)

    # 2. Sklearn on 3D Cartesian
    cart = latlon_to_cartesian_unit(coords)
    sklearn_cart = SklearnKMeans(n_clusters=k, random_state=random_state, n_init=10).fit(cart)
    sklearn_cart_sil = compute_geographic_silhouette(coords, sklearn_cart.labels_)

    # 3. Sklearn on raw degrees
    sklearn_deg = SklearnKMeans(n_clusters=k, random_state=random_state, n_init=10).fit(coords)
    sklearn_deg_sil = compute_geographic_silhouette(coords, sklearn_deg.labels_)

    return {
        "k": k,
        "spherical_kmeans_from_scratch": {
            "inertia_sq_meters": round(spherical_model.inertia_, 2),
            "silhouette_score": round(spherical_sil, 4),
            "converged": spherical_model.converged_,
            "iterations": spherical_model.n_iter_,
        },
        "sklearn_cartesian_3d": {
            "inertia_cartesian": round(float(sklearn_cart.inertia_), 6),
            "silhouette_score": round(sklearn_cart_sil, 4),
            "iterations": int(sklearn_cart.n_iter_),
        },
        "sklearn_euclidean_degrees": {
            "inertia_deg_sq": round(float(sklearn_deg.inertia_), 6),
            "silhouette_score": round(sklearn_deg_sil, 4),
            "iterations": int(sklearn_deg.n_iter_),
            "warning": "Euclidean degrees distort spatial distances by failing to account for cos(latitude) longitudinal convergence.",
        },
    }

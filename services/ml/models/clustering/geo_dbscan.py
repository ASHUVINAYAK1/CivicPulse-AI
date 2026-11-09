"""
CivicPulse AI — Geographic DBSCAN with Haversine Distance
Implements Density-Based Spatial Clustering of Applications with Noise using
exact Haversine metric and provides explainable diagnostics for noise points.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
try:
    from .geo_kmeans import (
        haversine_distance,
        pairwise_haversine_matrix,
        compute_geographic_silhouette,
        validate_coordinates,
    )
except (ImportError, ValueError):
    from geo_kmeans import (
        haversine_distance,
        pairwise_haversine_matrix,
        compute_geographic_silhouette,
        validate_coordinates,
    )


class GeographicDBSCAN:
    """
    Density-Based Spatial Clustering with Noise for WGS84 coordinates.
    Operates in meter units via Haversine distance.
    """

    def __init__(self, eps_meters: float = 500.0, min_samples: int = 3):
        if eps_meters <= 0:
            raise ValueError(f"eps_meters must be strictly positive, got {eps_meters}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {min_samples}")

        self.eps_meters = float(eps_meters)
        self.min_samples = int(min_samples)

        self.labels: Optional[np.ndarray] = None
        self.point_types: Optional[List[str]] = None  # "core", "border", "noise"
        self.diagnostics: Optional[List[Dict[str, Any]]] = None
        self.n_clusters_: int = 0
        self.n_noise_: int = 0

    def fit(self, coords: np.ndarray, incident_ids: Optional[List[str]] = None) -> "GeographicDBSCAN":
        """
        Fits DBSCAN on Nx2 [latitude, longitude] array.
        """
        coords = np.asarray(coords, dtype=np.float64)
        n = len(coords)

        if incident_ids is None:
            incident_ids = [f"INC-{i}" for i in range(n)]

        if n == 0:
            self.labels = np.array([], dtype=int)
            self.point_types = []
            self.diagnostics = []
            self.n_clusters_ = 0
            self.n_noise_ = 0
            return self

        # 1. Compute pairwise Haversine distance matrix
        dist_matrix = pairwise_haversine_matrix(coords)

        # 2. Find epsilon-neighborhood for each point
        neighborhoods = []
        is_core = np.zeros(n, dtype=bool)

        for i in range(n):
            neighbors = np.where(dist_matrix[i] <= self.eps_meters)[0]
            neighborhoods.append(neighbors)
            if len(neighbors) >= self.min_samples:
                is_core[i] = True

        # 3. Cluster expansion
        labels = np.full(n, -1, dtype=int)  # -1 represents unassigned / noise
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            if not is_core[i]:
                continue

            # Start new cluster from this core point
            labels[i] = cluster_id
            seeds = list(neighborhoods[i])

            seed_idx = 0
            while seed_idx < len(seeds):
                current_pt = seeds[seed_idx]
                seed_idx += 1

                if not visited[current_pt]:
                    visited[current_pt] = True
                    if is_core[current_pt]:
                        # Add new neighbors to seeds if not already present
                        for neighbor in neighborhoods[current_pt]:
                            if neighbor not in seeds:
                                seeds.append(neighbor)

                if labels[current_pt] == -1:
                    labels[current_pt] = cluster_id

            cluster_id += 1

        # 4. Classify point types & generate diagnostic rationale
        point_types = []
        diagnostics = []

        for i in range(n):
            num_neighbors = len(neighborhoods[i])
            # Nearest neighbor distance (excluding self)
            if n > 1:
                other_dists = np.delete(dist_matrix[i], i)
                nearest_dist = float(np.min(other_dists))
            else:
                nearest_dist = float("inf")

            if is_core[i]:
                pt_type = "core"
                reason = (
                    f"Core incident: Has {num_neighbors} incidents within {self.eps_meters:.0f}m "
                    f"(meets min_samples={self.min_samples} threshold)."
                )
            elif labels[i] >= 0:
                pt_type = "border"
                reason = (
                    f"Border incident: Has {num_neighbors} incidents within {self.eps_meters:.0f}m "
                    f"(< min_samples={self.min_samples}), but is within reach of a core incident."
                )
            else:
                pt_type = "noise"
                reason = (
                    f"Noise / isolated incident: Only {num_neighbors} incidents within {self.eps_meters:.0f}m "
                    f"(requires min_samples={self.min_samples}). Nearest incident is {nearest_dist:.1f}m away."
                )

            point_types.append(pt_type)
            diagnostics.append({
                "incident_id": incident_ids[i],
                "latitude": round(float(coords[i, 0]), 6),
                "longitude": round(float(coords[i, 1]), 6),
                "cluster_label": int(labels[i]),
                "point_type": pt_type,
                "neighbors_within_eps": num_neighbors,
                "nearest_neighbor_distance_meters": round(nearest_dist, 2),
                "classification_reason": reason,
            })

        self.labels = labels
        self.point_types = point_types
        self.diagnostics = diagnostics
        self.n_clusters_ = cluster_id
        self.n_noise_ = int(np.sum(labels == -1))
        return self

    def get_cluster_summaries(self, coords: np.ndarray) -> List[Dict[str, Any]]:
        """Returns statistical metrics per cluster and noise summary."""
        summaries = []
        for c_id in range(self.n_clusters_):
            mask = (self.labels == c_id)
            members = coords[mask]
            count = int(np.sum(mask))

            # Centroid as mean coordinates
            c_lat = float(np.mean(members[:, 0]))
            c_lon = float(np.mean(members[:, 1]))

            dists = [
                haversine_distance(m[0], m[1], c_lat, c_lon)
                for m in members
            ]
            avg_dist = float(np.mean(dists)) if dists else 0.0
            max_dist = float(np.max(dists)) if dists else 0.0

            core_count = sum(1 for idx in np.where(mask)[0] if self.point_types[idx] == "core")
            border_count = count - core_count

            summaries.append({
                "cluster_index": c_id,
                "centroid_latitude": round(c_lat, 6),
                "centroid_longitude": round(c_lon, 6),
                "member_count": count,
                "core_points": core_count,
                "border_points": border_count,
                "average_distance_meters": round(avg_dist, 2),
                "max_distance_meters": round(max_dist, 2),
            })
        return summaries

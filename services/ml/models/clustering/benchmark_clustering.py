"""
CivicPulse AI — Geographical Hotspot Benchmark Suite
Executes empirical evaluations of Spherical K-Means and Geographic DBSCAN
on the 66 real incident coordinates from PostgreSQL.
"""

import json
from pathlib import Path
import numpy as np

from geo_kmeans import (
    SphericalKMeans,
    compute_geographic_silhouette,
    compare_with_sklearn,
    haversine_distance,
)
from geo_dbscan import GeographicDBSCAN


def run_benchmark():
    artifact_dir = Path(__file__).resolve().parent.parent.parent / "model_artifacts"
    coords_file = artifact_dir / "incident_coordinates.json"

    if not coords_file.exists():
        raise FileNotFoundError(f"Incident coordinates not found at {coords_file}")

    with open(coords_file, "r") as f:
        incidents = json.load(f)

    coords = np.array([[inc["latitude"], inc["longitude"]] for inc in incidents], dtype=np.float64)
    incident_ids = [inc["id"] for inc in incidents]
    n_points = len(coords)

    print(f"Loaded {n_points} real incident coordinates for spatial evaluation.")

    # 1. K-Means Parameter Sweep across K = 2..8
    kmeans_results = []
    for k in range(2, 9):
        model = SphericalKMeans(k=k, random_state=42).fit(coords)
        sil = compute_geographic_silhouette(coords, model.labels)
        summaries = model.get_cluster_summaries(coords)

        # Seed stability analysis (evaluate 10 seeds)
        seed_inertias = []
        for seed in [1, 7, 13, 21, 42, 77, 99, 123, 256, 512]:
            m_seed = SphericalKMeans(k=k, random_state=seed).fit(coords)
            seed_inertias.append(m_seed.inertia_)

        inertia_std = float(np.std(seed_inertias))
        inertia_mean = float(np.mean(seed_inertias))
        stability_cv = (inertia_std / inertia_mean) if inertia_mean > 0 else 0.0

        kmeans_results.append({
            "k": k,
            "inertia_sq_meters": round(model.inertia_, 2),
            "silhouette_score": round(sil, 4),
            "iterations": model.n_iter_,
            "converged": model.converged_,
            "seed_stability": {
                "inertia_mean": round(inertia_mean, 2),
                "inertia_std": round(inertia_std, 2),
                "coefficient_of_variation": round(stability_cv, 4),
            },
            "clusters": summaries,
        })
        print(f"K-Means (K={k}): Inertia={model.inertia_:,.0f} m^2, Silhouette={sil:.4f}, Stability CV={stability_cv:.4f}")

    # 2. Scikit-learn Baseline Comparison at K=4
    sklearn_comp = compare_with_sklearn(coords, k=4, random_state=42)
    print("\n--- Scikit-Learn Comparison (K=4) ---")
    print(json.dumps(sklearn_comp, indent=2))

    # 3. DBSCAN Parameter Sweep across eps (meters) and min_samples
    dbscan_grid = []
    eps_range = [250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0]
    min_samples_range = [2, 3, 5]

    for eps in eps_range:
        for ms in min_samples_range:
            db_model = GeographicDBSCAN(eps_meters=eps, min_samples=ms).fit(coords, incident_ids)
            sil = compute_geographic_silhouette(coords, db_model.labels)
            summaries = db_model.get_cluster_summaries(coords)

            dbscan_grid.append({
                "eps_meters": eps,
                "min_samples": ms,
                "n_clusters": db_model.n_clusters_,
                "n_noise": db_model.n_noise_,
                "noise_ratio": round(db_model.n_noise_ / n_points, 4),
                "silhouette_score": round(sil, 4),
                "clusters": summaries,
            })
            print(f"DBSCAN (eps={eps:4.0f}m, min_samples={ms}): Clusters={db_model.n_clusters_:2d}, Noise={db_model.n_noise_:2d} ({db_model.n_noise_/n_points*100:4.1f}%), Silhouette={sil:.4f}")

    # Find recommended parameters
    best_k = max(kmeans_results, key=lambda x: x["silhouette_score"])
    valid_dbscan = [g for g in dbscan_grid if g["n_clusters"] >= 2 and g["noise_ratio"] <= 0.40]
    best_dbscan = max(valid_dbscan, key=lambda x: x["silhouette_score"]) if valid_dbscan else dbscan_grid[0]

    output_payload = {
        "dataset_name": "Delhi_NCR_66_Real_Incidents",
        "total_incidents": n_points,
        "is_exploratory": True,
        "disclaimer": "Clusters describe the spatial distribution of incidents currently available in the system. They are not validated predictions of future road damage.",
        "kmeans_evaluation": {
            "tested_k_range": [2, 8],
            "recommended_k": best_k["k"],
            "recommended_k_reason": f"K={best_k['k']} maximizes geographic silhouette score ({best_k['silhouette_score']}) while achieving balanced cluster compactness.",
            "results": kmeans_results,
        },
        "sklearn_comparison": sklearn_comp,
        "dbscan_evaluation": {
            "tested_eps_range_meters": eps_range,
            "tested_min_samples_range": min_samples_range,
            "recommended_eps_meters": best_dbscan["eps_meters"],
            "recommended_min_samples": best_dbscan["min_samples"],
            "recommended_dbscan_reason": (
                f"eps={best_dbscan['eps_meters']:.0f}m with min_samples={best_dbscan['min_samples']} yields {best_dbscan['n_clusters']} dense clusters "
                f"with {best_dbscan['n_noise']} noise points ({best_dbscan['noise_ratio']*100:.1f}%) and silhouette score of {best_dbscan['silhouette_score']}."
            ),
            "results": dbscan_grid,
        },
    }

    out_path = artifact_dir / "hotspot_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved comprehensive benchmark results to {out_path}")
    return output_payload


if __name__ == "__main__":
    run_benchmark()


import json
from pathlib import Path
import numpy as np

coords_file = Path("services/ml/model_artifacts/incident_coordinates.json")
with open(coords_file, "r") as f:
    incidents = json.load(f)

coords = np.array([[inc["latitude"], inc["longitude"]] for inc in incidents], dtype=np.float64)
n = len(coords)

print(f"Total incidents: {n}")
bangalore = [inc for inc in incidents if inc["latitude"] < 20.0]
delhi = [inc for inc in incidents if inc["latitude"] >= 20.0]

print(f"Bangalore incidents (lat ~ 12.9°): {len(bangalore)}")
print(f"Delhi incidents (lat ~ 28.6°): {len(delhi)}")

print("\n--- Bangalore incident sample ---")
for b in bangalore[:5]:
    print(f"  {b['incidentNumber']} ({b['id']}): ({b['latitude']}, {b['longitude']}) - {b['title']} - {b['address']}")

print("\n--- Delhi incident sample ---")
for d in delhi[:5]:
    print(f"  {d['incidentNumber']} ({d['id']}): ({d['latitude']}, {d['longitude']}) - {d['title']} - {d['address']}")

# Let's import geo_kmeans and analyze K-Means from K=2 to K=8
import sys
sys.path.insert(0, "services/ml/models/clustering")
from geo_kmeans import SphericalKMeans, haversine_distance, compute_geographic_silhouette, pairwise_haversine_matrix
from geo_dbscan import GeographicDBSCAN

dist_matrix = pairwise_haversine_matrix(coords)
upper_dists = dist_matrix[np.triu_indices(n, k=1)]

print("\n--- Geographic Distance Distribution (meters) ---")
print(f"  Min: {upper_dists.min():.1f}m")
print(f"  25th percentile: {np.percentile(upper_dists, 25):.1f}m")
print(f"  Median: {np.median(upper_dists):.1f}m")
print(f"  75th percentile: {np.percentile(upper_dists, 75):.1f}m")
print(f"  Mean: {np.mean(upper_dists):.1f}m")
print(f"  90th percentile: {np.percentile(upper_dists, 90):.1f}m")
print(f"  Max: {upper_dists.max():.1f}m")

print("\n--- Detailed K-Means Evaluation (K=2..8) ---")
for k in range(2, 9):
    model = SphericalKMeans(k=k, random_state=42).fit(coords)
    sil = compute_geographic_silhouette(coords, model.labels)
    print(f"\n==================== K = {k} ====================")
    print(f"Inertia: {model.inertia_:,.2f} m^2, Silhouette: {sil:.4f}, Converged: {model.converged_}, Iterations: {model.n_iter_}")
    
    for k_idx in range(k):
        members_mask = (model.labels == k_idx)
        member_indices = np.where(members_mask)[0]
        c_lat, c_lon = model.centroids_latlon[k_idx]
        
        member_dists = [haversine_distance(coords[i, 0], coords[i, 1], c_lat, c_lon) for i in member_indices]
        mean_dist = np.mean(member_dists) if member_dists else 0
        median_dist = np.median(member_dists) if member_dists else 0
        max_dist = np.max(member_dists) if member_dists else 0
        
        member_ids = [incidents[i]["incidentNumber"] for i in member_indices]
        print(f"  Cluster {k_idx}: size={len(member_indices)}, centroid=({c_lat:.4f}, {c_lon:.4f})")
        print(f"    Distance to centroid: mean={mean_dist:.1f}m, median={median_dist:.1f}m, max={max_dist:.1f}m")
        print(f"    Member IDs ({len(member_ids)}): {member_ids[:10]}{'...' if len(member_ids) > 10 else ''}")

# Seed stability across 10 seeds
print("\n--- Seed Stability (K=2..8) across 10 deterministic seeds ---")
seeds = [1, 7, 13, 21, 42, 77, 99, 123, 256, 512]
for k in range(2, 9):
    seed_inertias = []
    seed_assignments = []
    for s in seeds:
        m = SphericalKMeans(k=k, random_state=s).fit(coords)
        seed_inertias.append(m.inertia_)
        seed_assignments.append(m.labels)
    
    # Check assignment stability
    ref_labels = seed_assignments[0]
    identical_count = 0
    from sklearn.metrics import adjusted_rand_score
    ari_scores = [adjusted_rand_score(ref_labels, l) for l in seed_assignments[1:]]
    mean_ari = np.mean(ari_scores)
    
    cv = np.std(seed_inertias) / np.mean(seed_inertias) if np.mean(seed_inertias) > 0 else 0
    print(f"K={k}: Inertia Mean={np.mean(seed_inertias):,.1f}, Std={np.std(seed_inertias):,.1f}, CV={cv:.4f}, Mean ARI vs Seed 1={mean_ari:.4f}")

# Audit DBSCAN silhouette = 1.0000
print("\n--- DBSCAN Audit ---")
for eps, ms in [(500.0, 3), (1000.0, 3), (250.0, 3)]:
    db = GeographicDBSCAN(eps_meters=eps, min_samples=ms).fit(coords)
    sil = compute_geographic_silhouette(coords, db.labels)
    print(f"\nDBSCAN eps={eps}m, min_samples={ms}:")
    print(f"  n_clusters: {db.n_clusters_}, n_noise: {db.n_noise_} ({db.n_noise_/n*100:.1f}%)")
    for c_id in range(db.n_clusters_):
        c_mask = (db.labels == c_id)
        c_coords = coords[c_mask]
        print(f"  Cluster {c_id}: size={np.sum(c_mask)}, lat_range=[{c_coords[:,0].min():.4f}, {c_coords[:,0].max():.4f}]")
    
    # Check why silhouette was calculated as 1.0
    # Let's inspect intra-cluster distances vs inter-cluster distances
    labels = db.labels
    unique_clusters = [l for l in np.unique(labels) if l >= 0]
    print(f"  Unique non-noise clusters: {unique_clusters}")
    if len(unique_clusters) >= 2:
        c0_coords = coords[labels == 0]
        c1_coords = coords[labels == 1]
        
        # Intra cluster distances
        c0_dists = [haversine_distance(c0_coords[i,0], c0_coords[i,1], c0_coords[j,0], c0_coords[j,1])
                    for i in range(len(c0_coords)) for j in range(i+1, len(c0_coords))]
        c1_dists = [haversine_distance(c1_coords[i,0], c1_coords[i,1], c1_coords[j,0], c1_coords[j,1])
                    for i in range(len(c1_coords)) for j in range(i+1, len(c1_coords))]
        
        # Inter cluster distances
        inter_dists = [haversine_distance(c0_coords[i,0], c0_coords[i,1], c1_coords[j,0], c1_coords[j,1])
                       for i in range(len(c0_coords)) for j in range(len(c1_coords))]
        
        print(f"  Cluster 0 max intra-distance: {max(c0_dists):.1f}m, mean intra: {np.mean(c0_dists):.1f}m")
        print(f"  Cluster 1 max intra-distance: {max(c1_dists):.1f}m, mean intra: {np.mean(c1_dists):.1f}m")
        print(f"  Inter-cluster distance: min={min(inter_dists):.1f}m, mean={np.mean(inter_dists):.1f}m")
        print(f"  Ratio inter/intra: {min(inter_dists)/max(c0_dists):.1f}x")
        print(f"  Silhouette score on clustered points: {sil:.6f}")


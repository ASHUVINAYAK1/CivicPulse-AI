# CivicPulse AI — Stage 5 Evaluation Audit Addendum

**Audit Date:** November 7, 2026  
**Subject:** Mathematical Audit of Spatial Clustering Metrics on the 66-Incident Dataset  
**Auditor:** Senior ML Engineer & Data Scientist  

---

## 1. Executive Summary & Root Cause Analysis

A thorough empirical investigation into the Stage 5 spatial clustering evaluation on the 66 incident records in PostgreSQL revealed fundamental dataset characteristics that explain:
1. The dramatic inertia drop between $K=3$ and $K=4$.
2. The DBSCAN silhouette score of $1.0000$.
3. The exact mathematical differences between Spherical K-Means and Scikit-Learn.

---

## 2. Dataset Geographic Composition

The 66 incidents in PostgreSQL do not reside within a single city corridor; they span two distinct metropolitan regions separated by approximately **1,740 kilometers**:

| Geographic Region | Latitude Range | Longitude Range | Incident Count | Spatial Characteristics |
|---|---|---|---|---|
| **Bangalore Urban** | $12.9260^\circ\text{N} - 12.9716^\circ\text{N}$ | $77.5946^\circ\text{E} - 77.6762^\circ\text{E}$ | **58 incidents** | **57 incidents** are tightly co-located on the Marathahalli Outer Ring Road corridor within a 4-meter radius (mean distance = $0.3\text{m}$). Exactly **1 incident** (`INC-357677-895`) is located in south-east Bangalore, **$10,017\text{ meters}$** away. |
| **Delhi NCR** | $28.6139^\circ\text{N} - 28.6412^\circ\text{N}$ | $77.2045^\circ\text{E} - 77.2250^\circ\text{E}$ | **8 incidents** | 5 incidents in Central Delhi / Connaught Place corridor; 3 incidents in North Delhi / University Road corridor. |

### Pairwise Distance Distribution (66 Incidents, 2,145 Pairs)
- **Minimum:** $0.0\text{ m}$ (co-located duplicate burst reports)
- **25th Percentile:** $0.0\text{ m}$
- **Median:** $0.0\text{ m}$
- **75th Percentile:** $1,434.2\text{ m}$
- **Mean:** $376,868.5\text{ m}$ ($376.9\text{ km}$)
- **90th Percentile:** $1,740,474.2\text{ m}$ ($1,740.5\text{ km}$)
- **Maximum:** $1,748,089.7\text{ m}$ ($1,748.1\text{ km}$)

---

## 3. Investigation: K-Means Inertia Drop ($K=3 \rightarrow K=4$)

### Cluster Structure & Distances ($K=2 \dots 8$)

```
K=2: Inertia = 115,855,545.65 m² | Silhouette = 0.9997
  - Cluster 0 (Size=58, Bangalore): Centroid=(12.9708, 77.5960) | Mean=345.4m, Max=10,017.7m
  - Cluster 1 (Size=8, Delhi):       Centroid=(28.6226, 77.2126) | Mean=1219.5m, Max=2,185.0m

K=3: Inertia = 103,987,737.31 m² | Silhouette = 0.9676
  - Cluster 0 (Size=58, Bangalore): Centroid=(12.9708, 77.5960) | Mean=345.4m, Max=10,017.7m
  - Cluster 1 (Size=3, North Delhi): Centroid=(28.6351, 77.2201) | Mean=660.6m, Max=829.3m
  - Cluster 2 (Size=5, Central Delhi): Centroid=(28.6151, 77.2081) | Mean=253.3m, Max=633.2m

K=4: Inertia = 1,872,294.39 m² | Silhouette = 0.9526
  - Cluster 0 (Size=57, Bangalore ORR): Centroid=(12.9716, 77.5946) | Mean=0.3m, Max=3.9m
  - Cluster 1 (Size=3, North Delhi):     Centroid=(28.6351, 77.2201) | Mean=660.6m, Max=829.3m
  - Cluster 2 (Size=5, Central Delhi):   Centroid=(28.6151, 77.2081) | Mean=253.3m, Max=633.2m
  - Cluster 3 (Size=1, Bangalore Outlier INC-357677-895): Centroid=(12.9260, 77.6762) | Mean=0.0m, Max=0.0m
```

### Root Cause of the $98.2\%$ Inertia Drop
1. At $K=2$ and $K=3$, all 58 Bangalore incidents are forced into a single cluster (Cluster 0).
2. Within Cluster 0, 57 incidents sit at the Marathahalli corridor, but incident `INC-357677-895` sits $10,017.7\text{ meters}$ away.
3. Because K-Means minimizes the sum of **squared** distances:
   $$\text{Inertia contribution of } \text{INC-357677-895} = (10,017.7\text{ m})^2 \approx 100,354,000\text{ m}^2$$
   This single incident accounted for **$96.5\%$ of the total system inertia** at $K=3$ ($103,987,737\text{ m}^2$).
4. At $K=4$, the algorithm allocates the 4th centroid to `INC-357677-895` as a singleton cluster. Its distance to its centroid becomes $0.0\text{ m}$, instantly eliminating the $\sim 100\text{M m}^2$ penalty.
5. The remaining 57 Bangalore incidents collapse to an intra-cluster distance of only $0.3\text{ m}$ (total inertia $\approx 5\text{ m}^2$). The entire remaining inertia at $K=4$ ($1,872,294\text{ m}^2$) belongs solely to the intra-Delhi groupings.

### Seed Stability Audit (10 Deterministic Seeds)
Across seeds `[1, 7, 13, 21, 42, 77, 99, 123, 256, 512]`:
- **$K=2$:** $\text{Mean} = 115.85\text{M}$, $\text{CV} = 0.0000$, $\text{ARI} = 1.0000$.
- **$K=3$:** $\text{Mean} = 40.81\text{M}$, $\text{CV} = 1.0133$, $\text{ARI} = 0.9270$. At $K=3$, seeds vary between splitting Delhi vs isolating the Bangalore outlier.
- **$K=4$:** $\text{Mean} = 1.872\text{M}$, $\text{CV} = 0.0000$, $\text{ARI} = 1.0000$. Complete convergence to the same partition across all 10 seeds.

---

## 4. Investigation: DBSCAN Silhouette Score of 1.0000

At parameters $eps=500\text{m}$, $min\_samples=3$:
- **Non-noise clusters formed:** 2 (Cluster 0: 4 incidents in Delhi; Cluster 1: 57 incidents in Bangalore).
- **Excluded noise points:** 5 incidents (`INC-2024-002`, `003`, `004`, `005` in Delhi, and `INC-357677-895` in Bangalore).
- **Clustered sample set evaluated for silhouette:** 61 points (4 in Delhi, 57 in Bangalore).

### Distance Matrix Analysis on the 61 Clustered Points
- **Cluster 0 (Delhi, $N=4$):**
  - Mean intra-cluster distance: $\bar{a}_0 = 2.8\text{ meters}$
  - Maximum intra-cluster distance: $4.9\text{ meters}$
- **Cluster 1 (Bangalore, $N=57$):**
  - Mean intra-cluster distance: $\bar{a}_1 = 0.3\text{ meters}$
  - Maximum intra-cluster distance: $4.0\text{ meters}$
- **Inter-Cluster Distance between Delhi and Bangalore:**
  - Minimum: $1,739,802.1\text{ meters}$ ($1,739.8\text{ km}$)
  - Mean: $1,739,804.8\text{ meters}$

### Mathematical Derivation of the Silhouette Value
For any incident $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} = \frac{1,739,804.8 - 2.8}{1,739,804.8} = 0.99999839$$
When rounded to four decimal places, the score displays as **`1.0000`**.

> [!WARNING]
> **Governance Finding:**  
> The silhouette score of $1.0000$ is mathematically accurate according to the standard formula applied to the clustered subset, but it is an artifact of computing silhouette across two ultra-dense clusters located in two different states ($1,740\text{ km}$ apart) while treating scattered urban incidents as noise. In a single-city municipal deployment, the silhouette score will reflect realistic local corridor separation ($0.40 - 0.75$).

---

## 5. Architectural Comparison: Spherical K-Means vs. Scikit-Learn

| Evaluation Dimension | CivicPulse Spherical K-Means (Scratch) | Scikit-Learn `KMeans` (3D Cartesian) | Scikit-Learn `KMeans` (Raw Degrees) |
|---|---|---|---|
| **Coordinate Space** | 3D Unit Sphere $\mathbb{S}^2 \subset \mathbb{R}^3$ | Euclidean Space $\mathbb{R}^3$ | Planar Euclidean $\mathbb{R}^2$ (Degrees) |
| **Assignment Metric** | Exact Great-Circle Arc Length (Haversine, in meters) | Chord Euclidean Distance $\|\mathbf{u} - \mathbf{v}\|_2$ | Degree Euclidean $\sqrt{(\Delta\text{lat})^2 + (\Delta\text{lon})^2}$ |
| **Centroid Computation** | Normalized vector mean: $\hat{\mathbf{v}} = \frac{\sum \mathbf{v}_i}{\|\sum \mathbf{v}_i\|}$, mapped to $(\text{lat}, \text{lon})$ | Unnormalized arithmetic mean in $\mathbb{R}^3$ (sits inside unit ball: $\|\bar{\mathbf{v}}\| < 1$) | Arithmetic mean of degrees: $(\frac{1}{N}\sum \text{lat}, \frac{1}{N}\sum \text{lon})$ |
| **Optimization Objective** | Minimizes $\sum d_{\text{Haversine}}(p, \mu)^2$ $[\text{meters}^2]$ | Minimizes $\sum \|\mathbf{x}_i - \mathbf{c}_k\|^2$ [Chord units$^2$] | Minimizes $\sum [(\Delta\text{lat})^2 + (\Delta\text{lon})^2]$ [Deg$^2$] |
| **Geographic Validity** | **Globally valid:** Spheroid curvature preserved across all latitudes and longitudes. | **Locally valid:** Chord distance is monotonic with angular arc length ($c = 2\sin(\theta/2)$). | **Invalid globally:** Fails to scale longitude by $\cos(\text{lat})$. At lat $28.6^\circ$, $1^\circ \text{lon} \approx 97.8\text{km}$ vs $1^\circ \text{lat} \approx 110.8\text{km}$. |

### Conclusion on Objectives
Spherical K-Means and Scikit-Learn do **not** optimize identical objective functions:
- Spherical K-Means minimizes geodesic surface distance squared in SI meters.
- Scikit-Learn 3D minimizes Euclidean secant (tunnel) chord distance squared.
While both yield identical partition assignments on local clusters because chord length is monotonic with spherical arc length for $\theta \in [0, \pi]$, the objective values and gradient steps are mathematically distinct.


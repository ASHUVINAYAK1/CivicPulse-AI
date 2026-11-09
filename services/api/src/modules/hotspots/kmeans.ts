/**
 * CivicPulse AI - Spherical K-Means from Scratch (TypeScript)
 * Implements spherical K-Means clustering on unit-sphere WGS84 coordinates.
 * Features deterministic seeding, spherical k-means++ initialization,
 * empty cluster recovery, and geographic silhouette evaluation.
 */

import {
  haversineDistance,
  latLonToCartesian,
  cartesianToLatLon,
  computeGeographicSilhouetteScore,
  GeoPoint,
  CartesianPoint3D,
} from './geo-utils';
import { ValidatedSpatialIncident } from './preprocessor';

/**
 * Deterministic 32-bit pseudo-random number generator (Mulberry32).
 */
export function createSeededRng(seed: number): () => number {
  let s = Math.floor(seed) >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface KMeansClusterSummary {
  clusterIndex: number;
  centroidLatitude: number;
  centroidLongitude: number;
  memberCount: number;
  averageDistanceMeters: number;
  maxDistanceMeters: number;
  damageTypeBreakdown: Record<string, number>;
  severityBreakdown: Record<string, number>;
  verifiedCount: number;
  memberIncidentIds: string[];
}

export interface KMeansResult {
  algorithm: string;
  k: number;
  maxIterations: number;
  toleranceMeters: number;
  randomSeed: number;
  converged: boolean;
  iterations: number;
  inertiaSqMeters: number;
  silhouetteScore: number;
  clusters: KMeansClusterSummary[];
  assignments: {
    incidentId: string;
    clusterIndex: number;
    distanceToCentroidMeters: number;
  }[];
}

export interface KMeansRangeEvaluationItem {
  k: number;
  inertiaSqMeters: number;
  silhouetteScore: number;
  iterations: number;
  converged: boolean;
}

export class SphericalKMeansEngine {
  private k: number;
  private maxIter: number;
  private tolMeters: number;
  private randomSeed: number;

  constructor(
    k: number = 3,
    maxIter: number = 100,
    tolMeters: number = 1.0,
    randomSeed: number = 42
  ) {
    this.k = Math.max(1, Math.floor(k));
    this.maxIter = Math.max(1, Math.floor(maxIter));
    this.tolMeters = Math.max(0.01, tolMeters);
    this.randomSeed = randomSeed;
  }

  public fit(incidents: ValidatedSpatialIncident[]): KMeansResult {
    const n = incidents.length;
    if (n < this.k) {
      throw new Error(
        `Number of incidents (${n}) is smaller than requested clusters K=${this.k}.`
      );
    }

    const rng = createSeededRng(this.randomSeed);
    const coords: GeoPoint[] = incidents.map((i) => ({
      latitude: i.latitude,
      longitude: i.longitude,
    }));
    const cartPoints: CartesianPoint3D[] = coords.map((c) =>
      latLonToCartesian(c.latitude, c.longitude)
    );

    // 1. Spherical K-Means++ Initialization
    const initialIndices: number[] = [];
    const firstIdx = Math.floor(rng() * n);
    initialIndices.push(firstIdx);

    while (initialIndices.length < this.k) {
      const distSq: number[] = [];
      let totalDistSq = 0;

      for (let i = 0; i < n; i++) {
        let minDist = Infinity;
        for (const cIdx of initialIndices) {
          const d = haversineDistance(
            coords[i].latitude,
            coords[i].longitude,
            coords[cIdx].latitude,
            coords[cIdx].longitude
          );
          if (d < minDist) minDist = d;
        }
        const dSq = minDist * minDist;
        distSq.push(dSq);
        totalDistSq += dSq;
      }

      if (totalDistSq > 0) {
        let target = rng() * totalDistSq;
        let chosen = 0;
        for (let i = 0; i < n; i++) {
          target -= distSq[i];
          if (target <= 0) {
            chosen = i;
            break;
          }
        }
        if (!initialIndices.includes(chosen)) {
          initialIndices.push(chosen);
        } else {
          // Fallback to random unselected index
          const unselected = Array.from({ length: n }, (_, i) => i).filter(
            (i) => !initialIndices.includes(i)
          );
          initialIndices.push(unselected[Math.floor(rng() * unselected.length)]);
        }
      } else {
        const unselected = Array.from({ length: n }, (_, i) => i).filter(
          (i) => !initialIndices.includes(i)
        );
        initialIndices.push(unselected[Math.floor(rng() * unselected.length)]);
      }
    }

    // Centroids as Cartesian unit vectors
    let centroidsCart: CartesianPoint3D[] = initialIndices.map(
      (idx) => cartPoints[idx]
    );
    let centroidsLatLon: GeoPoint[] = centroidsCart.map((c) => cartesianToLatLon(c));

    let assignments = new Array(n).fill(0);
    let distancesToCentroids = new Array(n).fill(0);
    let converged = false;
    let iteration = 0;

    // 2. Convergence Loop
    for (let it = 0; it < this.maxIter; it++) {
      iteration = it + 1;

      // Assignment step using exact Haversine distance
      for (let i = 0; i < n; i++) {
        let bestK = 0;
        let bestDist = Infinity;

        for (let kIdx = 0; kIdx < this.k; kIdx++) {
          const dist = haversineDistance(
            coords[i].latitude,
            coords[i].longitude,
            centroidsLatLon[kIdx].latitude,
            centroidsLatLon[kIdx].longitude
          );
          if (dist < bestDist) {
            bestDist = dist;
            bestK = kIdx;
          }
        }
        assignments[i] = bestK;
        distancesToCentroids[i] = bestDist;
      }

      // Centroid update step on unit sphere
      const newCentroidsCart: CartesianPoint3D[] = [];

      for (let kIdx = 0; kIdx < this.k; kIdx++) {
        const memberIndices: number[] = [];
        for (let i = 0; i < n; i++) {
          if (assignments[i] === kIdx) memberIndices.push(i);
        }

        if (memberIndices.length > 0) {
          let sumX = 0;
          let sumY = 0;
          let sumZ = 0;
          for (const mIdx of memberIndices) {
            sumX += cartPoints[mIdx].x;
            sumY += cartPoints[mIdx].y;
            sumZ += cartPoints[mIdx].z;
          }
          const meanX = sumX / memberIndices.length;
          const meanY = sumY / memberIndices.length;
          const meanZ = sumZ / memberIndices.length;
          const norm = Math.sqrt(meanX * meanX + meanY * meanY + meanZ * meanZ);

          if (norm > 1e-9) {
            newCentroidsCart.push({
              x: meanX / norm,
              y: meanY / norm,
              z: meanZ / norm,
            });
          } else {
            // Degenerate mean, fall back to random member
            newCentroidsCart.push(cartPoints[memberIndices[0]]);
          }
        } else {
          // Empty cluster handling: assign to point furthest from any centroid
          let maxDist = -1;
          let furthestIdx = 0;
          for (let i = 0; i < n; i++) {
            if (distancesToCentroids[i] > maxDist) {
              maxDist = distancesToCentroids[i];
              furthestIdx = i;
            }
          }
          newCentroidsCart.push({ ...cartPoints[furthestIdx] });
        }
      }

      const newCentroidsLatLon = newCentroidsCart.map((c) => cartesianToLatLon(c));

      // Check convergence via maximum centroid shift (meters)
      let maxShift = 0;
      for (let kIdx = 0; kIdx < this.k; kIdx++) {
        const shift = haversineDistance(
          centroidsLatLon[kIdx].latitude,
          centroidsLatLon[kIdx].longitude,
          newCentroidsLatLon[kIdx].latitude,
          newCentroidsLatLon[kIdx].longitude
        );
        if (shift > maxShift) maxShift = shift;
      }

      centroidsCart = newCentroidsCart;
      centroidsLatLon = newCentroidsLatLon;

      if (maxShift < this.tolMeters) {
        converged = true;
        break;
      }
    }

    // 3. Compute final inertia (sum of squared Haversine distances in meters^2)
    let totalInertia = 0;
    const finalAssignments = [];

    for (let i = 0; i < n; i++) {
      const cIdx = assignments[i];
      const dist = haversineDistance(
        coords[i].latitude,
        coords[i].longitude,
        centroidsLatLon[cIdx].latitude,
        centroidsLatLon[cIdx].longitude
      );
      totalInertia += dist * dist;
      finalAssignments.push({
        incidentId: incidents[i].id,
        clusterIndex: cIdx,
        distanceToCentroidMeters: Number(dist.toFixed(1)),
      });
    }

    // 4. Compute Silhouette Score
    const silhouetteScore = computeGeographicSilhouetteScore(coords, assignments);

    // 5. Build Cluster Summaries
    const clusters: KMeansClusterSummary[] = [];
    for (let kIdx = 0; kIdx < this.k; kIdx++) {
      const memberIncidents = incidents.filter((_, idx) => assignments[idx] === kIdx);
      const memberDistances = memberIncidents.map((inc) =>
        haversineDistance(
          inc.latitude,
          inc.longitude,
          centroidsLatLon[kIdx].latitude,
          centroidsLatLon[kIdx].longitude
        )
      );

      const damageBreakdown: Record<string, number> = {};
      const severityBreakdown: Record<string, number> = {};
      let verifiedCount = 0;

      for (const inc of memberIncidents) {
        damageBreakdown[inc.damageType] = (damageBreakdown[inc.damageType] || 0) + 1;
        const sev = inc.severity || 'unassigned';
        severityBreakdown[sev] = (severityBreakdown[sev] || 0) + 1;
        if (inc.status === 'verified' || inc.verifiedAt !== null) {
          verifiedCount++;
        }
      }

      const avgDist =
        memberDistances.length > 0
          ? memberDistances.reduce((a, b) => a + b, 0) / memberDistances.length
          : 0;
      const maxDist =
        memberDistances.length > 0 ? Math.max(...memberDistances) : 0;

      clusters.push({
        clusterIndex: kIdx,
        centroidLatitude: Number(centroidsLatLon[kIdx].latitude.toFixed(6)),
        centroidLongitude: Number(centroidsLatLon[kIdx].longitude.toFixed(6)),
        memberCount: memberIncidents.length,
        averageDistanceMeters: Number(avgDist.toFixed(1)),
        maxDistanceMeters: Number(maxDist.toFixed(1)),
        damageTypeBreakdown: damageBreakdown,
        severityBreakdown: severityBreakdown,
        verifiedCount,
        memberIncidentIds: memberIncidents.map((i) => i.id),
      });
    }

    return {
      algorithm: 'Spherical_KMeans_From_Scratch',
      k: this.k,
      maxIterations: this.maxIter,
      toleranceMeters: this.tolMeters,
      randomSeed: this.randomSeed,
      converged,
      iterations: iteration,
      inertiaSqMeters: Number(totalInertia.toFixed(2)),
      silhouetteScore: Number(silhouetteScore.toFixed(4)),
      clusters,
      assignments: finalAssignments,
    };
  }

  /**
   * Evaluates a range of K values [minK..maxK] to produce elbow and silhouette curves.
   */
  public static evaluateRange(
    incidents: ValidatedSpatialIncident[],
    minK: number = 2,
    maxK: number = 8,
    randomSeed: number = 42
  ): KMeansRangeEvaluationItem[] {
    const results: KMeansRangeEvaluationItem[] = [];
    const maxAvailableK = Math.min(maxK, incidents.length - 1);

    for (let currentK = minK; currentK <= maxAvailableK; currentK++) {
      const engine = new SphericalKMeansEngine(currentK, 50, 1.0, randomSeed);
      const res = engine.fit(incidents);
      results.push({
        k: currentK,
        inertiaSqMeters: res.inertiaSqMeters,
        silhouetteScore: res.silhouetteScore,
        iterations: res.iterations,
        converged: res.converged,
      });
    }

    return results;
  }
}


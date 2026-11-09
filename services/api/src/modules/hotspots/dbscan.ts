/**
 * CivicPulse AI - Geographic DBSCAN with Haversine Distance (TypeScript)
 * Implements density-based clustering using exact meter radius and provides
 * explainable diagnostic rationales for noise and border classifications.
 */

import {
  haversineDistance,
  computePairwiseHaversineMatrix,
  computeGeographicSilhouetteScore,
  GeoPoint,
} from './geo-utils';
import { ValidatedSpatialIncident } from './preprocessor';

export type PointClassificationType = 'core' | 'border' | 'noise';

export interface DBSCANPointDiagnostic {
  incidentId: string;
  incidentNumber: string;
  latitude: number;
  longitude: number;
  clusterLabel: number; // -1 for noise
  pointType: PointClassificationType;
  neighborsWithinEps: number;
  nearestNeighborDistanceMeters: number;
  classificationReason: string;
}

export interface DBSCANClusterSummary {
  clusterIndex: number;
  centroidLatitude: number;
  centroidLongitude: number;
  memberCount: number;
  corePoints: number;
  borderPoints: number;
  averageDistanceMeters: number;
  maxDistanceMeters: number;
  damageTypeBreakdown: Record<string, number>;
  severityBreakdown: Record<string, number>;
  verifiedCount: number;
  memberIncidentIds: string[];
}

export interface DBSCANResult {
  algorithm: string;
  epsMeters: number;
  minSamples: number;
  totalPoints: number;
  clusterCount: number;
  noiseCount: number;
  noiseRatio: number;
  silhouetteScore: number;
  clusters: DBSCANClusterSummary[];
  diagnostics: DBSCANPointDiagnostic[];
}

export class GeographicDBSCANEngine {
  private epsMeters: number;
  private minSamples: number;

  constructor(epsMeters: number = 500.0, minSamples: number = 3) {
    if (epsMeters <= 0) {
      throw new Error(`epsMeters must be strictly positive, got ${epsMeters}`);
    }
    if (minSamples < 1) {
      throw new Error(`minSamples must be at least 1, got ${minSamples}`);
    }
    this.epsMeters = epsMeters;
    this.minSamples = Math.floor(minSamples);
  }

  public fit(incidents: ValidatedSpatialIncident[]): DBSCANResult {
    const n = incidents.length;
    if (n === 0) {
      return {
        algorithm: 'Geographic_DBSCAN_Haversine',
        epsMeters: this.epsMeters,
        minSamples: this.minSamples,
        totalPoints: 0,
        clusterCount: 0,
        noiseCount: 0,
        noiseRatio: 0.0,
        silhouetteScore: 0.0,
        clusters: [],
        diagnostics: [],
      };
    }

    const points: GeoPoint[] = incidents.map((i) => ({
      latitude: i.latitude,
      longitude: i.longitude,
    }));

    // 1. Compute NxN pairwise Haversine distance matrix in meters
    const distMatrix = computePairwiseHaversineMatrix(points);

    // 2. Identify epsilon-neighborhoods and core points
    const neighborhoods: number[][] = [];
    const isCore: boolean[] = new Array(n).fill(false);

    for (let i = 0; i < n; i++) {
      const neighbors: number[] = [];
      for (let j = 0; j < n; j++) {
        if (distMatrix[i][j] <= this.epsMeters) {
          neighbors.push(j);
        }
      }
      neighborhoods.push(neighbors);
      if (neighbors.length >= this.minSamples) {
        isCore[i] = true;
      }
    }

    // 3. Density Cluster Expansion
    const labels: number[] = new Array(n).fill(-1); // -1 represents noise
    const visited: boolean[] = new Array(n).fill(false);
    let clusterId = 0;

    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      visited[i] = true;

      if (!isCore[i]) continue; // Skip non-core seed points

      labels[i] = clusterId;
      const queue: number[] = [...neighborhoods[i]];

      let head = 0;
      while (head < queue.length) {
        const pt = queue[head++];

        if (!visited[pt]) {
          visited[pt] = true;
          if (isCore[pt]) {
            for (const neighbor of neighborhoods[pt]) {
              if (!queue.includes(neighbor)) {
                queue.push(neighbor);
              }
            }
          }
        }

        if (labels[pt] === -1) {
          labels[pt] = clusterId;
        }
      }

      clusterId++;
    }

    // 4. Classify point types and formulate explainable diagnostic reasons
    const diagnostics: DBSCANPointDiagnostic[] = [];

    for (let i = 0; i < n; i++) {
      const neighborCount = neighborhoods[i].length;

      // Find distance to closest other incident
      let nearestDist = Infinity;
      for (let j = 0; j < n; j++) {
        if (i !== j && distMatrix[i][j] < nearestDist) {
          nearestDist = distMatrix[i][j];
        }
      }

      let pointType: PointClassificationType;
      let reason: string;

      if (isCore[i]) {
        pointType = 'core';
        reason = `Core incident: Has ${neighborCount} incidents within ${this.epsMeters.toFixed(
          0
        )}m (meets min_samples=${this.minSamples} threshold).`;
      } else if (labels[i] >= 0) {
        pointType = 'border';
        reason = `Border incident: Has ${neighborCount} incidents within ${this.epsMeters.toFixed(
          0
        )}m (< min_samples=${this.minSamples}), but is within reach of a core incident.`;
      } else {
        pointType = 'noise';
        reason = `Noise / isolated incident: Only ${neighborCount} incidents within ${this.epsMeters.toFixed(
          0
        )}m (requires min_samples=${this.minSamples}). Nearest incident is ${nearestDist.toFixed(
          1
        )}m away.`;
      }

      diagnostics.push({
        incidentId: incidents[i].id,
        incidentNumber: incidents[i].incidentNumber,
        latitude: incidents[i].latitude,
        longitude: incidents[i].longitude,
        clusterLabel: labels[i],
        pointType,
        neighborsWithinEps: neighborCount,
        nearestNeighborDistanceMeters: nearestDist === Infinity ? 0 : Number(nearestDist.toFixed(1)),
        classificationReason: reason,
      });
    }

    // 5. Cluster Summaries
    const clusters: DBSCANClusterSummary[] = [];

    for (let c = 0; c < clusterId; c++) {
      const memberIndices: number[] = [];
      for (let i = 0; i < n; i++) {
        if (labels[i] === c) memberIndices.push(i);
      }

      const memberIncidents = memberIndices.map((idx) => incidents[idx]);
      const meanLat =
        memberIncidents.reduce((sum, inc) => sum + inc.latitude, 0) /
        memberIncidents.length;
      const meanLon =
        memberIncidents.reduce((sum, inc) => sum + inc.longitude, 0) /
        memberIncidents.length;

      const memberDistances = memberIncidents.map((inc) =>
        haversineDistance(inc.latitude, inc.longitude, meanLat, meanLon)
      );

      const coreCount = memberIndices.filter((idx) => isCore[idx]).length;
      const borderCount = memberIncidents.length - coreCount;

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
        clusterIndex: c,
        centroidLatitude: Number(meanLat.toFixed(6)),
        centroidLongitude: Number(meanLon.toFixed(6)),
        memberCount: memberIncidents.length,
        corePoints: coreCount,
        borderPoints: borderCount,
        averageDistanceMeters: Number(avgDist.toFixed(1)),
        maxDistanceMeters: Number(maxDist.toFixed(1)),
        damageTypeBreakdown: damageBreakdown,
        severityBreakdown: severityBreakdown,
        verifiedCount,
        memberIncidentIds: memberIncidents.map((i) => i.id),
      });
    }

    const noiseCount = labels.filter((l) => l === -1).length;
    const silhouetteScore = computeGeographicSilhouetteScore(points, labels);

    return {
      algorithm: 'Geographic_DBSCAN_Haversine',
      epsMeters: this.epsMeters,
      minSamples: this.minSamples,
      totalPoints: n,
      clusterCount: clusterId,
      noiseCount,
      noiseRatio: Number((noiseCount / n).toFixed(4)),
      silhouetteScore: Number(silhouetteScore.toFixed(4)),
      clusters,
      diagnostics,
    };
  }
}


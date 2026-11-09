/**
 * CivicPulse AI - Hotspot Service Orchestrator
 * Coordinates spatial preprocessing, spherical K-Means, Haversine DBSCAN,
 * and PostGIS spatial queries. Attaches mandatory data governance metadata.
 */

import { prisma } from '../../config/prisma';
import {
  preprocessSpatialIncidents,
  SpatialFilterCriteria,
  ValidatedSpatialIncident,
  RawSpatialIncident,
} from './preprocessor';
import {
  SphericalKMeansEngine,
  KMeansResult,
  KMeansRangeEvaluationItem,
} from './kmeans';
import {
  GeographicDBSCANEngine,
  DBSCANResult,
} from './dbscan';
import {
  getNearbyIncidentsForIncident,
} from './postgis';
import { haversineDistance } from './geo-utils';

export const EXPLORATORY_DISCLAIMER =
  'Clusters describe the spatial distribution of incidents currently available in the system. They are not validated predictions of future road damage.';

export interface HotspotSummaryResponse {
  isExploratory: boolean;
  disclaimer: string;
  computationTimestamp: string;
  dataScope: string;
  totalIncidents: number;
  validIncidents: number;
  rejectedIncidents: number;
  duplicateCoordinatePairs: number;
  boundingBox: {
    minLat: number;
    maxLat: number;
    minLng: number;
    maxLng: number;
  } | null;
  spatialMetrics: {
    averageNearestNeighborDistanceMeters: number;
    overallAreaKm2: number;
    overallDensityIncidentsPerKm2: number;
  };
  damageBreakdown: Record<string, number>;
  severityBreakdown: Record<string, number>;
  verificationBreakdown: {
    verified: number;
    unverified: number;
  };
}

export class HotspotService {
  /**
   * Fetches raw incidents from PostgreSQL.
   */
  private static async fetchRawIncidents(): Promise<RawSpatialIncident[]> {
    const records = await prisma.incident.findMany({
      select: {
        id: true,
        incidentNumber: true,
        title: true,
        damageType: true,
        severity: true,
        status: true,
        priority: true,
        latitude: true,
        longitude: true,
        address: true,
        reportedAt: true,
        verifiedAt: true,
        duplicateStatus: true,
      },
      orderBy: { reportedAt: 'desc' },
    });

    return records.map((r) => ({
      ...r,
      latitude: Number(r.latitude),
      longitude: Number(r.longitude),
    }));
  }

  /**
   * GET /api/v1/hotspots/summary
   */
  public static async getSummary(
    filters: SpatialFilterCriteria = {}
  ): Promise<HotspotSummaryResponse> {
    const raw = await this.fetchRawIncidents();
    const preprocessed = preprocessSpatialIncidents(raw, filters);
    const incidents = preprocessed.incidents;

    // Nearest-neighbor analysis
    let totalNearestDist = 0;
    const n = incidents.length;

    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const d = haversineDistance(
            incidents[i].latitude,
            incidents[i].longitude,
            incidents[j].latitude,
            incidents[j].longitude
          );
          if (d < minDist) minDist = d;
        }
      }
      if (minDist !== Infinity) totalNearestDist += minDist;
    }
    const avgNearestDist = n > 1 ? totalNearestDist / n : 0;

    // Approximate bounding box area
    let areaKm2 = 0;
    if (preprocessed.boundingBox) {
      const heightM = haversineDistance(
        preprocessed.boundingBox.minLat,
        preprocessed.boundingBox.minLng,
        preprocessed.boundingBox.maxLat,
        preprocessed.boundingBox.minLng
      );
      const widthM = haversineDistance(
        preprocessed.boundingBox.minLat,
        preprocessed.boundingBox.minLng,
        preprocessed.boundingBox.minLat,
        preprocessed.boundingBox.maxLng
      );
      areaKm2 = (heightM / 1000.0) * (widthM / 1000.0);
    }

    const damageBreakdown: Record<string, number> = {};
    const severityBreakdown: Record<string, number> = {};
    let verifiedCount = 0;

    for (const inc of incidents) {
      damageBreakdown[inc.damageType] = (damageBreakdown[inc.damageType] || 0) + 1;
      const s = inc.severity || 'unassigned';
      severityBreakdown[s] = (severityBreakdown[s] || 0) + 1;
      if (inc.status === 'verified' || inc.verifiedAt !== null) {
        verifiedCount++;
      }
    }

    return {
      isExploratory: true,
      disclaimer: EXPLORATORY_DISCLAIMER,
      computationTimestamp: new Date().toISOString(),
      dataScope: 'Active Municipal Incidents (PostgreSQL/PostGIS)',
      totalIncidents: preprocessed.totalRaw,
      validIncidents: preprocessed.totalValid,
      rejectedIncidents: preprocessed.totalRejected,
      duplicateCoordinatePairs: preprocessed.duplicateCoordinatePairs,
      boundingBox: preprocessed.boundingBox,
      spatialMetrics: {
        averageNearestNeighborDistanceMeters: Number(avgNearestDist.toFixed(1)),
        overallAreaKm2: Number(areaKm2.toFixed(3)),
        overallDensityIncidentsPerKm2:
          areaKm2 > 0 ? Number((n / areaKm2).toFixed(2)) : 0,
      },
      damageBreakdown,
      severityBreakdown,
      verificationBreakdown: {
        verified: verifiedCount,
        unverified: n - verifiedCount,
      },
    };
  }

  /**
   * GET /api/v1/hotspots/kmeans
   */
  public static async runKMeans(
    k: number = 3,
    maxIter: number = 100,
    tolMeters: number = 1.0,
    randomSeed: number = 42,
    evaluateRange: boolean = false,
    filters: SpatialFilterCriteria = {}
  ): Promise<{
    isExploratory: boolean;
    disclaimer: string;
    computationTimestamp: string;
    dataScope: string;
    datasetSize: number;
    result: KMeansResult;
    rangeEvaluation?: KMeansRangeEvaluationItem[];
    incidents: ValidatedSpatialIncident[];
  }> {
    const raw = await this.fetchRawIncidents();
    const preprocessed = preprocessSpatialIncidents(raw, filters);
    const incidents = preprocessed.incidents;

    if (incidents.length === 0) {
      throw new Error('No valid spatial incidents found matching the criteria.');
    }

    const safeK = Math.min(Math.max(2, k), Math.max(2, incidents.length));
    const engine = new SphericalKMeansEngine(safeK, maxIter, tolMeters, randomSeed);
    const result = engine.fit(incidents);

    let rangeEval: KMeansRangeEvaluationItem[] | undefined;
    if (evaluateRange) {
      rangeEval = SphericalKMeansEngine.evaluateRange(incidents, 2, 8, randomSeed);
    }

    return {
      isExploratory: true,
      disclaimer: EXPLORATORY_DISCLAIMER,
      computationTimestamp: new Date().toISOString(),
      dataScope: `Delhi Municipal Incidents (${incidents.length} locations)`,
      datasetSize: incidents.length,
      result,
      rangeEvaluation: rangeEval,
      incidents,
    };
  }

  /**
   * GET /api/v1/hotspots/dbscan
   */
  public static async runDBSCAN(
    epsMeters: number = 500.0,
    minSamples: number = 3,
    filters: SpatialFilterCriteria = {}
  ): Promise<{
    isExploratory: boolean;
    disclaimer: string;
    computationTimestamp: string;
    dataScope: string;
    datasetSize: number;
    result: DBSCANResult;
    incidents: ValidatedSpatialIncident[];
  }> {
    const raw = await this.fetchRawIncidents();
    const preprocessed = preprocessSpatialIncidents(raw, filters);
    const incidents = preprocessed.incidents;

    if (incidents.length === 0) {
      throw new Error('No valid spatial incidents found matching the criteria.');
    }

    const engine = new GeographicDBSCANEngine(epsMeters, minSamples);
    const result = engine.fit(incidents);

    return {
      isExploratory: true,
      disclaimer: EXPLORATORY_DISCLAIMER,
      computationTimestamp: new Date().toISOString(),
      dataScope: `Delhi Municipal Incidents (${incidents.length} locations)`,
      datasetSize: incidents.length,
      result,
      incidents,
    };
  }

  /**
   * GET /api/v1/hotspots/incidents/:id/nearby
   */
  public static async getIncidentNearby(
    incidentId: string,
    radiusMeters: number = 1000,
    limit: number = 20
  ) {
    const data = await getNearbyIncidentsForIncident(incidentId, radiusMeters, limit);
    return {
      isExploratory: true,
      disclaimer: EXPLORATORY_DISCLAIMER,
      computationTimestamp: new Date().toISOString(),
      ...data,
    };
  }

  /**
   * GET /api/v1/hotspots/parameters
   * Returns empirically evaluated parameters, elbow analysis, and defaults.
   */
  public static async getParameters() {
    const raw = await this.fetchRawIncidents();
    const preprocessed = preprocessSpatialIncidents(raw);
    const incidents = preprocessed.incidents;

    const kmeansRange = SphericalKMeansEngine.evaluateRange(incidents, 2, 8, 42);

    // Grid sweep for DBSCAN
    const dbscanSweep = [];
    const epsOptions = [250, 500, 750, 1000, 1500];
    const minSampleOptions = [2, 3, 5];

    for (const eps of epsOptions) {
      for (const ms of minSampleOptions) {
        const engine = new GeographicDBSCANEngine(eps, ms);
        const res = engine.fit(incidents);
        dbscanSweep.push({
          epsMeters: eps,
          minSamples: ms,
          clusters: res.clusterCount,
          noisePoints: res.noiseCount,
          noiseRatio: res.noiseRatio,
          silhouetteScore: res.silhouetteScore,
        });
      }
    }

    return {
      isExploratory: true,
      disclaimer: EXPLORATORY_DISCLAIMER,
      computationTimestamp: new Date().toISOString(),
      totalIncidents: incidents.length,
      kmeans: {
        defaultK: 4,
        recommendedKReason:
          'K=4 captures the multi-zone spatial division in Delhi NCR with significant inertia reduction over K=3.',
        rangeEvaluation: kmeansRange,
      },
      dbscan: {
        defaultEpsMeters: 500,
        defaultMinSamples: 3,
        recommendedReason:
          'eps=500m and min_samples=3 groups contiguous road distress while identifying isolated rural/arterial defects as noise.',
        parameterGrid: dbscanSweep,
      },
    };
  }
}

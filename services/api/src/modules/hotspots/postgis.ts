/**
 * CivicPulse AI - PostGIS Geospatial Service
 * Leverages PostgreSQL 16 + PostGIS 3.4 geography column and GIST index
 * for spatial radius queries, local incident density, and native spatial clustering.
 */

import { prisma } from '../../config/prisma';

export interface NearbyIncidentResult {
  id: string;
  incidentNumber: string;
  title: string;
  damageType: string;
  severity: string | null;
  status: string;
  priority: string | null;
  latitude: number;
  longitude: number;
  address: string | null;
  reportedAt: Date;
  distanceMeters: number;
}

export interface SpatialDensityResult {
  centerLatitude: number;
  centerLongitude: number;
  radiusMeters: number;
  incidentCount: number;
  areaKm2: number;
  densityIncidentsPerKm2: number;
}

export interface PostGISClusterRow {
  id: string;
  incident_number: string;
  latitude: number;
  longitude: number;
  damage_type: string;
  severity: string | null;
  cluster_id: number | null; // null represents noise in PostGIS ST_ClusterDBSCAN
}

/**
 * Retrieves all incidents within radiusMeters of a target coordinate using PostGIS ST_DWithin and GIST index.
 */
export async function getIncidentsWithinRadius(
  centerLat: number,
  centerLng: number,
  radiusMeters: number = 1000,
  limit: number = 50
): Promise<NearbyIncidentResult[]> {
  const safeRadius = Math.max(1, Math.min(radiusMeters, 50000));
  const safeLimit = Math.max(1, Math.min(limit, 200));

  const rows = await prisma.$queryRaw<
    Array<{
      id: string;
      incident_number: string;
      title: string;
      damage_type: string;
      severity: string | null;
      status: string;
      priority: string | null;
      latitude: any;
      longitude: any;
      address: string | null;
      reported_at: Date;
      distance_meters: number;
    }>
  >`
    SELECT id, incident_number, title, damage_type, severity, status, priority,
           latitude, longitude, address, reported_at,
           ST_Distance(location, ST_SetSRID(ST_MakePoint(${centerLng}, ${centerLat}), 4326)::geography) AS distance_meters
    FROM incidents
    WHERE ST_DWithin(location, ST_SetSRID(ST_MakePoint(${centerLng}, ${centerLat}), 4326)::geography, ${safeRadius})
    ORDER BY distance_meters ASC
    LIMIT ${safeLimit};
  `;

  return rows.map((r) => ({
    id: r.id,
    incidentNumber: r.incident_number,
    title: r.title,
    damageType: r.damage_type,
    severity: r.severity,
    status: r.status,
    priority: r.priority,
    latitude: Number(r.latitude),
    longitude: Number(r.longitude),
    address: r.address,
    reportedAt: r.reported_at,
    distanceMeters: Number(Number(r.distance_meters).toFixed(1)),
  }));
}

/**
 * Calculates spatial defect density within a circular zone in incidents per square kilometer.
 */
export async function calculateSpatialDensity(
  centerLat: number,
  centerLng: number,
  radiusMeters: number = 1000
): Promise<SpatialDensityResult> {
  const safeRadius = Math.max(1, Math.min(radiusMeters, 50000));
  const radiusKm = safeRadius / 1000.0;
  const areaKm2 = Math.PI * radiusKm * radiusKm;

  const countResult = await prisma.$queryRaw<Array<{ count: bigint }>>`
    SELECT count(*)::bigint AS count
    FROM incidents
    WHERE ST_DWithin(location, ST_SetSRID(ST_MakePoint(${centerLng}, ${centerLat}), 4326)::geography, ${safeRadius});
  `;

  const incidentCount = Number(countResult[0]?.count || 0);
  const density = areaKm2 > 0 ? incidentCount / areaKm2 : 0;

  return {
    centerLatitude: centerLat,
    centerLongitude: centerLng,
    radiusMeters: safeRadius,
    incidentCount,
    areaKm2: Number(areaKm2.toFixed(3)),
    densityIncidentsPerKm2: Number(density.toFixed(2)),
  };
}

/**
 * Finds incidents nearby an existing incident by ID using PostGIS.
 */
export async function getNearbyIncidentsForIncident(
  incidentId: string,
  radiusMeters: number = 1000,
  limit: number = 20
): Promise<{
  targetIncident: NearbyIncidentResult | null;
  nearby: NearbyIncidentResult[];
  density: SpatialDensityResult | null;
}> {
  const target = await prisma.incident.findUnique({
    where: { id: incidentId },
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
    },
  });

  if (!target) {
    return { targetIncident: null, nearby: [], density: null };
  }

  const lat = Number(target.latitude);
  const lng = Number(target.longitude);

  const nearby = await getIncidentsWithinRadius(lat, lng, radiusMeters, limit + 1);
  const filteredNearby = nearby.filter((i) => i.id !== incidentId).slice(0, limit);
  const density = await calculateSpatialDensity(lat, lng, radiusMeters);

  return {
    targetIncident: {
      id: target.id,
      incidentNumber: target.incidentNumber,
      title: target.title,
      damageType: target.damageType,
      severity: target.severity,
      status: target.status,
      priority: target.priority,
      latitude: lat,
      longitude: lng,
      address: target.address,
      reportedAt: target.reportedAt,
      distanceMeters: 0,
    },
    nearby: filteredNearby,
    density,
  };
}

/**
 * Runs PostGIS native ST_ClusterDBSCAN for database-side density clustering.
 * Converts radius in meters to approximate decimal degrees for planar projection.
 */
export async function runPostGISNativeDBSCAN(
  epsMeters: number = 500,
  minSamples: number = 3
): Promise<{
  algorithm: string;
  epsMeters: number;
  minSamples: number;
  totalPoints: number;
  clustersCount: number;
  noiseCount: number;
  rows: PostGISClusterRow[];
}> {
  // Approximate conversion: 1 degree latitude ~ 111,320 meters
  const epsDegrees = epsMeters / 111320.0;

  const results = await prisma.$queryRaw<
    Array<{
      id: string;
      incident_number: string;
      latitude: any;
      longitude: any;
      damage_type: string;
      severity: string | null;
      cluster_id: number | null;
    }>
  >`
    SELECT id, incident_number, latitude, longitude, damage_type, severity,
           ST_ClusterDBSCAN(location::geometry, eps := ${epsDegrees}::float8, minpoints := ${minSamples}::integer) OVER () AS cluster_id
    FROM incidents;
  `;

  const rows: PostGISClusterRow[] = results.map((r) => ({
    id: r.id,
    incident_number: r.incident_number,
    latitude: Number(r.latitude),
    longitude: Number(r.longitude),
    damage_type: r.damage_type,
    severity: r.severity,
    cluster_id: r.cluster_id !== null ? Number(r.cluster_id) : null,
  }));

  const clusterIds = new Set(rows.map((r) => r.cluster_id).filter((c) => c !== null));
  const noiseCount = rows.filter((r) => r.cluster_id === null).length;

  return {
    algorithm: 'PostGIS_Native_ST_ClusterDBSCAN',
    epsMeters,
    minSamples,
    totalPoints: rows.length,
    clustersCount: clusterIds.size,
    noiseCount,
    rows,
  };
}

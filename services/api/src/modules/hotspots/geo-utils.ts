/**
 * CivicPulse AI - Geographic & Spherical Mathematics Utilities
 * Provides exact Haversine distance calculations, WGS84 unit-sphere coordinate
 * transformations, and geographic clustering metrics.
 */

export const EARTH_RADIUS_METERS = 6371000.0; // WGS84 mean Earth radius

export interface GeoPoint {
  latitude: number;
  longitude: number;
}

export interface CartesianPoint3D {
  x: number;
  y: number;
  z: number;
}

/**
 * Validates that latitude and longitude conform to WGS84 standard ranges.
 */
export function validateCoordinates(
  lat: number,
  lon: number
): { valid: boolean; error?: string } {
  if (lat === null || lat === undefined || lon === null || lon === undefined) {
    return { valid: false, error: 'Latitude and Longitude cannot be null or undefined.' };
  }
  if (typeof lat !== 'number' || typeof lon !== 'number') {
    return { valid: false, error: 'Latitude and Longitude must be numerical values.' };
  }
  if (isNaN(lat) || isNaN(lon) || !isFinite(lat) || !isFinite(lon)) {
    return { valid: false, error: 'Coordinates cannot be NaN or Infinite.' };
  }
  if (lat < -90.0 || lat > 90.0) {
    return { valid: false, error: `Latitude ${lat} out of valid WGS84 range [-90.0, 90.0].` };
  }
  if (lon < -180.0 || lon > 180.0) {
    return { valid: false, error: `Longitude ${lon} out of valid WGS84 range [-180.0, 180.0].` };
  }
  return { valid: true };
}

/**
 * Computes great-circle distance (in meters) between two WGS84 points
 * using the numerically stable Haversine formula.
 */
export function haversineDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const phi1 = (lat1 * Math.PI) / 180.0;
  const phi2 = (lat2 * Math.PI) / 180.0;
  const deltaPhi = ((lat2 - lat1) * Math.PI) / 180.0;
  const deltaLambda = ((lon2 - lon1) * Math.PI) / 180.0;

  const sinDeltaPhiHalf = Math.sin(deltaPhi / 2.0);
  const sinDeltaLambdaHalf = Math.sin(deltaLambda / 2.0);

  const a =
    sinDeltaPhiHalf * sinDeltaPhiHalf +
    Math.cos(phi1) * Math.cos(phi2) * sinDeltaLambdaHalf * sinDeltaLambdaHalf;

  const clampedA = Math.min(Math.max(a, 0.0), 1.0);
  const c = 2.0 * Math.atan2(Math.sqrt(clampedA), Math.sqrt(1.0 - clampedA));

  return EARTH_RADIUS_METERS * c;
}

/**
 * Projects WGS84 (latitude, longitude) in degrees onto a 3D Cartesian unit sphere:
 * x = cos(lat) * cos(lon)
 * y = cos(lat) * sin(lon)
 * z = sin(lat)
 */
export function latLonToCartesian(lat: number, lon: number): CartesianPoint3D {
  const latRad = (lat * Math.PI) / 180.0;
  const lonRad = (lon * Math.PI) / 180.0;

  return {
    x: Math.cos(latRad) * Math.cos(lonRad),
    y: Math.cos(latRad) * Math.sin(lonRad),
    z: Math.sin(latRad),
  };
}

/**
 * Inverts 3D Cartesian unit-sphere coordinates back into WGS84 (lat, lon) in degrees:
 * lat = arcsin(z)
 * lon = atan2(y, x)
 */
export function cartesianToLatLon(cart: CartesianPoint3D): GeoPoint {
  const norm = Math.sqrt(cart.x * cart.x + cart.y * cart.y + cart.z * cart.z);
  const safeNorm = norm === 0 ? 1.0 : norm;

  const unitX = cart.x / safeNorm;
  const unitY = cart.y / safeNorm;
  const unitZ = Math.min(Math.max(cart.z / safeNorm, -1.0), 1.0);

  const latRad = Math.asin(unitZ);
  const lonRad = Math.atan2(unitY, unitX);

  return {
    latitude: (latRad * 180.0) / Math.PI,
    longitude: (lonRad * 180.0) / Math.PI,
  };
}

/**
 * Computes pairwise Haversine distance matrix for an array of points in meters.
 */
export function computePairwiseHaversineMatrix(points: GeoPoint[]): number[][] {
  const n = points.length;
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dist = haversineDistance(
        points[i].latitude,
        points[i].longitude,
        points[j].latitude,
        points[j].longitude
      );
      matrix[i][j] = dist;
      matrix[j][i] = dist;
    }
  }

  return matrix;
}

/**
 * Computes geographic Silhouette Score [-1.0, 1.0] using exact pairwise
 * Haversine distances in meters. Handles noise labels (-1).
 */
export function computeGeographicSilhouetteScore(
  points: GeoPoint[],
  labels: number[]
): number {
  const n = points.length;
  if (n <= 1) return 0.0;

  const uniqueClusters = Array.from(new Set(labels)).filter((l) => l >= 0);
  if (uniqueClusters.length <= 1 || uniqueClusters.length >= n) {
    return 0.0;
  }

  const distMatrix = computePairwiseHaversineMatrix(points);
  let silhouetteSum = 0.0;
  let validEvaluations = 0;

  for (let i = 0; i < n; i++) {
    const ownCluster = labels[i];
    if (ownCluster < 0) continue; // Skip noise

    // 1. Mean intra-cluster distance a(i)
    let intraSum = 0;
    let intraCount = 0;
    for (let j = 0; j < n; j++) {
      if (i !== j && labels[j] === ownCluster) {
        intraSum += distMatrix[i][j];
        intraCount++;
      }
    }
    const a_i = intraCount > 0 ? intraSum / intraCount : 0.0;

    // 2. Mean nearest-cluster distance b(i)
    let minOtherMean = Infinity;
    for (const otherCluster of uniqueClusters) {
      if (otherCluster === ownCluster) continue;

      let otherSum = 0;
      let otherCount = 0;
      for (let j = 0; j < n; j++) {
        if (labels[j] === otherCluster) {
          otherSum += distMatrix[i][j];
          otherCount++;
        }
      }
      if (otherCount > 0) {
        const meanDist = otherSum / otherCount;
        if (meanDist < minOtherMean) {
          minOtherMean = meanDist;
        }
      }
    }
    const b_i = minOtherMean === Infinity ? 0.0 : minOtherMean;

    const max_ab = Math.max(a_i, b_i);
    const s_i = max_ab > 0 ? (b_i - a_i) / max_ab : 0.0;

    silhouetteSum += s_i;
    validEvaluations++;
  }

  return validEvaluations > 0 ? silhouetteSum / validEvaluations : 0.0;
}


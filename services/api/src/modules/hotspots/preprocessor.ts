/**
 * CivicPulse AI - Spatial Data Preprocessor & Integrity Validator
 * Validates WGS84 coordinates, detects duplicate coordinate pairs,
 * applies spatio-temporal filters, and preserves complete incident metadata.
 */

import { validateCoordinates } from './geo-utils';

export interface RawSpatialIncident {
  id: string;
  incidentNumber: string;
  title: string;
  damageType: string;
  severity: string | null;
  status: string;
  priority: string | null;
  latitude: number | string | any;
  longitude: number | string | any;
  address: string | null;
  reportedAt: Date | string;
  verifiedAt: Date | string | null;
  duplicateStatus?: string;
}

export interface ValidatedSpatialIncident {
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
  verifiedAt: Date | null;
  duplicateStatus: string;
  isCoordinateDuplicate: boolean;
  duplicateCoordinateCount: number;
}

export interface SpatialFilterCriteria {
  dateFrom?: Date | string;
  dateTo?: Date | string;
  damageType?: string;
  verificationStatus?: 'all' | 'verified' | 'unverified';
  minLat?: number;
  maxLat?: number;
  minLng?: number;
  maxLng?: number;
}

export interface PreprocessingResult {
  incidents: ValidatedSpatialIncident[];
  totalRaw: number;
  totalValid: number;
  totalRejected: number;
  rejectedReasons: { incidentId: string; reason: string }[];
  duplicateCoordinatePairs: number;
  boundingBox: {
    minLat: number;
    maxLat: number;
    minLng: number;
    maxLng: number;
  } | null;
}

export function preprocessSpatialIncidents(
  rawIncidents: RawSpatialIncident[],
  filters: SpatialFilterCriteria = {}
): PreprocessingResult {
  const rejectedReasons: { incidentId: string; reason: string }[] = [];
  const validList: ValidatedSpatialIncident[] = [];

  const coordKeyCounts = new Map<string, number>();

  // Pass 1: Parse, validate and count coordinate occurrences
  for (const raw of rawIncidents) {
    const lat = typeof raw.latitude === 'string' ? parseFloat(raw.latitude) : raw.latitude;
    const lon = typeof raw.longitude === 'string' ? parseFloat(raw.longitude) : raw.longitude;

    const validation = validateCoordinates(lat, lon);
    if (!validation.valid) {
      rejectedReasons.push({
        incidentId: raw.id,
        reason: validation.error || 'Invalid coordinates.',
      });
      continue;
    }

    // Key precision to 6 decimal places (~0.1m)
    const key = `${lat.toFixed(6)}_${lon.toFixed(6)}`;
    coordKeyCounts.set(key, (coordKeyCounts.get(key) || 0) + 1);

    const repDate = new Date(raw.reportedAt);
    const verDate = raw.verifiedAt ? new Date(raw.verifiedAt) : null;

    validList.push({
      id: raw.id,
      incidentNumber: raw.incidentNumber,
      title: raw.title,
      damageType: raw.damageType,
      severity: raw.severity,
      status: raw.status,
      priority: raw.priority,
      latitude: lat,
      longitude: lon,
      address: raw.address,
      reportedAt: isNaN(repDate.getTime()) ? new Date() : repDate,
      verifiedAt: verDate && !isNaN(verDate.getTime()) ? verDate : null,
      duplicateStatus: raw.duplicateStatus || 'none',
      isCoordinateDuplicate: false,
      duplicateCoordinateCount: 1,
    });
  }

  // Pass 2: Tag duplicate coordinates
  let duplicatePairsCount = 0;
  for (const item of validList) {
    const key = `${item.latitude.toFixed(6)}_${item.longitude.toFixed(6)}`;
    const count = coordKeyCounts.get(key) || 1;
    item.duplicateCoordinateCount = count;
    if (count > 1) {
      item.isCoordinateDuplicate = true;
      duplicatePairsCount++;
    }
  }

  // Pass 3: Apply Filters
  let filtered = validList;

  if (filters.dateFrom) {
    const from = new Date(filters.dateFrom);
    if (!isNaN(from.getTime())) {
      filtered = filtered.filter((i) => i.reportedAt >= from);
    }
  }

  if (filters.dateTo) {
    const to = new Date(filters.dateTo);
    if (!isNaN(to.getTime())) {
      filtered = filtered.filter((i) => i.reportedAt <= to);
    }
  }

  if (filters.damageType && filters.damageType !== 'all') {
    const targetType = filters.damageType.toLowerCase();
    filtered = filtered.filter((i) => i.damageType.toLowerCase() === targetType);
  }

  if (filters.verificationStatus && filters.verificationStatus !== 'all') {
    if (filters.verificationStatus === 'verified') {
      filtered = filtered.filter((i) => i.status === 'verified' || i.verifiedAt !== null);
    } else if (filters.verificationStatus === 'unverified') {
      filtered = filtered.filter((i) => i.status !== 'verified' && i.verifiedAt === null);
    }
  }

  if (filters.minLat !== undefined && !isNaN(filters.minLat)) {
    filtered = filtered.filter((i) => i.latitude >= filters.minLat!);
  }
  if (filters.maxLat !== undefined && !isNaN(filters.maxLat)) {
    filtered = filtered.filter((i) => i.latitude <= filters.maxLat!);
  }
  if (filters.minLng !== undefined && !isNaN(filters.minLng)) {
    filtered = filtered.filter((i) => i.longitude >= filters.minLng!);
  }
  if (filters.maxLng !== undefined && !isNaN(filters.maxLng)) {
    filtered = filtered.filter((i) => i.longitude <= filters.maxLng!);
  }

  // Compute bounding box
  let boundingBox = null;
  if (filtered.length > 0) {
    let minLat = filtered[0].latitude;
    let maxLat = filtered[0].latitude;
    let minLng = filtered[0].longitude;
    let maxLng = filtered[0].longitude;

    for (const item of filtered) {
      if (item.latitude < minLat) minLat = item.latitude;
      if (item.latitude > maxLat) maxLat = item.latitude;
      if (item.longitude < minLng) minLng = item.longitude;
      if (item.longitude > maxLng) maxLng = item.longitude;
    }

    boundingBox = {
      minLat: Number(minLat.toFixed(6)),
      maxLat: Number(maxLat.toFixed(6)),
      minLng: Number(minLng.toFixed(6)),
      maxLng: Number(maxLng.toFixed(6)),
    };
  }

  return {
    incidents: filtered,
    totalRaw: rawIncidents.length,
    totalValid: filtered.length,
    totalRejected: rejectedReasons.length,
    rejectedReasons,
    duplicateCoordinatePairs: Math.floor(duplicatePairsCount / 2),
    boundingBox,
  };
}

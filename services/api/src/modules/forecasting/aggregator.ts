import { prisma } from '../../config/prisma';
import {
  AggregationFrequency,
  BucketObservationStatus,
  ForecastRegion,
  RegionalTimeSeries,
  TimeSeriesBucket,
} from './types';

export interface CollectionCoverageWindow {
  start: Date;
  end: Date;
  isCoverageActive: boolean;
}

export interface CollectionOutageWindow {
  start: Date;
  end: Date;
  reason: string;
}

export interface AggregationOptions {
  region?: ForecastRegion;
  frequency?: AggregationFrequency;
  damageCategory?: string;
  verificationStatus?: 'all' | 'verified_only' | 'unverified_only';
  canonicalOnly?: boolean;
  timeZone?: string;
  customStartDate?: Date;
  customEndDate?: Date;
  coverageWindows?: CollectionCoverageWindow[];
  outageWindows?: CollectionOutageWindow[];
}

/**
 * Validates and classifies geographic coordinates into administrative forecasting regions.
 * Reuses validated bounding boxes established in Stage 3 & 5 audits.
 */
export function classifyCoordinateRegion(lat: number, lon: number): 'BENGALURU_METRO' | 'DELHI_NCR' | 'UNKNOWN' {
  if (lat >= 12.5 && lat <= 13.5 && lon >= 77.2 && lon <= 78.0) {
    return 'BENGALURU_METRO';
  }
  if (lat >= 28.0 && lat <= 29.0 && lon >= 76.8 && lon <= 77.8) {
    return 'DELHI_NCR';
  }
  return 'UNKNOWN';
}

/**
 * Known empirical field collection coverage window for CivicPulse AI Phase 2/3 dataset.
 * Active inspector and citizen field logging occurred between 11:00 and 18:00 UTC on 2026-11-07.
 */
const DEFAULT_OPERATIONAL_COVERAGE_WINDOWS: CollectionCoverageWindow[] = [
  {
    start: new Date('2026-11-07T11:00:00.000Z'),
    end: new Date('2026-11-07T18:00:00.000Z'),
    isCoverageActive: true,
  },
];

/**
 * Format timestamp in local display timezone (e.g. Asia/Kolkata / IST UTC+05:30).
 */
export function formatLocalDisplayTimestamp(date: Date, timeZone: string = 'Asia/Kolkata'): string {
  try {
    return new Intl.DateTimeFormat('en-IN', {
      timeZone,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    }).format(date);
  } catch {
    return date.toISOString();
  }
}

/**
 * Format bucket period label for clean human readability and charting.
 */
function formatBucketLabel(date: Date, frequency: AggregationFrequency): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  const y = date.getUTCFullYear();
  const m = pad(date.getUTCMonth() + 1);
  const d = pad(date.getUTCDate());
  const h = pad(date.getUTCHours());

  switch (frequency) {
    case 'hourly':
      return `${y}-${m}-${d} ${h}:00`;
    case 'daily':
      return `${y}-${m}-${d}`;
    case 'weekly': {
      // Approximate ISO week label
      const dObj = new Date(Date.UTC(y, date.getUTCMonth(), date.getUTCDate()));
      const dayNum = dObj.getUTCDay() || 7;
      dObj.setUTCDate(dObj.getUTCDate() + 4 - dayNum);
      const yearStart = new Date(Date.UTC(dObj.getUTCFullYear(), 0, 1));
      const weekNo = Math.ceil(((dObj.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
      return `${y}-W${pad(weekNo)}`;
    }
    case 'monthly':
      return `${y}-${m}`;
  }
}

/**
 * Truncate a Date to the start of its bucket period in UTC.
 */
function floorToBucket(date: Date, frequency: AggregationFrequency): Date {
  const d = new Date(date.getTime());
  switch (frequency) {
    case 'hourly':
      d.setUTCMinutes(0, 0, 0);
      return d;
    case 'daily':
      d.setUTCHours(0, 0, 0, 0);
      return d;
    case 'weekly': {
      d.setUTCHours(0, 0, 0, 0);
      const day = d.getUTCDay(); // 0 is Sunday
      const diff = d.getUTCDate() - day + (day === 0 ? -6 : 1); // Monday start
      d.setUTCDate(diff);
      return d;
    }
    case 'monthly':
      d.setUTCDate(1);
      d.setUTCHours(0, 0, 0, 0);
      return d;
  }
}

/**
 * Step forward by one period interval in UTC.
 */
function stepBucket(date: Date, frequency: AggregationFrequency): Date {
  const d = new Date(date.getTime());
  switch (frequency) {
    case 'hourly':
      d.setUTCHours(d.getUTCHours() + 1);
      return d;
    case 'daily':
      d.setUTCDate(d.getUTCDate() + 1);
      return d;
    case 'weekly':
      d.setUTCDate(d.getUTCDate() + 7);
      return d;
    case 'monthly':
      d.setUTCMonth(d.getUTCMonth() + 1);
      return d;
  }
}

export class IncidentTimeSeriesAggregator {
  /**
   * Fetches real incident records from PostgreSQL and aggregates them into uniform time series buckets.
   */
  public static async aggregate(options: AggregationOptions = {}): Promise<RegionalTimeSeries> {
    const region = options.region ?? 'ALL';
    const frequency = options.frequency ?? 'hourly';
    const canonicalOnly = options.canonicalOnly ?? true;
    const verificationStatus = options.verificationStatus ?? 'all';
    const timeZone = options.timeZone ?? 'Asia/Kolkata';

    // 1. Fetch all real incident records
    const incidents = await prisma.incident.findMany({
      select: {
        id: true,
        incidentNumber: true,
        latitude: true,
        longitude: true,
        damageType: true,
        severity: true,
        status: true,
        duplicateStatus: true,
        canonicalIncidentId: true,
        reportedAt: true,
        createdAt: true,
      },
      orderBy: {
        reportedAt: 'asc',
      },
    });

    // 2. Filter incidents in-memory to ensure strict geographic and status governance
    const filtered = incidents.filter((inc) => {
      const lat = Number(inc.latitude);
      const lon = Number(inc.longitude);
      const incRegion = classifyCoordinateRegion(lat, lon);

      if (region !== 'ALL' && incRegion !== region) {
        return false;
      }

      if (options.damageCategory && inc.damageType !== options.damageCategory) {
        return false;
      }

      if (canonicalOnly && inc.duplicateStatus === 'duplicate' && inc.canonicalIncidentId) {
        return false;
      }

      if (verificationStatus === 'verified_only' && inc.status !== 'verified') {
        return false;
      }

      if (verificationStatus === 'unverified_only' && inc.status === 'verified') {
        return false;
      }

      return true;
    });

    if (filtered.length === 0) {
      return {
        region,
        frequency,
        totalIncidents: 0,
        windowStart: new Date().toISOString(),
        windowEnd: new Date().toISOString(),
        buckets: [],
        timeZone,
      };
    }

    // 3. Determine time boundary window
    const incidentTimestamps = filtered.map((i) => new Date(i.reportedAt || i.createdAt).getTime());
    const minTimestamp = options.customStartDate
      ? options.customStartDate.getTime()
      : Math.min(...incidentTimestamps);
    const maxTimestamp = options.customEndDate
      ? options.customEndDate.getTime()
      : Math.max(...incidentTimestamps);

    const bucketStart = floorToBucket(new Date(minTimestamp), frequency);
    const bucketEnd = floorToBucket(new Date(maxTimestamp), frequency);

    // 4. Construct continuous time grid
    const buckets: TimeSeriesBucket[] = [];
    let currentBucket = new Date(bucketStart.getTime());
    let bucketIndex = 0;

    // Map incidents into bucket keys
    const bucketKeyToIncidents = new Map<string, typeof filtered>();

    for (const inc of filtered) {
      const incDate = new Date(inc.reportedAt || inc.createdAt);
      const incBucket = floorToBucket(incDate, frequency);
      const key = incBucket.toISOString();

      if (!bucketKeyToIncidents.has(key)) {
        bucketKeyToIncidents.set(key, []);
      }
      bucketKeyToIncidents.get(key)!.push(inc);
    }

    // Ensure we iterate up to and including bucketEnd
    while (currentBucket.getTime() <= bucketEnd.getTime()) {
      const key = currentBucket.toISOString();
      const bucketIncidents = bucketKeyToIncidents.get(key) || [];

      const damageBreakdown: Record<string, number> = {};
      let verifiedCount = 0;
      let criticalCount = 0;

      for (const inc of bucketIncidents) {
        damageBreakdown[inc.damageType] = (damageBreakdown[inc.damageType] || 0) + 1;
        if (inc.status === 'verified') {
          verifiedCount++;
        }
        if (inc.severity === 'critical') {
          criticalCount++;
        }
      }

      const count = bucketIncidents.length;
      let observationStatus: BucketObservationStatus = 'UNOBSERVED';
      let outageReason: string | undefined = undefined;

      const coverageWindows = options.coverageWindows || DEFAULT_OPERATIONAL_COVERAGE_WINDOWS;
      const outageWindows = options.outageWindows || [];

      // Check if bucket falls within an outage window
      const matchedOutage = outageWindows.find(
        (o) => currentBucket.getTime() >= o.start.getTime() && currentBucket.getTime() < o.end.getTime()
      );

      if (matchedOutage) {
        observationStatus = 'COLLECTION_OUTAGE';
        outageReason = matchedOutage.reason;
      } else if (count > 0) {
        observationStatus = 'OBSERVED_NONZERO';
      } else {
        // Zero count: only classify as OBSERVED_ZERO when independent coverage is established
        const hasCoverage = coverageWindows.some(
          (w) =>
            w.isCoverageActive &&
            currentBucket.getTime() >= w.start.getTime() &&
            currentBucket.getTime() <= w.end.getTime()
        );
        observationStatus = hasCoverage ? 'OBSERVED_ZERO' : 'UNOBSERVED';
      }

      const isZero = count === 0;
      const isObserved =
        observationStatus === 'OBSERVED_NONZERO' || observationStatus === 'OBSERVED_ZERO';

      buckets.push({
        bucketIndex,
        timestamp: key,
        periodLabel: formatBucketLabel(currentBucket, frequency),
        incidentCount: count,
        verifiedCount,
        criticalCount,
        damageBreakdown,
        observationStatus,
        isZero,
        isObserved,
        outageReason,
        displayTimestampLocal: formatLocalDisplayTimestamp(currentBucket, timeZone),
      });

      bucketIndex++;
      currentBucket = stepBucket(currentBucket, frequency);
    }

    return {
      region,
      frequency,
      totalIncidents: filtered.length,
      windowStart: bucketStart.toISOString(),
      windowEnd: new Date(maxTimestamp).toISOString(),
      buckets,
      timeZone,
    };
  }
}

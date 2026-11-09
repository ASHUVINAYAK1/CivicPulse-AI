export type AggregationFrequency = 'hourly' | 'daily' | 'weekly' | 'monthly';

export type ForecastRegion = 'BENGALURU_METRO' | 'DELHI_NCR' | 'ALL';

export type ModelType =
  | 'naive'
  | 'moving_average'
  | 'seasonal_naive'
  | 'linear_trend'
  | 'arima_scaffold'
  | 'lstm'
  | 'synthetic_seed';

export type ReadinessStatus = 'DATA_SUFFICIENT' | 'DATA_INSUFFICIENT' | 'PARTIALLY_SUFFICIENT';

export type ModelOperationalStatus = 'ACTIVE' | 'LOCKED' | 'EXPERIMENTAL' | 'DEPRECATED';

export type BucketObservationStatus =
  | 'OBSERVED_NONZERO'
  | 'OBSERVED_ZERO'
  | 'UNOBSERVED'
  | 'COLLECTION_OUTAGE';

export interface TimeSeriesBucket {
  bucketIndex: number;
  timestamp: string; // ISO string of bucket start in UTC
  periodLabel: string;
  incidentCount: number;
  verifiedCount: number;
  criticalCount: number;
  damageBreakdown: Record<string, number>;
  observationStatus: BucketObservationStatus;
  isZero: boolean;
  isObserved: boolean;
  outageReason?: string;
  displayTimestampLocal?: string; // Formatted in target display timezone (e.g. Asia/Kolkata)
}

export interface RegionalTimeSeries {
  region: ForecastRegion;
  frequency: AggregationFrequency;
  totalIncidents: number;
  windowStart: string;
  windowEnd: string;
  buckets: TimeSeriesBucket[];
  timeZone: string;
}

export interface DataReadinessEvaluation {
  status: ReadinessStatus;
  region: ForecastRegion;
  frequency: AggregationFrequency;
  modelType: ModelType;
  temporalSpanHours: number;
  temporalSpanDays: number;
  totalObservedBuckets: number;
  nonZeroBuckets: number;
  minimumRequiredBuckets: number;
  seasonalPeriodsObserved: number;
  minimumRequiredSeasonalPeriods: number;
  isLocked: boolean;
  reasons: string[];
  recommendations: string[];
  evaluationTimestamp: string;
}

export interface ModelDescriptor {
  id: string;
  name: string;
  modelType: ModelType;
  description: string;
  framework: string;
  status: ModelOperationalStatus;
  statusReason: string;
  minimumBucketsRequired: Record<AggregationFrequency, number>;
  seasonalPeriodsRequired: number;
  formulaDescription: string;
  pros: string[];
  cons: string[];
}

export interface ForecastPoint {
  period: string; // ISO string
  periodLabel: string;
  predictedIncidents: number;
  lowerBound: number;
  upperBound: number;
  confidenceInterval: number; // e.g. 0.95
}

export interface ForecastResult {
  id?: string;
  region: ForecastRegion;
  frequency: AggregationFrequency;
  modelType: ModelType;
  horizon: number;
  cutoffTimestamp: string;
  trainingWindowStart: string;
  trainingWindowEnd: string;
  historicalPointsCount: number;
  predictions: ForecastPoint[];
  evaluationMetrics?: {
    mae?: number;
    rmse?: number;
    mape?: number;
    directionalAccuracy?: number;
  };
  readinessStatus: ReadinessStatus;
  isSynthetic: boolean;
  provenance: {
    generatedAt: string;
    engineVersion: string;
    dataCutoff: string;
    modelHyperparameters?: Record<string, any>;
    governanceNote: string;
  };
}

export interface ForecastSummary {
  datasetOverview: {
    totalRealIncidents: number;
    longitudinalSpanHours: number;
    longitudinalSpanDays: number;
    earliestTimestamp: string;
    latestTimestamp: string;
    bengaluruCount: number;
    delhiCount: number;
    unassignedCount: number;
  };
  syntheticSeedAudit: {
    totalSyntheticForecasts: number;
    status: string;
    note: string;
  };
  governanceStatus: {
    overallReadiness: ReadinessStatus;
    predictiveMaintenanceGated: boolean;
    neuralForecastingLocked: boolean;
    gateReason: string;
  };
  availableFrequencies: AggregationFrequency[];
  availableRegions: ForecastRegion[];
}


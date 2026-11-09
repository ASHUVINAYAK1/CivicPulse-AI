import { prisma } from '../../config/prisma';
import {
  AggregationFrequency,
  ForecastRegion,
  ForecastResult,
  ForecastSummary,
  ModelDescriptor,
  ModelType,
  RegionalTimeSeries,
} from './types';
import { IncidentTimeSeriesAggregator } from './aggregator';
import { ForecastingReadinessGate } from './readiness-gate';
import { FORECASTING_MODELS, ForecastingEngine } from './models';

export interface GenerateForecastRequest {
  region: ForecastRegion;
  frequency: AggregationFrequency;
  modelType: ModelType;
  horizon?: number;
  saveToDb?: boolean;
  forceSimulate?: boolean;
}

export class ForecastingService {
  /**
   * High-level summary of operational data availability, governance gates, and synthetic seed audit.
   */
  public static async getSummary(): Promise<ForecastSummary> {
    const incidents = await prisma.incident.findMany({
      select: {
        id: true,
        latitude: true,
        longitude: true,
        reportedAt: true,
        createdAt: true,
      },
      orderBy: { reportedAt: 'asc' },
    });

    let bengaluruCount = 0;
    let delhiCount = 0;
    let unassignedCount = 0;

    const timestamps: number[] = [];

    for (const inc of incidents) {
      const lat = Number(inc.latitude);
      const lon = Number(inc.longitude);
      if (lat >= 12.5 && lat <= 13.5 && lon >= 77.2 && lon <= 78.0) {
        bengaluruCount++;
      } else if (lat >= 28.0 && lat <= 29.0 && lon >= 76.8 && lon <= 77.8) {
        delhiCount++;
      } else {
        unassignedCount++;
      }

      const t = new Date(inc.reportedAt || inc.createdAt).getTime();
      if (!isNaN(t)) {
        timestamps.push(t);
      }
    }

    const minTs = timestamps.length > 0 ? Math.min(...timestamps) : Date.now();
    const maxTs = timestamps.length > 0 ? Math.max(...timestamps) : Date.now();
    const spanMs = Math.max(0, maxTs - minTs);
    const spanHours = Math.round((spanMs / (1000 * 60 * 60)) * 10) / 10;
    const spanDays = Math.round((spanMs / (1000 * 60 * 60 * 24)) * 10) / 10;

    // Check synthetic forecasts in DB
    const syntheticForecastsCount = await prisma.incidentForecast.count({
      where: { isSynthetic: true },
    });

    // Check readiness gate
    const readiness = await ForecastingReadinessGate.evaluateReadiness({
      region: 'ALL',
      frequency: 'hourly',
      modelType: 'naive',
    });

    return {
      datasetOverview: {
        totalRealIncidents: incidents.length,
        longitudinalSpanHours: spanHours,
        longitudinalSpanDays: spanDays,
        earliestTimestamp: new Date(minTs).toISOString(),
        latestTimestamp: new Date(maxTs).toISOString(),
        bengaluruCount,
        delhiCount,
        unassignedCount,
      },
      syntheticSeedAudit: {
        totalSyntheticForecasts: syntheticForecastsCount,
        status: 'FLAGGED_SYNTHETIC',
        note: 'All existing seed forecast records in incident_forecasts are synthetic demo records created in prisma/seed.ts. Excluded from real model training.',
      },
      governanceStatus: {
        overallReadiness: readiness.status,
        predictiveMaintenanceGated: true,
        neuralForecastingLocked: true,
        gateReason:
          'Longitudinal observation window is 5.8 hours (single day). Minimum 30 continuous daily or 12 monthly periods required before statistical/neural forecasting can be unlocked.',
      },
      availableFrequencies: ['hourly', 'daily', 'weekly', 'monthly'],
      availableRegions: ['BENGALURU_METRO', 'DELHI_NCR', 'ALL'],
    };
  }

  /**
   * Fetches historical time-series aggregation.
   */
  public static async getHistory(options: {
    region?: ForecastRegion;
    frequency?: AggregationFrequency;
    damageCategory?: string;
    verificationStatus?: 'all' | 'verified_only' | 'unverified_only';
    canonicalOnly?: boolean;
    timeZone?: string;
  }): Promise<RegionalTimeSeries> {
    return IncidentTimeSeriesAggregator.aggregate(options);
  }

  /**
   * Retrieves full catalog of forecasting models with mathematical formulas and readiness constraints.
   */
  public static getModels(): ModelDescriptor[] {
    return FORECASTING_MODELS;
  }

  /**
   * Evaluates readiness for specific region, frequency, and model.
   */
  public static async checkReadiness(params: {
    region?: ForecastRegion;
    frequency?: AggregationFrequency;
    modelType?: ModelType;
  }) {
    return ForecastingReadinessGate.evaluateReadiness(params);
  }

  /**
   * Generate forecast if data is sufficient or dry-run simulated.
   */
  public static async generateForecast(params: GenerateForecastRequest): Promise<ForecastResult> {
    const horizon = params.horizon ?? 6;

    // 1. Audit readiness
    const readiness = await ForecastingReadinessGate.evaluateReadiness({
      region: params.region,
      frequency: params.frequency,
      modelType: params.modelType,
    });

    if (readiness.status === 'DATA_INSUFFICIENT' && !params.forceSimulate) {
      const err = new Error(
        `Forecasting blocked: DATA_INSUFFICIENT. ${readiness.reasons.join(' ')}`
      );
      (err as any).statusCode = 422;
      (err as any).readiness = readiness;
      throw err;
    }

    // 2. Fetch history
    const history = await IncidentTimeSeriesAggregator.aggregate({
      region: params.region,
      frequency: params.frequency,
      canonicalOnly: true,
    });

    // 3. Generate forecast points
    const predictions = ForecastingEngine.forecast(
      history.buckets,
      params.frequency,
      params.modelType,
      horizon
    );

    // 4. Chronological evaluation
    const metrics = ForecastingEngine.evaluateChronological(history.buckets, params.modelType);

    const result: ForecastResult = {
      region: params.region,
      frequency: params.frequency,
      modelType: params.modelType,
      horizon,
      cutoffTimestamp: history.windowEnd,
      trainingWindowStart: history.windowStart,
      trainingWindowEnd: history.windowEnd,
      historicalPointsCount: history.buckets.length,
      predictions,
      evaluationMetrics: metrics,
      readinessStatus: readiness.status,
      isSynthetic: params.forceSimulate === true,
      provenance: {
        generatedAt: new Date().toISOString(),
        engineVersion: 'CivicPulse-TimeSeries-v1.0.0',
        dataCutoff: history.windowEnd,
        governanceNote:
          readiness.status === 'DATA_INSUFFICIENT'
            ? 'SIMULATION ONLY: Insufficient longitudinal observation window. Not suitable for operational dispatch.'
            : 'Validated historical baseline prediction.',
      },
    };

    // 5. Persist to DB if requested and permitted
    if (params.saveToDb && predictions.length > 0) {
      for (const pt of predictions) {
        await prisma.incidentForecast.create({
          data: {
            regionId: params.region,
            forecastPeriod: new Date(pt.period),
            predictedIncidents: pt.predictedIncidents,
            lowerBound: pt.lowerBound,
            upperBound: pt.upperBound,
            aggregationFrequency: params.frequency,
            modelType: params.modelType,
            readinessStatus: readiness.status,
            isSynthetic: result.isSynthetic,
            trainingWindowStart: new Date(history.windowStart),
            trainingWindowEnd: new Date(history.windowEnd),
            metadata: {
              horizon,
              evaluationMetrics: metrics,
              periodLabel: pt.periodLabel,
            },
          },
        });
      }
    }

    return result;
  }
}

import {
  AggregationFrequency,
  DataReadinessEvaluation,
  ForecastRegion,
  ModelType,
  ReadinessStatus,
} from './types';
import { IncidentTimeSeriesAggregator } from './aggregator';

export interface ReadinessCheckParams {
  region?: ForecastRegion;
  frequency?: AggregationFrequency;
  modelType?: ModelType;
}

export const MINIMUM_BUCKET_THRESHOLDS: Record<ModelType, Record<AggregationFrequency, number>> = {
  naive: {
    hourly: 24, // 1 full day of diurnal cycle
    daily: 14,  // 2 full weeks
    weekly: 8,  // 2 months
    monthly: 6, // half a year
  },
  moving_average: {
    hourly: 48, // 2 full days
    daily: 30,  // 1 full month
    weekly: 12, // 1 quarter
    monthly: 12, // 1 full year
  },
  seasonal_naive: {
    hourly: 48,  // 2 complete diurnal cycles (24h * 2)
    daily: 28,   // 4 complete 7-day weekly cycles
    weekly: 52,  // 1 full year
    monthly: 24, // 2 full annual cycles
  },
  linear_trend: {
    hourly: 48,
    daily: 30,
    weekly: 16,
    monthly: 12,
  },
  arima_scaffold: {
    hourly: 72,
    daily: 45,
    weekly: 26,
    monthly: 18,
  },
  lstm: {
    hourly: 720,  // 30 days continuous
    daily: 365,   // 1 full year continuous
    weekly: 104,  // 2 full years
    monthly: 24,  // 2 full years
  },
  synthetic_seed: {
    hourly: 1,
    daily: 1,
    weekly: 1,
    monthly: 1,
  },
};

export class ForecastingReadinessGate {
  /**
   * Evaluates historical longitudinal readiness for time-series forecasting.
   */
  public static async evaluateReadiness(params: ReadinessCheckParams = {}): Promise<DataReadinessEvaluation> {
    const region = params.region ?? 'ALL';
    const frequency = params.frequency ?? 'hourly';
    const modelType = params.modelType ?? 'naive';

    // 1. Fetch real time series
    const timeSeries = await IncidentTimeSeriesAggregator.aggregate({
      region,
      frequency,
      canonicalOnly: true,
    });

    const totalObservedBuckets = timeSeries.buckets.length;
    const nonZeroBuckets = timeSeries.buckets.filter((b) => b.incidentCount > 0).length;

    // Calculate temporal span in hours and days
    let temporalSpanHours = 0;
    let temporalSpanDays = 0;

    if (timeSeries.windowStart && timeSeries.windowEnd) {
      const startMs = new Date(timeSeries.windowStart).getTime();
      const endMs = new Date(timeSeries.windowEnd).getTime();
      const spanMs = Math.max(0, endMs - startMs);
      temporalSpanHours = Math.round((spanMs / (1000 * 60 * 60)) * 10) / 10;
      temporalSpanDays = Math.round((spanMs / (1000 * 60 * 60 * 24)) * 10) / 10;
    }

    // Determine seasonal cycles observed
    let seasonalPeriodsObserved = 0;
    let minimumRequiredSeasonalPeriods = 0;

    if (frequency === 'hourly') {
      seasonalPeriodsObserved = Math.floor(temporalSpanHours / 24);
      minimumRequiredSeasonalPeriods = 2; // at least 2 full 24h cycles
    } else if (frequency === 'daily') {
      seasonalPeriodsObserved = Math.floor(temporalSpanDays / 7);
      minimumRequiredSeasonalPeriods = 4; // at least 4 full 7-day cycles
    } else if (frequency === 'monthly') {
      seasonalPeriodsObserved = Math.floor(temporalSpanDays / 365);
      minimumRequiredSeasonalPeriods = 2; // at least 2 full 12-month annual cycles
    }

    const minRequiredBuckets = MINIMUM_BUCKET_THRESHOLDS[modelType]?.[frequency] ?? 30;

    const reasons: string[] = [];
    const recommendations: string[] = [];
    let isLocked = false;
    let status: ReadinessStatus = 'DATA_SUFFICIENT';

    // Enforce data checks
    if (timeSeries.totalIncidents === 0) {
      status = 'DATA_INSUFFICIENT';
      isLocked = true;
      reasons.push(`Zero real incidents found for region ${region}.`);
      recommendations.push('Gather incident submissions before attempting time-series forecasting.');
    } else if (temporalSpanHours < 24 && frequency === 'hourly') {
      status = 'DATA_INSUFFICIENT';
      isLocked = true;
      reasons.push(
        `Operational data window spans only ${temporalSpanHours} hours (single afternoon on 2026-11-07). Minimum 24-48 continuous hours required for hourly forecasting.`
      );
      recommendations.push(
        'Maintain continuous incident logging across multiple consecutive days to capture diurnal reporting variations.'
      );
    } else if (temporalSpanDays < 14 && (frequency === 'daily' || frequency === 'weekly' || frequency === 'monthly')) {
      status = 'DATA_INSUFFICIENT';
      isLocked = true;
      reasons.push(
        `Temporal depth spans only ${temporalSpanHours} hours (${temporalSpanDays} days). Minimum ${minRequiredBuckets} continuous ${frequency} periods required.`
      );
      recommendations.push(
        `Accumulate longitudinal incident data over at least ${minRequiredBuckets} continuous ${frequency} periods before unlocking ${modelType} forecasting.`
      );
    } else if (totalObservedBuckets < minRequiredBuckets) {
      status = 'DATA_INSUFFICIENT';
      isLocked = true;
      reasons.push(
        `Observed buckets (${totalObservedBuckets}) below minimum requirement (${minRequiredBuckets}) for ${modelType} at ${frequency} resolution.`
      );
      recommendations.push(
        `Require at least ${minRequiredBuckets - totalObservedBuckets} more consecutive ${frequency} periods.`
      );
    }

    // Special gate for deep learning (LSTM)
    if (modelType === 'lstm') {
      isLocked = true;
      status = 'DATA_INSUFFICIENT';
      reasons.push(
        'Deep learning recurrent neural networks (BiLSTM) are strictly locked to prevent ungrounded hallucinations on small-sample datasets.'
      );
      recommendations.push(
        'Collect at least 1-2 years of verified municipal incident data across all municipal wards before training neural architectures.'
      );
    }

    return {
      status,
      region,
      frequency,
      modelType,
      temporalSpanHours,
      temporalSpanDays,
      totalObservedBuckets,
      nonZeroBuckets,
      minimumRequiredBuckets: minRequiredBuckets,
      seasonalPeriodsObserved,
      minimumRequiredSeasonalPeriods,
      isLocked,
      reasons,
      recommendations,
      evaluationTimestamp: new Date().toISOString(),
    };
  }
}


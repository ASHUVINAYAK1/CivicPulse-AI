import {
  AggregationFrequency,
  ForecastPoint,
  ModelDescriptor,
  ModelType,
  TimeSeriesBucket,
} from './types';
import { MINIMUM_BUCKET_THRESHOLDS } from './readiness-gate';

/**
 * Model Catalog defining statistical baselines and ML specifications.
 */
export const FORECASTING_MODELS: ModelDescriptor[] = [
  {
    id: 'baseline-naive-v1',
    name: 'Naive Persistence Forecaster',
    modelType: 'naive',
    description: 'Carries forward the most recent observed volume count to all future horizon periods.',
    framework: 'Rule-based / Exact Persistence',
    status: 'ACTIVE',
    statusReason: 'Unconditional baseline for benchmarking any higher-order statistical or machine learning model.',
    minimumBucketsRequired: MINIMUM_BUCKET_THRESHOLDS.naive,
    seasonalPeriodsRequired: 0,
    formulaDescription: 'y_hat[t+h] = y[t]',
    pros: ['Zero hyperparameter tuning', 'Robust to extreme outliers', 'Immediate reference baseline'],
    cons: ['Ignores growth trends', 'Ignores day-of-week and seasonal cycles'],
  },
  {
    id: 'baseline-moving-average-v1',
    name: 'Simple Moving Average (SMA-3)',
    modelType: 'moving_average',
    description: 'Computes unweighted rolling mean of the last w historical observations (default w=3).',
    framework: 'Time Series Statistics',
    status: 'ACTIVE',
    statusReason: 'Standard smoothing filter to dampen reporting noise and temporary collection spikes.',
    minimumBucketsRequired: MINIMUM_BUCKET_THRESHOLDS.moving_average,
    seasonalPeriodsRequired: 0,
    formulaDescription: 'y_hat[t+h] = (1/w) * sum_{i=0}^{w-1} y[t-i]',
    pros: ['Reduces local reporting variance', 'Smooths high-frequency anomalies'],
    cons: ['Lags behind abrupt trend shifts', 'Underpredicts during accelerated monsoon surges'],
  },
  {
    id: 'baseline-seasonal-naive-v1',
    name: 'Seasonal Naive Forecaster',
    modelType: 'seasonal_naive',
    description: 'Projects volume from the corresponding period of the previous observed seasonal cycle.',
    framework: 'Seasonal Statistical Baseline',
    status: 'LOCKED',
    statusReason: 'Requires at least 2 complete seasonal cycles (e.g. 24h for diurnal, 7 days for weekly, 12 months for annual).',
    minimumBucketsRequired: MINIMUM_BUCKET_THRESHOLDS.seasonal_naive,
    seasonalPeriodsRequired: 2,
    formulaDescription: 'y_hat[t+h] = y[t + h - m*s] where s = cycle length',
    pros: ['Captures diurnal commuter peaks and weekend reporting dips accurately'],
    cons: ['Requires extensive historical depth', 'Fails if collection calendar is disrupted'],
  },
  {
    id: 'baseline-linear-trend-v1',
    name: 'Ordinary Least Squares Linear Trend',
    modelType: 'linear_trend',
    description: 'Fits parametric linear regression line against time indices with analytical confidence bounds.',
    framework: 'Parametric Regression',
    status: 'ACTIVE',
    statusReason: 'Transparent trend quantification with standard analytical error bounds.',
    minimumBucketsRequired: MINIMUM_BUCKET_THRESHOLDS.linear_trend,
    seasonalPeriodsRequired: 0,
    formulaDescription: 'y_hat[t] = alpha + beta * t',
    pros: ['Quantifies rate of growth or decline', 'Closed-form exact solution with prediction intervals'],
    cons: ['Assumes linear trajectory unbounded indefinitely', 'Sensitive to early outlier leverage points'],
  },
  {
    id: 'scaffold-arima-v1',
    name: 'Auto-Regressive Integrated Moving Average (ARIMA)',
    modelType: 'arima_scaffold',
    description: 'Parametric Box-Jenkins time series framework supporting differencing and auto-regressive lags.',
    framework: 'Box-Jenkins Time Series',
    status: 'LOCKED',
    statusReason: 'Requires stationary differencing tests and sufficient longitudinal depth (>45 daily buckets).',
    minimumBucketsRequired: MINIMUM_BUCKET_THRESHOLDS.arima_scaffold,
    seasonalPeriodsRequired: 0,
    formulaDescription: 'Phi(B)(1-B)^d y[t] = Theta(B) epsilon[t]',
    pros: ['Models autocorrelated disturbance terms', 'Established econometrics standard'],
    cons: ['Requires strict stationarity', 'Sensitive to structural breaks'],
  },
  {
    id: 'neural-forecaster-lstm-v1',
    name: 'Bidirectional LSTM Neural Forecaster',
    modelType: 'lstm',
    description: 'Deep recurrent neural network with gating units for multi-horizon sequence modeling.',
    framework: 'PyTorch / BiLSTM',
    status: 'LOCKED',
    statusReason: 'Strictly locked due to insufficient training samples. Deep architectures severely overfit on short series.',
    minimumBucketsRequired: MINIMUM_BUCKET_THRESHOLDS.lstm,
    seasonalPeriodsRequired: 2,
    formulaDescription: 'h[t] = LSTM(x[t], h[t-1]); y_hat = Linear(h[t])',
    pros: ['Captures complex non-linear interactions across weather and spatial covariates'],
    cons: ['Requires massive training history', 'High risk of severe hallucination on small sample sizes'],
  },
];

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

/**
 * Format bucket period label for clean human readability.
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

export class ForecastingEngine {
  /**
   * Generates time-series forecast points using the selected baseline model.
   */
  public static forecast(
    buckets: TimeSeriesBucket[],
    frequency: AggregationFrequency,
    modelType: ModelType,
    horizon: number = 6
  ): ForecastPoint[] {
    if (buckets.length === 0) {
      return [];
    }

    const series = buckets.map((b) => b.incidentCount);
    const n = series.length;
    const lastBucketDate = new Date(buckets[n - 1].timestamp);

    const points: ForecastPoint[] = [];
    let currentDate = new Date(lastBucketDate.getTime());

    switch (modelType) {
      case 'naive': {
        const lastVal = series[n - 1];
        // Sample standard error of differences
        let diffSumSq = 0;
        for (let i = 1; i < n; i++) {
          diffSumSq += Math.pow(series[i] - series[i - 1], 2);
        }
        const sigma = n > 1 ? Math.sqrt(diffSumSq / (n - 1)) : Math.max(1, lastVal * 0.2);

        for (let h = 1; h <= horizon; h++) {
          currentDate = stepBucket(currentDate, frequency);
          const margin = 1.96 * sigma * Math.sqrt(h);
          points.push({
            period: currentDate.toISOString(),
            periodLabel: formatBucketLabel(currentDate, frequency),
            predictedIncidents: Math.round(lastVal * 10) / 10,
            lowerBound: Math.max(0, Math.round((lastVal - margin) * 10) / 10),
            upperBound: Math.round((lastVal + margin) * 10) / 10,
            confidenceInterval: 0.95,
          });
        }
        break;
      }

      case 'moving_average': {
        const windowSize = Math.min(3, n);
        const window = series.slice(n - windowSize);
        const mean = window.reduce((a, b) => a + b, 0) / windowSize;

        // Variance of series
        let sumSq = 0;
        for (let i = 0; i < n; i++) {
          sumSq += Math.pow(series[i] - mean, 2);
        }
        const s = n > 1 ? Math.sqrt(sumSq / (n - 1)) : Math.max(1, mean * 0.2);

        for (let h = 1; h <= horizon; h++) {
          currentDate = stepBucket(currentDate, frequency);
          const margin = 1.96 * s * Math.sqrt(1 + 1 / windowSize);
          points.push({
            period: currentDate.toISOString(),
            periodLabel: formatBucketLabel(currentDate, frequency),
            predictedIncidents: Math.round(mean * 10) / 10,
            lowerBound: Math.max(0, Math.round((mean - margin) * 10) / 10),
            upperBound: Math.round((mean + margin) * 10) / 10,
            confidenceInterval: 0.95,
          });
        }
        break;
      }

      case 'seasonal_naive': {
        const seasonLength = frequency === 'hourly' ? 24 : frequency === 'daily' ? 7 : 12;
        const fallbackVal = series[n - 1];

        for (let h = 1; h <= horizon; h++) {
          currentDate = stepBucket(currentDate, frequency);
          let predictedVal = fallbackVal;
          if (n >= seasonLength) {
            const seasonIdx = (n - seasonLength + ((h - 1) % seasonLength)) % n;
            predictedVal = series[seasonIdx];
          }
          const margin = Math.max(1, predictedVal * 0.25);
          points.push({
            period: currentDate.toISOString(),
            periodLabel: formatBucketLabel(currentDate, frequency),
            predictedIncidents: Math.round(predictedVal * 10) / 10,
            lowerBound: Math.max(0, Math.round((predictedVal - margin) * 10) / 10),
            upperBound: Math.round((predictedVal + margin) * 10) / 10,
            confidenceInterval: 0.95,
          });
        }
        break;
      }

      case 'linear_trend': {
        // Fit OLS
        const xMean = (n - 1) / 2;
        const yMean = series.reduce((a, b) => a + b, 0) / n;

        let num = 0;
        let den = 0;
        for (let i = 0; i < n; i++) {
          num += (i - xMean) * (series[i] - yMean);
          den += Math.pow(i - xMean, 2);
        }
        const beta = den > 0 ? num / den : 0;
        const alpha = yMean - beta * xMean;

        // Residual standard error
        let resSumSq = 0;
        for (let i = 0; i < n; i++) {
          const fitted = alpha + beta * i;
          resSumSq += Math.pow(series[i] - fitted, 2);
        }
        const sRes = n > 2 ? Math.sqrt(resSumSq / (n - 2)) : 1.0;

        for (let h = 1; h <= horizon; h++) {
          currentDate = stepBucket(currentDate, frequency);
          const xNew = n - 1 + h;
          const predicted = Math.max(0, alpha + beta * xNew);
          const sePred = den > 0 ? sRes * Math.sqrt(1 + 1 / n + Math.pow(xNew - xMean, 2) / den) : sRes;
          const margin = 1.96 * sePred;

          points.push({
            period: currentDate.toISOString(),
            periodLabel: formatBucketLabel(currentDate, frequency),
            predictedIncidents: Math.round(predicted * 10) / 10,
            lowerBound: Math.max(0, Math.round((predicted - margin) * 10) / 10),
            upperBound: Math.round((predicted + margin) * 10) / 10,
            confidenceInterval: 0.95,
          });
        }
        break;
      }

      default: {
        // Fallback naive
        const lastVal = series[n - 1] ?? 0;
        for (let h = 1; h <= horizon; h++) {
          currentDate = stepBucket(currentDate, frequency);
          points.push({
            period: currentDate.toISOString(),
            periodLabel: formatBucketLabel(currentDate, frequency),
            predictedIncidents: Math.round(lastVal * 10) / 10,
            lowerBound: Math.max(0, Math.round(lastVal * 0.8 * 10) / 10),
            upperBound: Math.round(lastVal * 1.2 * 10) / 10,
            confidenceInterval: 0.95,
          });
        }
      }
    }

    return points;
  }

  /**
   * Evaluates historical accuracy chronologically using expanding window.
   * Never leaks future data into the past.
   */
  public static evaluateChronological(
    buckets: TimeSeriesBucket[],
    modelType: ModelType
  ): { mae: number; rmse: number; mape: number } {
    const series = buckets.map((b) => b.incidentCount);
    const n = series.length;

    if (n < 4) {
      return { mae: 0, rmse: 0, mape: 0 };
    }

    const trainSplit = Math.max(2, Math.floor(n * 0.6));
    const errors: number[] = [];
    const absErrors: number[] = [];
    const pctErrors: number[] = [];

    for (let t = trainSplit; t < n; t++) {
      const trainSeries = buckets.slice(0, t);
      const actual = series[t];
      const preds = this.forecast(trainSeries, 'hourly', modelType, 1);
      const predicted = preds[0]?.predictedIncidents ?? series[t - 1];

      const err = actual - predicted;
      errors.push(err);
      absErrors.push(Math.abs(err));
      pctErrors.push(Math.abs(err) / Math.max(1, actual));
    }

    const mae = absErrors.reduce((a, b) => a + b, 0) / absErrors.length;
    const rmse = Math.sqrt(errors.reduce((a, b) => a + b * b, 0) / errors.length);
    const mape = (pctErrors.reduce((a, b) => a + b, 0) / pctErrors.length) * 100;

    return {
      mae: Math.round(mae * 100) / 100,
      rmse: Math.round(rmse * 100) / 100,
      mape: Math.round(mape * 10) / 10,
    };
  }
}

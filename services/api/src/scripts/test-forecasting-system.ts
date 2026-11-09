import http from 'http';
import { createApp } from '../app';
import { prisma } from '../config/prisma';
import { ForecastingService } from '../modules/forecasting/service';
import { IncidentTimeSeriesAggregator } from '../modules/forecasting/aggregator';
import { ForecastingReadinessGate } from '../modules/forecasting/readiness-gate';
import { ForecastingEngine } from '../modules/forecasting/models';
import { TimeSeriesBucket } from '../modules/forecasting/types';

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string) {
  if (condition) {
    console.log(`  âœ“ PASS: ${message}`);
    passed++;
  } else {
    console.error(`  âœ— FAIL: ${message}`);
    failed++;
  }
}

async function request(server: http.Server, method: string, path: string, body?: any): Promise<{ status: number; body: any }> {
  return new Promise((resolve, reject) => {
    const address = server.address() as any;
    const req = http.request(
      {
        hostname: '127.0.0.1',
        port: address.port,
        path,
        method,
        headers: {
          'Content-Type': 'application/json',
        },
      },
      (res) => {
        let data = '';
        res.on('data', (chunk) => (data += chunk));
        res.on('end', () => {
          try {
            resolve({ status: res.statusCode || 500, body: JSON.parse(data) });
          } catch {
            resolve({ status: res.statusCode || 500, body: data });
          }
        });
      }
    );
    req.on('error', reject);
    if (body) {
      req.write(JSON.stringify(body));
    }
    req.end();
  });
}

async function runTests() {
  console.log('\n=============================================================');
  console.log('CivicPulse AI - Phase 3 Stage 6: Incident Volume Forecasting');
  console.log('Automated Verification and Governance Gate Test Suite');
  console.log('=============================================================\n');

  // Test Group 1: Dataset Summary & Audit
  console.log('--- Test Group 1: Data Audit & Regional Census ---');
  const summary = await ForecastingService.getSummary();
  assert(summary.datasetOverview.totalRealIncidents === 66, 'Total real incidents in database equals 66');
  assert(summary.datasetOverview.bengaluruCount === 58, 'Bengaluru Metropolitan Area census equals 58');
  assert(summary.datasetOverview.delhiCount === 8, 'Delhi NCR census equals 8');
  assert(summary.datasetOverview.unassignedCount === 0, 'Zero unassigned/out-of-boundary incidents');
  assert(summary.datasetOverview.longitudinalSpanHours > 5.0 && summary.datasetOverview.longitudinalSpanHours < 6.5,
    `Observation span is ~5.8 hours (${summary.datasetOverview.longitudinalSpanHours}h)`);
  assert(summary.syntheticSeedAudit.totalSyntheticForecasts === 6, 'Total synthetic seed forecasts flagged equals 6');
  assert(summary.syntheticSeedAudit.status === 'FLAGGED_SYNTHETIC', 'Seed forecasts explicitly marked FLAGGED_SYNTHETIC');
  assert(summary.governanceStatus.overallReadiness === 'DATA_INSUFFICIENT', 'Overall readiness is DATA_INSUFFICIENT');
  assert(summary.governanceStatus.neuralForecastingLocked === true, 'Neural forecasting (BiLSTM) is locked');

  // Test Group 2: Time-Series Aggregation Pipeline
  console.log('\n--- Test Group 2: Time-Series Aggregation Pipeline ---');
  const hourlyAll = await IncidentTimeSeriesAggregator.aggregate({ frequency: 'hourly', region: 'ALL' });
  assert(hourlyAll.totalIncidents === 66, 'Hourly ALL aggregation includes all 66 incidents');
  assert(hourlyAll.buckets.length >= 6, `Hourly ALL created ${hourlyAll.buckets.length} continuous buckets`);
  assert(hourlyAll.buckets.some((b) => b.incidentCount > 0), 'Hourly ALL has non-zero incident buckets');
  
  const hourlyBLR = await IncidentTimeSeriesAggregator.aggregate({ frequency: 'hourly', region: 'BENGALURU_METRO' });
  assert(hourlyBLR.totalIncidents === 58, 'Hourly BENGALURU_METRO aggregation totals exactly 58 incidents');
  
  const hourlyDEL = await IncidentTimeSeriesAggregator.aggregate({ frequency: 'hourly', region: 'DELHI_NCR' });
  assert(hourlyDEL.totalIncidents === 8, 'Hourly DELHI_NCR aggregation totals exactly 8 incidents');
  assert(hourlyBLR.totalIncidents + hourlyDEL.totalIncidents === 66, 'BLR (58) + DEL (8) partitions the 66 incidents completely');

  const dailyAll = await IncidentTimeSeriesAggregator.aggregate({ frequency: 'daily', region: 'ALL' });
  assert(dailyAll.totalIncidents === 66, 'Daily ALL totals 66 incidents');
  assert(dailyAll.buckets.length === 1, 'Daily aggregation produces exactly 1 bucket for single afternoon dataset');

  const monthlyAll = await IncidentTimeSeriesAggregator.aggregate({ frequency: 'monthly', region: 'ALL' });
  assert(monthlyAll.totalIncidents === 66, 'Monthly ALL totals 66 incidents');
  assert(monthlyAll.buckets.length === 1, 'Monthly aggregation produces exactly 1 bucket');

  // Test Group 2b: Data Quality Observation Status Audit (Stage 7 Enhancement)
  console.log('\n--- Test Group 2b: Observation Status & Outage Audit ---');
  assert(hourlyAll.buckets.some((b) => b.observationStatus === 'OBSERVED_NONZERO'), 'Detected OBSERVED_NONZERO buckets with incidents');
  assert(hourlyAll.buckets.every((b) => b.displayTimestampLocal !== undefined), 'Canonical displayTimestampLocal populated for all buckets');

  // Test custom coverage and outage windows
  const customAgg = await IncidentTimeSeriesAggregator.aggregate({
    frequency: 'hourly',
    region: 'ALL',
    coverageWindows: [
      { start: new Date('2026-11-07T11:00:00Z'), end: new Date('2026-11-07T14:00:00Z'), isCoverageActive: true },
    ],
    outageWindows: [
      { start: new Date('2026-11-07T14:00:00Z'), end: new Date('2026-11-07T15:00:00Z'), reason: 'Server Maintenance Window' },
    ],
  });

  const outageBucket = customAgg.buckets.find((b) => b.observationStatus === 'COLLECTION_OUTAGE');
  assert(Boolean(outageBucket), 'Correctly identified COLLECTION_OUTAGE bucket');
  assert(outageBucket?.outageReason === 'Server Maintenance Window', 'Outage reason preserved in bucket');

  const unobservedBucket = customAgg.buckets.find((b) => b.observationStatus === 'UNOBSERVED' && b.incidentCount === 0);
  assert(Boolean(unobservedBucket), 'Correctly classified zero-incident bucket outside coverage as UNOBSERVED');

  // Test Group 3: Data Readiness Gates
  console.log('\n--- Test Group 3: Data Readiness Gate Verification ---');
  const readinessNaive = await ForecastingReadinessGate.evaluateReadiness({ frequency: 'hourly', modelType: 'naive' });
  assert(readinessNaive.status === 'DATA_INSUFFICIENT', 'Naive hourly gate returns DATA_INSUFFICIENT');
  assert(readinessNaive.isLocked === true, 'Naive hourly gate is locked');
  assert(readinessNaive.reasons.length > 0, 'Explains insufficient observation window (5.8h < 24h)');

  const readinessMA = await ForecastingReadinessGate.evaluateReadiness({ frequency: 'daily', modelType: 'moving_average' });
  assert(readinessMA.status === 'DATA_INSUFFICIENT', 'Moving Average daily gate returns DATA_INSUFFICIENT');
  assert(readinessMA.reasons.some((r) => r.includes('Minimum 30')), 'Rejection cites minimum 30 periods required');

  const readinessLSTM = await ForecastingReadinessGate.evaluateReadiness({ frequency: 'monthly', modelType: 'lstm' });
  assert(readinessLSTM.status === 'DATA_INSUFFICIENT', 'BiLSTM neural forecaster returns DATA_INSUFFICIENT');
  assert(readinessLSTM.reasons.some((r) => r.includes('strictly locked')), 'Cites strict lock against overfitting/hallucinations');

  // Test Group 4: Forecasting Baselines & Mathematical Scaffolding
  console.log('\n--- Test Group 4: Forecasting Models on Synthetic Time-Series ---');
  // Construct a synthetic 10-point test series to verify mathematical formulas
  const mockBuckets: TimeSeriesBucket[] = [
    { bucketIndex: 0, timestamp: '2026-11-01T00:00:00Z', periodLabel: '2026-11-01', incidentCount: 10, verifiedCount: 5, criticalCount: 1, damageBreakdown: {}, isZero: false, isObserved: true, observationStatus: 'OBSERVED_NONZERO' },
    { bucketIndex: 1, timestamp: '2026-11-02T00:00:00Z', periodLabel: '2026-11-02', incidentCount: 12, verifiedCount: 6, criticalCount: 2, damageBreakdown: {}, isZero: false, isObserved: true, observationStatus: 'OBSERVED_NONZERO' },
    { bucketIndex: 2, timestamp: '2026-11-03T00:00:00Z', periodLabel: '2026-11-03', incidentCount: 14, verifiedCount: 7, criticalCount: 1, damageBreakdown: {}, isZero: false, isObserved: true, observationStatus: 'OBSERVED_NONZERO' },
    { bucketIndex: 3, timestamp: '2026-11-04T00:00:00Z', periodLabel: '2026-11-04', incidentCount: 16, verifiedCount: 8, criticalCount: 3, damageBreakdown: {}, isZero: false, isObserved: true, observationStatus: 'OBSERVED_NONZERO' },
    { bucketIndex: 4, timestamp: '2026-11-05T00:00:00Z', periodLabel: '2026-11-05', incidentCount: 18, verifiedCount: 9, criticalCount: 2, damageBreakdown: {}, isZero: false, isObserved: true, observationStatus: 'OBSERVED_NONZERO' },
    { bucketIndex: 5, timestamp: '2026-11-06T00:00:00Z', periodLabel: '2026-11-06', incidentCount: 20, verifiedCount: 10, criticalCount: 4, damageBreakdown: {}, isZero: false, isObserved: true, observationStatus: 'OBSERVED_NONZERO' },
  ];

  const naivePreds = ForecastingEngine.forecast(mockBuckets, 'daily', 'naive', 3);
  assert(naivePreds.length === 3, 'Naive generates 3 forecast points');
  assert(naivePreds[0].predictedIncidents === 20, 'Naive predicts exact last observed value (20)');
  assert(naivePreds[0].lowerBound <= 20 && naivePreds[0].upperBound >= 20, 'Naive prediction interval contains point forecast');

  const maPreds = ForecastingEngine.forecast(mockBuckets, 'daily', 'moving_average', 3);
  assert(maPreds.length === 3, 'Moving Average generates 3 forecast points');
  assert(maPreds[0].predictedIncidents === 18, 'Moving Average (SMA-3) of [16, 18, 20] equals 18');

  const trendPreds = ForecastingEngine.forecast(mockBuckets, 'daily', 'linear_trend', 3);
  assert(trendPreds.length === 3, 'Linear Trend generates 3 forecast points');
  assert(trendPreds[0].predictedIncidents === 22, 'Linear Trend slope=+2 predicts 22 at t=6');
  assert(trendPreds[1].predictedIncidents === 24, 'Linear Trend predicts 24 at t=7');

  const chronoMetrics = ForecastingEngine.evaluateChronological(mockBuckets, 'naive');
  assert(chronoMetrics.mae > 0, `Chronological expanding-window evaluation calculated MAE: ${chronoMetrics.mae}`);

  // Test Group 5: HTTP Endpoints Integration
  console.log('\n--- Test Group 5: HTTP REST API Endpoints ---');
  const app = createApp();
  const server = http.createServer(app);

  await new Promise<void>((resolve) => server.listen(0, resolve));

  try {
    // 1. GET /api/v1/forecasts/summary
    const resSummary = await request(server, 'GET', '/api/v1/forecasts/summary');
    assert(resSummary.status === 200, 'GET /api/v1/forecasts/summary returns HTTP 200');
    assert(resSummary.body.data.datasetOverview.totalRealIncidents === 66, 'API summary reports 66 real incidents');

    // 2. GET /api/v1/forecasts/history
    const resHistoryBLR = await request(server, 'GET', '/api/v1/forecasts/history?frequency=hourly&region=BENGALURU_METRO');
    assert(resHistoryBLR.status === 200, 'GET /api/v1/forecasts/history for BLR returns HTTP 200');
    assert(resHistoryBLR.body.data.totalIncidents === 58, 'API history for BLR returns 58 incidents');

    const resHistoryDEL = await request(server, 'GET', '/api/v1/forecasts/history?frequency=hourly&region=DELHI_NCR');
    assert(resHistoryDEL.status === 200, 'GET /api/v1/forecasts/history for DEL returns HTTP 200');
    assert(resHistoryDEL.body.data.totalIncidents === 8, 'API history for DEL returns 8 incidents');

    // 3. GET /api/v1/forecasts/readiness
    const resReadiness = await request(server, 'GET', '/api/v1/forecasts/readiness?frequency=daily&modelType=moving_average');
    assert(resReadiness.status === 200, 'GET /api/v1/forecasts/readiness returns HTTP 200');
    assert(resReadiness.body.data.status === 'DATA_INSUFFICIENT', 'API readiness reports DATA_INSUFFICIENT');

    // 4. GET /api/v1/forecasts/models
    const resModels = await request(server, 'GET', '/api/v1/forecasts/models');
    assert(resModels.status === 200, 'GET /api/v1/forecasts/models returns HTTP 200');
    assert(Array.isArray(resModels.body.data) && resModels.body.data.length === 6, 'API models returns 6 catalog models');

    // 5. POST /api/v1/forecasts/generate (unforced -> should fail with 422 DATA_INSUFFICIENT)
    const resGenBlocked = await request(server, 'POST', '/api/v1/forecasts/generate', {
      region: 'BENGALURU_METRO',
      frequency: 'daily',
      modelType: 'moving_average',
      horizon: 6,
    });
    assert(resGenBlocked.status === 422, 'POST /api/v1/forecasts/generate rejects ungrounded generation with HTTP 422');
    assert(resGenBlocked.body.readiness.status === 'DATA_INSUFFICIENT', '422 payload includes detailed readiness audit');

    // 6. POST /api/v1/forecasts/generate (forced simulation -> should succeed with provenance)
    const resGenSim = await request(server, 'POST', '/api/v1/forecasts/generate', {
      region: 'BENGALURU_METRO',
      frequency: 'hourly',
      modelType: 'naive',
      horizon: 4,
      forceSimulate: true,
    });
    assert(resGenSim.status === 200, 'POST /api/v1/forecasts/generate with forceSimulate returns HTTP 200');
    assert(resGenSim.body.data.isSynthetic === true, 'Simulation forecast result is explicitly marked isSynthetic: true');
    assert(resGenSim.body.data.predictions.length === 4, 'Simulation returned 4 predicted points');

  } finally {
    server.close();
  }

  console.log('\n=============================================================');
  console.log(`Verification Complete: ${passed} Passed, ${failed} Failed`);
  console.log('=============================================================\n');

  if (failed > 0) {
    process.exit(1);
  }
}

runTests()
  .catch((err) => {
    console.error('Test script crashed:', err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });


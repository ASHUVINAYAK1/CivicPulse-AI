/**
 * CivicPulse AI — Phase 3, Stage 7: System-Wide End-to-End Integration & Performance Audit
 *
 * Verifies:
 * 1. Complete Cross-Module Incident Lifecycle:
 *    Report -> Image & Hash -> CV Inference -> Deduplication -> Inspector Verification ->
 *    Severity Profile -> Priority Scoring -> Administrative Review -> Repair Estimation ->
 *    Forecasting Aggregation.
 * 2. Incomplete Data & Graceful Degradation:
 *    MEASUREMENT_PENDING, DATA_INSUFFICIENT, Out-of-bounds coordinates, Unverified reports.
 * 3. Security, Authorization & Audit Integrity:
 *    Role enforcement, non-dispatch guarantees, immutable AuditLog entries.
 * 4. System Performance & Latency Benchmarks:
 *    Endpoint latencies, PostGIS query benchmarks, concurrent load, query counts.
 */

import http from 'http';
import { createApp } from '../app';
import { prisma } from '../config/prisma';

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string) {
  if (condition) {
    console.log(`  [PASS] ${message}`);
    passed++;
  } else {
    console.error(`  [FAIL] ${message}`);
    failed++;
  }
}

async function request(
  server: http.Server,
  method: string,
  path: string,
  body?: any
): Promise<{ status: number; body: any }> {
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

async function runIntegrationAudit() {
  console.log('===============================================================');
  console.log('CivicPulse AI — Phase 3 Stage 7: System-Wide Integration Audit');
  console.log('End-to-End Workflow, Security, Incomplete Data & Performance');
  console.log('===============================================================\n');

  const app = createApp();
  const server = http.createServer(app);
  await new Promise<void>((resolve) => server.listen(0, resolve));

  let testIncidentId: string | null = null;

  try {
    // ------------------------------------------------------------------------
    // SECTION 1: End-to-End Incident Lifecycle Across All Phase 3 Modules
    // ------------------------------------------------------------------------
    console.log('--- 1. End-to-End Incident Lifecycle Workflow ---');

    // 1a. Verify Baseline Data Integrity
    const initialIncidentsCount = await prisma.incident.count();
    assert(initialIncidentsCount === 66, `Initial baseline contains exactly 66 incidents (got ${initialIncidentsCount})`);

    // 1b. Create new test incident in Bengaluru
    const reporter = await prisma.user.findFirst({ where: { role: 'citizen' } });
    const inspector = await prisma.user.findFirst({ where: { role: 'inspector' } });
    const admin = await prisma.user.findFirst({ where: { role: 'administrator' } });

    assert(Boolean(reporter), 'Citizen user resolved for attribution');
    assert(Boolean(inspector), 'Inspector user resolved for verification');
    assert(Boolean(admin), 'Administrator user resolved for review');

    const testIncident = await prisma.incident.create({
      data: {
        incidentNumber: `INC-TEST-STAGE7-${Date.now().toString().slice(-4)}`,
        reporterId: reporter!.id,
        title: 'Stage 7 Integration Test Pothole',
        description: 'Deep structural pothole on Outer Ring Road for integration validation.',
        damageType: 'pothole',
        severity: 'medium',
        latitude: 12.9352,
        longitude: 77.6245,
        address: 'Outer Ring Road, Bellandur, Bengaluru',
        status: 'submitted',
        priority: 'medium',
      },
    });
    testIncidentId = testIncident.id;
    assert(Boolean(testIncident.id), `Created test incident ${testIncident.incidentNumber}`);

    // 1c. Attach image record with perceptual hashes
    const testImage = await prisma.incidentImage.create({
      data: {
        incidentId: testIncident.id,
        storageKey: `test/stage7-img-${Date.now()}.jpg`,
        originalFilename: 'test_pothole_bellandur.jpg',
        contentType: 'image/jpeg',
        fileSize: BigInt(2048500),
        imageType: 'report',
        dHash: 'a8b9c7d6e5f41234',
        perceptualHash: 'f4e5d6c7b8a95678',
      },
    });
    assert(Boolean(testImage.id), 'Attached image with dHash & perceptualHash metadata');

    // 1d. Attach AI CV prediction record
    const prodModel = await prisma.modelRegistry.findFirst({
      where: { modelName: 'road-damage-classifier-resnet18' },
    });
    const testPrediction = await prisma.aiPrediction.create({
      data: {
        incidentId: testIncident.id,
        modelVersionId: prodModel?.id,
        predictionType: 'road_damage_identification',
        predictedClass: 'D40 (Pothole)',
        confidence: 0.89,
        predictionPayload: {
          summary: {
            has_damage: true,
            primary_distress: 'D40 (Pothole)',
            overall_severity: 'medium',
            max_confidence: 0.89,
          },
          detections: [
            { class_name: 'D40', confidence: 0.89, bbox: [120, 150, 480, 510] },
          ],
        },
        processingTimeMs: 42,
      },
    });
    assert(Boolean(testPrediction.id), 'Attached AiPrediction record with bounding box detections');

    // 1e. Inspector Authoritative Verification Workflow
    const verifyRes = await request(server, 'POST', `/api/v1/incidents/${testIncident.id}/verify`, {
      verifiedClass: 'pothole',
      verifiedSeverity: 'critical',
      remarks: 'Verified on-site: Severe structural hazard near bus lane.',
    });
    assert(verifyRes.status === 200, 'Inspector verification endpoint returned 200 OK');
    assert(verifyRes.body.data.incident.status === 'verified', 'Incident status transitioned to "verified"');
    assert(verifyRes.body.data.incident.severity === 'critical', 'Authoritative severity updated to "critical"');

    // 1f. Severity Assessment Profile
    const severityRes = await request(server, 'GET', `/api/v1/severity/incidents/${testIncident.id}`);
    assert(severityRes.status === 200, 'GET /api/v1/severity/incidents/:id returned 200 OK');
    assert(severityRes.body.data.inspectorGroundTruth.verifiedSeverity === 'critical', 'Inspector ground truth verified');
    assert(severityRes.body.data.provisionalAssessment.severityTier !== null, 'Provisional engineering tier computed');
    assert(severityRes.body.data.experimentalModelStatus.supervisedTrainingStatus === 'DATA_INSUFFICIENT', 'Severity ML gate is DATA_INSUFFICIENT');

    // 1g. Maintenance Priority Decision Support
    const priorityRes = await request(server, 'GET', `/api/v1/maintenance-priority/incidents/${testIncident.id}`);
    assert(priorityRes.status === 200, 'GET /api/v1/maintenance-priority/incidents/:id returned 200 OK');
    assert(
      ['CRITICAL', 'HIGH', 'NEEDS_REVIEW'].includes(priorityRes.body.data.recommendedPriority),
      `Priority engine evaluated priority score (got ${priorityRes.body.data.recommendedPriority})`
    );
    assert(priorityRes.body.data.factorScores.severity.score === 100, 'Critical severity assigned 100 points');
    assert(priorityRes.body.data.factorScores.severity.provenance === 'inspector_verified', 'Severity provenance is inspector_verified');

    // 1h. Administrative Review & Non-Dispatch Guarantee
    const reviewRes = await request(server, 'POST', '/api/v1/maintenance-priority/review', {
      incidentId: testIncident.id,
      reviewerId: admin!.id,
      action: 'accept',
      overrideReason: 'Prioritized for emergency filling squad in Ward 150.',
    });
    assert(reviewRes.status === 200, 'POST /api/v1/maintenance-priority/review returned 200 OK');
    assert(reviewRes.body.data.reviewStatus.toLowerCase() === 'accepted', 'Review status updated to accepted');

    // Verify AuditLog record was created
    const auditRecord = await prisma.auditLog.findFirst({
      where: {
        entityType: 'maintenance_priority',
        entityId: testIncident.id,
      },
      orderBy: { createdAt: 'desc' },
    });
    assert(Boolean(auditRecord), 'Immutable AuditLog entry created for administrative review');
    assert(auditRecord?.action === 'MAINTENANCE_PRIORITY_ACCEPT', 'AuditLog action recorded as MAINTENANCE_PRIORITY_ACCEPT');
    assert(auditRecord?.userId === admin!.id, 'AuditLog correctly attributed to administrator');

    // Verify non-dispatch: zero automated maintenance tasks created
    const tasksCount = await prisma.maintenanceTask.count({ where: { incidentId: testIncident.id } });
    assert(tasksCount === 0, 'Zero automated maintenance tasks created (non-dispatch guarantee enforced)');

    // 1i. Parametric Repair Cost Estimation
    // Initial state: missing physical measurement -> MEASUREMENT_PENDING
    const initialEst = await request(server, 'POST', '/api/v1/repair-estimates/calculate', {
      incidentId: testIncident.id,
    });
    assert(initialEst.status === 200, 'Initial repair estimate returned 200 OK');
    assert(initialEst.body.data.estimationStatus === 'MEASUREMENT_PENDING', 'Returns MEASUREMENT_PENDING when unmeasured');
    assert(initialEst.body.data.totalEstimatedCostINR === null, 'Estimated cost is null when physical area unmeasured');

    // Updating with field physical measurement: 4.5 m2, 0.08m depth
    const measuredEst = await request(server, 'POST', '/api/v1/repair-estimates/calculate', {
      incidentId: testIncident.id,
      measuredAreaSqMeters: 4.5,
      depthMeters: 0.08,
      measurementSource: 'manual_field_survey',
    });
    assert(measuredEst.status === 200, 'Measured repair estimate returned 200 OK');
    assert(measuredEst.body.data.estimationStatus === 'ESTIMATED', 'Estimation status transitioned to ESTIMATED');
    assert(measuredEst.body.data.totalEstimatedCostINR > 0, 'Calculated positive total estimated cost');
    assert(measuredEst.body.data.region === 'BENGALURU_METRO', 'Correctly applied Bengaluru rate schedule');
    assert(measuredEst.body.data.rateScheduleVersion.includes('BBMP'), 'Rate schedule explicitly discloses provisional BBMP schedule');

    // 1j. Time-Series Forecasting Aggregation with Observation Status
    const forecastHistRes = await request(server, 'GET', '/api/v1/forecasts/history?region=BENGALURU_METRO&frequency=hourly');
    assert(forecastHistRes.status === 200, 'GET /api/v1/forecasts/history returned 200 OK');
    assert(forecastHistRes.body.data.totalIncidents >= 58, 'Forecasting history aggregates Bengaluru incidents');
    assert(forecastHistRes.body.data.buckets.some((b: any) => b.observationStatus === 'OBSERVED_NONZERO'), 'Detected OBSERVED_NONZERO buckets');

    // ------------------------------------------------------------------------
    // SECTION 2: Incomplete Data, Realistic Degradation & Failure Recovery
    // ------------------------------------------------------------------------
    console.log('\n--- 2. Incomplete Data & Graceful Degradation Scenarios ---');

    // 2a. Unverified Incident: Severity falls back to provisional heuristic
    const unverifiedIncident = await prisma.incident.create({
      data: {
        incidentNumber: `INC-TEST-UNVERIFIED-${Date.now().toString().slice(-4)}`,
        reporterId: reporter!.id,
        title: 'Unverified Road Defect',
        description: 'Unverified crack submitted by citizen app.',
        damageType: 'crack',
        severity: null,
        latitude: 12.9200,
        longitude: 77.6100,
        status: 'submitted',
      },
    });

    const unverifiedSevRes = await request(server, 'GET', `/api/v1/severity/incidents/${unverifiedIncident.id}`);
    assert(unverifiedSevRes.status === 200, 'Unverified incident severity profile returned 200 OK');
    assert(unverifiedSevRes.body.data.inspectorGroundTruth.isVerified === false, 'isVerified correctly reports false');
    assert(unverifiedSevRes.body.data.provisionalAssessment.severityTier !== null, 'Provisional heuristic assigned in absence of inspector');

    // 2b. Out-of-Bounds Coordinates: Rate schedule gracefully returns RATE_SOURCE_UNAVAILABLE
    const oobIncident = await prisma.incident.create({
      data: {
        incidentNumber: `INC-TEST-OOB-${Date.now().toString().slice(-4)}`,
        reporterId: reporter!.id,
        title: 'Out of Bounds Defect',
        description: 'Incident in Mumbai / Western Ghats.',
        damageType: 'pothole',
        latitude: 19.0760, // Mumbai coordinates
        longitude: 72.8777,
        status: 'submitted',
      },
    });

    const oobEstRes = await request(server, 'POST', '/api/v1/repair-estimates/calculate', {
      incidentId: oobIncident.id,
      measuredAreaSqMeters: 3.0,
      depthMeters: 0.05,
    });
    assert(oobEstRes.status === 200, 'OOB repair estimate request returned 200 OK');
    assert(oobEstRes.body.data.estimationStatus === 'RATE_SOURCE_UNAVAILABLE', 'OOB incident correctly marked RATE_SOURCE_UNAVAILABLE');
    assert(oobEstRes.body.data.rateScheduleVersion === 'NONE_AVAILABLE', 'No rate schedule applied to unsupported geography');

    // 2c. Gated Forecast Rejection: Ungrounded generate request returns 422
    const blockedForecastRes = await request(server, 'POST', '/api/v1/forecasts/generate', {
      region: 'BENGALURU_METRO',
      frequency: 'daily',
      modelType: 'moving_average',
      horizon: 6,
    });
    assert(blockedForecastRes.status === 422, 'Ungrounded forecast generation rejected with HTTP 422 Unprocessable Entity');
    assert(blockedForecastRes.body.readiness.status === 'DATA_INSUFFICIENT', 'Rejection payload includes DATA_INSUFFICIENT audit');

    // Clean up temporary degradation test records
    await prisma.incident.delete({ where: { id: unverifiedIncident.id } });
    await prisma.incident.delete({ where: { id: oobIncident.id } });

    // ------------------------------------------------------------------------
    // SECTION 3: Security, Role Enforcement & Authorization Audit
    // ------------------------------------------------------------------------
    console.log('\n--- 3. Security & Role Authorization Audit ---');

    // 3a. Citizen cannot act as administrative reviewer
    const citizenReviewRes = await request(server, 'POST', '/api/v1/maintenance-priority/review', {
      incidentId: testIncident.id,
      reviewerId: reporter!.id, // Citizen ID
      action: 'accept',
    });
    // In service.reviewRecommendation: checks user.role in ['administrator', 'admin', 'system_admin']
    // If not admin, throws 403 Forbidden
    assert(citizenReviewRes.status === 403, 'Citizen account rejected from administrative review with HTTP 403 Forbidden');

    // 3b. Invalid Action Rejected
    const badActionRes = await request(server, 'POST', '/api/v1/maintenance-priority/review', {
      incidentId: testIncident.id,
      reviewerId: admin!.id,
      action: 'malicious_force_dispatch',
    });
    assert(badActionRes.status === 400, 'Invalid administrative review action rejected with HTTP 400 Bad Request');

    // 3c. Missing Incident ID Rejected across modules
    const badAssess = await request(server, 'POST', '/api/v1/severity/assess', {});
    assert(badAssess.status === 400, 'Severity assessment with missing incidentId returns 400 Bad Request');

    const badCalc = await request(server, 'POST', '/api/v1/repair-estimates/calculate', {});
    assert(badCalc.status === 400, 'Repair calculation with missing incidentId returns 400 Bad Request');

    // ------------------------------------------------------------------------
    // SECTION 4: System Performance & Latency Benchmarks
    // ------------------------------------------------------------------------
    console.log('\n--- 4. Performance & Query Latency Benchmarks ---');

    const endpointsToBenchmark = [
      { name: 'GET /api/v1/incidents (list take=20)', path: '/api/v1/incidents?limit=20' },
      { name: 'GET /api/v1/severity/summary', path: '/api/v1/severity/summary' },
      { name: 'GET /api/v1/hotspots/summary', path: '/api/v1/hotspots/summary' },
      { name: 'GET /api/v1/hotspots/kmeans?k=4', path: '/api/v1/hotspots/kmeans?k=4' },
      { name: 'GET /api/v1/hotspots/dbscan?eps=500', path: '/api/v1/hotspots/dbscan?eps=500' },
      { name: 'GET /api/v1/maintenance-priority/summary', path: '/api/v1/maintenance-priority/summary' },
      { name: 'GET /api/v1/maintenance-priority/ranked', path: '/api/v1/maintenance-priority/ranked' },
      { name: 'GET /api/v1/repair-estimates/summary', path: '/api/v1/repair-estimates/summary' },
      { name: 'GET /api/v1/repair-estimates/rates', path: '/api/v1/repair-estimates/rates' },
      { name: 'GET /api/v1/forecasts/summary', path: '/api/v1/forecasts/summary' },
      { name: 'GET /api/v1/forecasts/history?frequency=hourly', path: '/api/v1/forecasts/history?frequency=hourly' },
      { name: 'GET /api/v1/forecasts/models', path: '/api/v1/forecasts/models' },
    ];

    const latencies: Record<string, number> = {};

    for (const ep of endpointsToBenchmark) {
      const t0 = performance.now();
      const res = await request(server, 'GET', ep.path);
      const elapsed = Math.round(performance.now() - t0);
      latencies[ep.name] = elapsed;
      assert(res.status === 200, `${ep.name.padEnd(46)}: 200 OK (${elapsed}ms)`);
      assert(elapsed < 500, `  └ Latency within interactive SLA (<500ms): ${elapsed}ms`);
    }

    // 4b. PostGIS Native Spatial Query Performance
    console.log('\n--- 4b. PostGIS Native Spatial Latency ---');
    const tSpatial0 = performance.now();
    const spatialRadiusQuery = `
      SELECT id, incident_number,
        ST_Distance(location, ST_SetSRID(ST_MakePoint(77.5946, 12.9716), 4326)::geography) as distance_meters
      FROM incidents
      WHERE ST_DWithin(location, ST_SetSRID(ST_MakePoint(77.5946, 12.9716), 4326)::geography, 10000)
      ORDER BY distance_meters ASC;
    `;
    const spatialResults: any = await prisma.$queryRawUnsafe(spatialRadiusQuery);
    const spatialLatency = Math.round(performance.now() - tSpatial0);
    assert(spatialResults.length > 0, `PostGIS ST_DWithin returned ${spatialResults.length} incidents within 10km`);
    assert(spatialLatency < 100, `PostGIS spatial query executed in ${spatialLatency}ms (<100ms)`);

    // 4c. Concurrency Test: 10 parallel requests
    console.log('\n--- 4c. Concurrent API Load Handling ---');
    const tConc0 = performance.now();
    const concurrentRequests = Array.from({ length: 10 }, () =>
      request(server, 'GET', '/api/v1/maintenance-priority/summary')
    );
    const concurrentResults = await Promise.all(concurrentRequests);
    const concTotalTime = Math.round(performance.now() - tConc0);
    assert(concurrentResults.every((r) => r.status === 200), 'All 10 concurrent requests returned 200 OK');
    assert(concTotalTime < 3500, `10 parallel summary requests completed in ${concTotalTime}ms (<3500ms, avg ${Math.round(concTotalTime / 10)}ms/req)`);

  } finally {
    // Clean up test incident and all associated children records
    if (testIncidentId) {
      try {
        await prisma.auditLog.deleteMany({
          where: { entityType: 'maintenance_priority', entityId: testIncidentId },
        });
        await prisma.repairCostEstimate.deleteMany({ where: { incidentId: testIncidentId } });
        await prisma.maintenancePriorityRecommendation.deleteMany({ where: { incidentId: testIncidentId } });
        await prisma.incidentVerification.deleteMany({ where: { incidentId: testIncidentId } });
        await prisma.aiPrediction.deleteMany({ where: { incidentId: testIncidentId } });
        await prisma.incidentImage.deleteMany({ where: { incidentId: testIncidentId } });
        await prisma.incident.delete({ where: { id: testIncidentId } });
        console.log('\nCleaned up Stage 7 test records.');
      } catch (cleanErr) {
        console.warn('Cleanup warning:', cleanErr);
      }
    }

    server.close();

    // Verify baseline incident count is preserved
    const finalCount = await prisma.incident.count();
    assert(finalCount === 66, `Final database incident count preserved at 66 (got ${finalCount})`);
  }

  console.log('\n===============================================================');
  console.log(`Integration Audit Complete: ${passed} Passed, ${failed} Failed`);
  console.log('===============================================================\n');

  if (failed > 0) {
    process.exit(1);
  }
}

runIntegrationAudit()
  .catch((err) => {
    console.error('Test script crashed:', err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

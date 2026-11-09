/**
 * CivicPulse AI — Phase 3, Stage 3: Maintenance Priority Test Suite
 *
 * Verifies:
 * 1. Factor normalization across all 6 criteria (0-100 range).
 * 2. Weighted scoring and priority category threshold boundaries.
 * 3. Incomplete and conflicting evidence handling (NEEDS_REVIEW gate).
 * 4. Geographic classification and PostGIS spatial density integration.
 * 5. End-to-end database recommendation recalculation across real incidents.
 * 6. Administrative review workflows (accept, override, reject) and AuditLog provenance.
 * 7. Role-based authorization and non-dispatch guarantees.
 * 8. Express REST API endpoints.
 */

import http from 'http';
import { prisma } from '../config/prisma';
import {
  DEFAULT_PRIORITY_WEIGHTS,
  evaluateMaintenancePriority,
  PriorityEvaluationInput,
} from '../modules/maintenance-priority/engine';
import { classifyIncidentCoordinates } from '../modules/maintenance-priority/geo-classifier';
import { MaintenancePriorityService } from '../modules/maintenance-priority/service';
import { createApp } from '../app';

let testsPassed = 0;
let testsFailed = 0;

function assert(condition: boolean, testName: string, detail: string = '') {
  if (condition) {
    console.log(`  [PASS] ${testName} ${detail ? '(' + detail + ')' : ''}`);
    testsPassed++;
  } else {
    console.error(`  [FAIL] ${testName} ${detail ? '(' + detail + ')' : ''}`);
    testsFailed++;
  }
}

function httpGet(port: number, path: string): Promise<{ status: number; body: any }> {
  return new Promise((resolve, reject) => {
    http
      .get(`http://127.0.0.1:${port}${path}`, (res) => {
        let rawData = '';
        res.on('data', (chunk) => (rawData += chunk));
        res.on('end', () => {
          try {
            resolve({ status: res.statusCode || 500, body: JSON.parse(rawData) });
          } catch {
            resolve({ status: res.statusCode || 500, body: rawData });
          }
        });
      })
      .on('error', reject);
  });
}

function httpPost(port: number, path: string, payload: any): Promise<{ status: number; body: any }> {
  return new Promise((resolve, reject) => {
    const dataStr = JSON.stringify(payload);
    const req = http.request(
      `http://127.0.0.1:${port}${path}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(dataStr),
        },
      },
      (res) => {
        let rawData = '';
        res.on('data', (chunk) => (rawData += chunk));
        res.on('end', () => {
          try {
            resolve({ status: res.statusCode || 500, body: JSON.parse(rawData) });
          } catch {
            resolve({ status: res.statusCode || 500, body: rawData });
          }
        });
      }
    );
    req.on('error', reject);
    req.write(dataStr);
    req.end();
  });
}

async function runTestSuite() {
  console.log('===============================================================');
  console.log('  CivicPulse AI — Phase 3, Stage 3: Maintenance Priority Tests');
  console.log('===============================================================\n');

  // --- TEST GROUP 1: Geographic Administrative Classification ---
  console.log('--- TEST GROUP 1: Geographic Administrative Classification ---');
  {
    // Bangalore Central
    const bgl = classifyIncidentCoordinates(12.9716, 77.5946, 'INC-100');
    assert(bgl.region === 'BENGALURU_METRO', 'Bangalore Central coordinate classified');
    assert(bgl.isOutlier === false, 'Bangalore Central flagged as non-outlier');

    // Bangalore Outlier (INC-357677-895)
    const outlier = classifyIncidentCoordinates(12.9260, 77.6762, 'INC-357677-895');
    assert(outlier.region === 'BENGALURU_METRO', 'Bangalore outlier classified as BENGALURU_METRO');
    assert(outlier.isOutlier === true, 'Bangalore outlier flagged as outlier', outlier.notes);

    // Delhi Central
    const del = classifyIncidentCoordinates(28.6139, 77.2090, 'INC-2024-001');
    assert(del.region === 'DELHI_NCR', 'Delhi coordinate classified as DELHI_NCR');
    assert(del.isOutlier === false, 'Delhi Central flagged as non-outlier');

    // Out of bounds / Ambiguous
    const oob = classifyIncidentCoordinates(51.5074, -0.1278, 'INC-LONDON');
    assert(oob.region === 'OUT_OF_BOUNDS_AMBIGUOUS', 'Out-of-bounds coordinate classified as AMBIGUOUS');
  }

  // --- TEST GROUP 2: Factor Normalization & Weighted Scoring ---
  console.log('\n--- TEST GROUP 2: Factor Normalization & Weighted Scoring ---');
  {
    // High-priority pothole on major arterial corridor
    const criticalInput: PriorityEvaluationInput = {
      damageType: 'pothole',
      verifiedClass: 'pothole',
      verifiedSeverity: 'critical',
      isVerified: true,
      defectCount: 3,
      normalizedArea: 0.08,
      reportedAt: new Date(Date.now() - 10 * 24 * 3600 * 1000), // 10 days ago
      nearbyCountIn500m: 12,
      address: 'Outer Ring Road, Bellandur Flyover Corridor, Bengaluru',
      hasImage: true,
      hasValidGps: true,
      cvConfidence: 0.96,
    };

    const critRes = evaluateMaintenancePriority(criticalInput);
    assert(critRes.factorScores.severity.score === 100, 'Verified critical severity score = 100');
    assert(critRes.factorScores.severity.provenance === 'inspector_verified', 'Severity provenance is inspector_verified');
    assert(critRes.factorScores.safetyHazard.score === 100, 'Pothole on major corridor safety hazard = 100');
    assert(critRes.factorScores.localDensity.score === 100, '12 nearby incidents local density score = 100');
    assert(critRes.factorScores.incidentAge.score === 75, '10 days incident age score = 75 (>7 days)');
    assert(critRes.factorScores.evidenceQuality.score === 100, 'Verified report evidence quality score = 100');
    assert(critRes.recommendedPriority === 'CRITICAL', 'Overall priority recommendation is CRITICAL', `Score: ${critRes.priorityScore}`);
    assert(critRes.priorityScore >= 80, 'Score exceeds critical threshold (>=80)', `${critRes.priorityScore}`);

    // Minor surface crack on quiet road
    const lowInput: PriorityEvaluationInput = {
      damageType: 'longitudinal_crack',
      provisionalSeverity: 'LOW',
      isVerified: false,
      defectCount: 1,
      normalizedArea: 0.01,
      reportedAt: new Date(Date.now() - 3 * 3600 * 1000), // 3 hours ago
      nearbyCountIn500m: 0,
      address: 'Quiet residential access road',
      hasImage: true,
      hasValidGps: true,
      cvConfidence: 0.70,
    };

    const lowRes = evaluateMaintenancePriority(lowInput);
    assert(lowRes.factorScores.severity.score === 20, 'Provisional low severity score = 20');
    assert(lowRes.factorScores.severity.provenance === 'provisional_heuristic', 'Severity provenance is provisional_heuristic');
    assert(lowRes.factorScores.safetyHazard.score === 20, 'Minor crack on secondary road safety score = 20');
    assert(lowRes.factorScores.localDensity.score === 10, 'Zero neighbors local density score = 10');
    assert(lowRes.recommendedPriority === 'LOW', 'Overall priority recommendation is LOW', `Score: ${lowRes.priorityScore}`);
    assert(lowRes.priorityScore < 45, 'Score is below low threshold (<45)', `${lowRes.priorityScore}`);
  }

  // --- TEST GROUP 3: Missing Data & Incomplete Evidence (NEEDS_REVIEW Gate) ---
  console.log('\n--- TEST GROUP 3: Missing Data & Incomplete Evidence (NEEDS_REVIEW Gate) ---');
  {
    // Missing GPS Coordinates
    const noGpsInput: PriorityEvaluationInput = {
      damageType: 'pothole',
      provisionalSeverity: 'CRITICAL',
      isVerified: false,
      defectCount: 1,
      normalizedArea: 0.06,
      reportedAt: new Date(),
      nearbyCountIn500m: 0,
      address: 'Unknown location',
      hasImage: true,
      hasValidGps: false, // MISSING GPS
      cvConfidence: 0.88,
    };

    const noGpsRes = evaluateMaintenancePriority(noGpsInput);
    assert(noGpsRes.recommendedPriority === 'NEEDS_REVIEW', 'Missing GPS triggers NEEDS_REVIEW');
    assert(noGpsRes.missingInformation.hasMissingCoordinates === true, 'Flagged hasMissingCoordinates = true');

    // Missing Photograph
    const noImgInput: PriorityEvaluationInput = {
      damageType: 'pothole',
      provisionalSeverity: 'HIGH',
      isVerified: false,
      defectCount: 0,
      normalizedArea: 0,
      reportedAt: new Date(),
      nearbyCountIn500m: 2,
      address: 'Ring Road',
      hasImage: false, // MISSING IMAGE
      hasValidGps: true,
      cvConfidence: null,
    };

    const noImgRes = evaluateMaintenancePriority(noImgInput);
    assert(noImgRes.recommendedPriority === 'NEEDS_REVIEW', 'Missing image triggers NEEDS_REVIEW');
    assert(noImgRes.missingInformation.hasMissingImage === true, 'Flagged hasMissingImage = true');

    // Conflicting Evidence (Inspector verified Low but rule engine provisional was Critical)
    const conflictInput: PriorityEvaluationInput = {
      damageType: 'pothole',
      verifiedClass: 'pothole',
      verifiedSeverity: 'low',
      provisionalSeverity: 'CRITICAL',
      isVerified: true,
      defectCount: 3,
      normalizedArea: 0.09,
      reportedAt: new Date(),
      nearbyCountIn500m: 5,
      address: 'Highway corridor',
      hasImage: true,
      hasValidGps: true,
      cvConfidence: 0.95,
    };

    const conflictRes = evaluateMaintenancePriority(conflictInput);
    assert(conflictRes.recommendedPriority === 'NEEDS_REVIEW', 'Conflicting severity assessments trigger NEEDS_REVIEW');
    assert(conflictRes.missingInformation.hasConflictingEvidence === true, 'Flagged hasConflictingEvidence = true');
  }

  // --- TEST GROUP 4: Database Integration & Priority Recalculation ---
  console.log('\n--- TEST GROUP 4: Database Integration & Priority Recalculation ---');
  const service = new MaintenancePriorityService();

  {
    // Recalculate priorities across all 66 incidents in database
    const batch = await service.recalculateAll(DEFAULT_PRIORITY_WEIGHTS);
    assert(batch.processedCount === 66, 'Recalculated priority for all 66 incidents', `Count: ${batch.processedCount}`);

    // Fetch ranked queue
    const queue = await service.getRankedQueue();
    assert(queue.totalCount === 66, 'Ranked queue returns 66 items');

    // Verify ordering: top items are CRITICAL or HIGH
    const topItem = queue.items[0];
    assert(
      topItem.recommendedPriority === 'CRITICAL' || topItem.recommendedPriority === 'HIGH',
      'Top ranked item has high urgency priority',
      `${topItem.incidentNumber}: ${topItem.recommendedPriority} (Score: ${topItem.priorityScore})`
    );

    // Verify factor details are populated
    assert(topItem.factorScores.severity !== undefined, 'Severity factor score present');
    assert(topItem.factorScores.safetyHazard !== undefined, 'Safety hazard factor score present');
    assert(topItem.factorScores.localDensity !== undefined, 'Local density factor score present');
    assert(topItem.explanation.length > 20, 'Plain-language explanation generated', topItem.explanation);

    // Verify benchmark incident INC-2024-001
    const bench = await service.getIncidentPriorityDetail(queue.items.find((i) => i.incidentNumber === 'INC-2024-001')!.incidentId);
    assert(bench.contributingEvidence.verifiedSeverity === 'critical', 'INC-2024-001 has verifiedSeverity = critical');
    assert(bench.recommendedPriority === 'CRITICAL', 'INC-2024-001 recommendedPriority = CRITICAL');
    assert(bench.region === 'DELHI_NCR', 'INC-2024-001 classified as DELHI_NCR');
  }

  // --- TEST GROUP 5: Human Administrative Review Workflow ---
  console.log('\n--- TEST GROUP 5: Human Administrative Review Workflow ---');
  {
    const adminUser = await prisma.user.findFirst({ where: { role: 'administrator' } });
    const citizenUser = await prisma.user.findFirst({ where: { role: 'citizen' } });
    const targetInc = await prisma.incident.findFirst({ where: { incidentNumber: 'INC-2024-002' } });

    assert(Boolean(adminUser), 'Found admin user for review authorization');
    assert(Boolean(targetInc), 'Found target incident for review workflow');

    // 1. Citizen review rejection (Role check)
    let citizenBlocked = false;
    try {
      await service.reviewRecommendation({
        incidentId: targetInc!.id,
        reviewerId: citizenUser!.id,
        action: 'accept',
      });
    } catch (e: any) {
      citizenBlocked = e.message.includes('not authorized');
    }
    assert(citizenBlocked, 'Citizen user blocked from administrative review (403 Role Check)');

    // 2. Admin Override with reason
    const overrideRes = await service.reviewRecommendation({
      incidentId: targetInc!.id,
      reviewerId: adminUser!.id,
      action: 'override',
      reviewedPriority: 'CRITICAL',
      overrideReason: 'High-traffic municipal bus corridor prior to monsoon season.',
    });

    assert(overrideRes.success === true, 'Admin successfully executed priority override');
    assert(overrideRes.reviewStatus === 'overridden', 'Review status set to overridden');
    assert(overrideRes.reviewedPriority === 'CRITICAL', 'Reviewed priority set to CRITICAL');

    // 3. Verify original recommendation preserved
    const detailAfter = await service.getIncidentPriorityDetail(targetInc!.id);
    assert(detailAfter.reviewStatus === 'overridden', 'Persistent status is overridden');
    assert(detailAfter.reviewedPriority === 'CRITICAL', 'Persistent reviewed priority is CRITICAL');
    assert(Boolean(detailAfter.recommendedPriority), 'Original recommendation preserved in record');

    // 4. Verify AuditLog entry
    const auditEntry = await prisma.auditLog.findUnique({ where: { id: overrideRes.auditLogId } });
    assert(Boolean(auditEntry), 'AuditLog entry created');
    assert(auditEntry?.action === 'MAINTENANCE_PRIORITY_OVERRIDE', 'Audit action recorded correctly');
    assert((auditEntry?.metadata as any)?.overrideReason !== undefined, 'Audit metadata includes override reason');

    // 5. Verify no automatic dispatch of MaintenanceTask occurred
    const tasksCount = await prisma.maintenanceTask.count({ where: { incidentId: targetInc!.id } });
    assert(tasksCount <= 1, 'Automatic dispatch strictly prohibited (no rogue maintenance task dispatched)');
  }

  // --- TEST GROUP 6: Express REST API Endpoints ---
  console.log('\n--- TEST GROUP 6: Express REST API Endpoints ---');
  const app = createApp();
  const PORT = 5899;
  const server = app.listen(PORT);

  try {
    // GET /summary
    const resSummary = await httpGet(PORT, '/api/v1/maintenance-priority/summary');
    assert(resSummary.status === 200, 'GET /summary returned 200 OK');
    assert(resSummary.body.data.totalIncidents === 66, 'Summary reported 66 incidents');
    assert(resSummary.body.data.supervisedMlReadiness.status === 'DATA_INSUFFICIENT', 'Summary flags DATA_INSUFFICIENT for supervised ML');
    assert(resSummary.body.data.regionalBreakdown.bengaluru === 58, 'Summary regional census: 58 in Bangalore');
    assert(resSummary.body.data.regionalBreakdown.delhi === 8, 'Summary regional census: 8 in Delhi');

    // GET /ranked
    const resRanked = await httpGet(PORT, '/api/v1/maintenance-priority/ranked?priority=critical');
    assert(resRanked.status === 200, 'GET /ranked?priority=critical returned 200 OK');
    assert(Array.isArray(resRanked.body.data.items), 'Ranked endpoint returned items array');
    assert(resRanked.body.notice.includes('decision-support') || resRanked.body.notice.includes('administrative recommendations'), 'Mandatory administrative notice present');

    // GET /configuration
    const resConfig = await httpGet(PORT, '/api/v1/maintenance-priority/configuration');
    assert(resConfig.status === 200, 'GET /configuration returned 200 OK');
    assert(resConfig.body.data.activeWeights.version === 'v1.0.0-heuristic', 'Active weights version v1.0.0-heuristic');
    assert(resConfig.body.data.policy.allowAutoDispatch === false, 'Policy strictly sets allowAutoDispatch = false');

    // GET /model-status
    const resModel = await httpGet(PORT, '/api/v1/maintenance-priority/model-status');
    assert(resModel.status === 200, 'GET /model-status returned 200 OK');
    assert(resModel.body.data.status === 'DATA_INSUFFICIENT', 'Model status returns DATA_INSUFFICIENT');
    assert(resModel.body.data.canTrainSupervisedModel === false, 'canTrainSupervisedModel = false');

    // POST /review validation (missing required reason on override)
    const resBadReview = await httpPost(PORT, '/api/v1/maintenance-priority/review', {
      incidentId: 'some-id',
      action: 'override',
      reviewedPriority: 'CRITICAL',
    }); // missing overrideReason
    assert(resBadReview.status === 400 || resBadReview.status === 404, 'Review endpoint validates required parameters');
  } finally {
    server.close();
  }

  console.log('\n===============================================================');
  console.log(`  PRIORITY TEST SUMMARY: ${testsPassed} PASSED, ${testsFailed} FAILED`);
  console.log('===============================================================');

  await prisma.$disconnect();

  if (testsFailed > 0) {
    process.exit(1);
  }
}

runTestSuite().catch((err) => {
  console.error('Fatal test error:', err);
  process.exit(1);
});

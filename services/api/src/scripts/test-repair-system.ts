/**
 * CivicPulse AI — Phase 3, Stage 4: Repair Cost & Duration Estimation Test Suite
 *
 * Verifies:
 * 1. Rate configuration for Bengaluru and Delhi.
 * 2. Strict regional isolation (no cross-applying rates).
 * 3. Unit conversion, volume arithmetic, and cost summation.
 * 4. Missing physical measurement behavior (MEASUREMENT_PENDING gate).
 * 5. Duration calculation (operational, setup, curing, working days).
 * 6. Actual vs estimated separation (never presenting estimates as actuals).
 * 7. End-to-end database persistence in repair_cost_estimates.
 * 8. Supervised regression readiness gate (DATA_INSUFFICIENT).
 * 9. Express REST API endpoints.
 */

import http from 'http';
import { prisma } from '../config/prisma';
import {
  BENGALURU_RATE_SCHEDULE,
  DELHI_RATE_SCHEDULE,
  getRegionalRateSchedule,
} from '../modules/repair-estimation/rate-schedules';
import { estimateRepairCostAndDuration } from '../modules/repair-estimation/estimator';
import { RepairEstimationService } from '../modules/repair-estimation/service';
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
  console.log('  CivicPulse AI — Phase 3, Stage 4: Repair Estimation Tests');
  console.log('===============================================================\n');

  // --- TEST GROUP 1: Rate Schedules & Regional Isolation ---
  console.log('--- TEST GROUP 1: Rate Schedules & Regional Isolation ---');
  {
    assert(BENGALURU_RATE_SCHEDULE.regionId === 'BENGALURU_METRO', 'Bengaluru schedule regionId is BENGALURU_METRO');
    assert(BENGALURU_RATE_SCHEDULE.currency === 'INR', 'Bengaluru currency is INR');
    assert(BENGALURU_RATE_SCHEDULE.isOfficialDocument === false, 'Bengaluru flagged isOfficialDocument = false');
    assert(BENGALURU_RATE_SCHEDULE.materials.cold_mix_asphalt.unitRateINR === 4200.0, 'Bengaluru cold mix rate = ₹4,200/m3');

    assert(DELHI_RATE_SCHEDULE.regionId === 'DELHI_NCR', 'Delhi schedule regionId is DELHI_NCR');
    assert(DELHI_RATE_SCHEDULE.currency === 'INR', 'Delhi currency is INR');
    assert(DELHI_RATE_SCHEDULE.isOfficialDocument === false, 'Delhi flagged isOfficialDocument = false');
    assert(DELHI_RATE_SCHEDULE.materials.cold_mix_asphalt.unitRateINR === 4500.0, 'Delhi cold mix rate = ₹4,500/m3');

    // Test regional isolation
    const bglLookup = getRegionalRateSchedule('BENGALURU_METRO');
    assert(bglLookup?.scheduleVersion === 'KPWD-BBMP-PROVISIONAL-2026.1', 'Lookup BENGALURU_METRO returns Bengaluru schedule');

    const delLookup = getRegionalRateSchedule('DELHI_NCR');
    assert(delLookup?.scheduleVersion === 'CPWD-DELHI-PROVISIONAL-2026.1', 'Lookup DELHI_NCR returns Delhi schedule');

    const oobLookup = getRegionalRateSchedule('OUT_OF_BOUNDS_AMBIGUOUS');
    assert(oobLookup === null, 'Lookup unconfigured region returns null (RATE_SOURCE_UNAVAILABLE)');
  }

  // --- TEST GROUP 2: Missing Measurement Behavior (MEASUREMENT_PENDING) ---
  console.log('\n--- TEST GROUP 2: Missing Measurement Behavior (MEASUREMENT_PENDING) ---');
  {
    const unmeasuredIncident = {
      id: 'test-inc-1',
      incidentNumber: 'TEST-001',
      title: 'Pothole on ORR',
      address: 'Outer Ring Road, Bengaluru',
      damageType: 'pothole',
      latitude: 12.9716,
      longitude: 77.5946,
      normalizedArea: 0.05, // 5% of camera frame
    };

    // Call estimator with no physical measurement
    const res = estimateRepairCostAndDuration({
      incident: unmeasuredIncident,
      measuredAreaSqMeters: null, // NO PHYSICAL MEASUREMENT
      depthMeters: null,
      measurementSource: 'unmeasured_missing',
    });

    assert(res.estimationStatus === 'MEASUREMENT_PENDING', 'Missing physical area returns MEASUREMENT_PENDING');
    assert(res.totalEstimatedCostINR === null, 'totalEstimatedCostINR is null when unmeasured');
    assert(res.totalEstimatedDurationHours === null, 'totalEstimatedDurationHours is null when unmeasured');
    assert(res.measurement.isPhysicalAreaMeasured === false, 'isPhysicalAreaMeasured is false');
    assert(res.measurement.measurementCaveat.includes('cannot be converted into physical ground area'), 'Explains pixel area ratio cannot be converted to physical m2');
    assert(res.isActualExpenditure === false, 'isActualExpenditure is strictly false');
  }

  // --- TEST GROUP 3: Parametric Cost & Duration Arithmetic ---
  console.log('\n--- TEST GROUP 3: Parametric Cost & Duration Arithmetic ---');
  {
    const measuredIncident = {
      id: 'test-inc-2',
      incidentNumber: 'TEST-002',
      title: 'Measured Pothole on Ring Road',
      address: 'Ring Road Sector 2, New Delhi',
      damageType: 'pothole',
      latitude: 28.6139,
      longitude: 77.2090, // Delhi coordinates
      normalizedArea: 0.08,
    };

    // Measured cavity: 2.0 m2, 0.05m (50mm) depth
    const res = estimateRepairCostAndDuration({
      incident: measuredIncident,
      measuredAreaSqMeters: 2.0,
      depthMeters: 0.05,
      measurementSource: 'field_measurement',
    });

    assert(res.estimationStatus === 'ESTIMATED', 'Measured incident status = ESTIMATED');
    assert(res.region === 'DELHI_NCR', 'Correctly classified as DELHI_NCR');
    assert(res.rateScheduleVersion === 'CPWD-DELHI-PROVISIONAL-2026.1', 'Used Delhi rate schedule');
    assert(res.measurement.measuredAreaSqMeters === 2.0, 'Measured area = 2.0 m2');
    assert(res.measurement.depthMeters === 0.05, 'Depth = 0.05m');

    // Material Volume: 2.0 * 0.05 = 0.1 m3 of Cold mix at ₹4,500/m3 = ₹450
    // Tack coat: 2.0 m2 * ₹50/m2 = ₹100
    // Material subtotal = ₹550
    assert(res.costBreakdown?.subtotalMaterialsINR === 550.0, 'Material subtotal = ₹550.00', `Got ${res.costBreakdown?.subtotalMaterialsINR}`);

    // Labor and equipment checks
    assert(res.costBreakdown!.subtotalLaborINR > 0, 'Labor subtotal > 0', `Got ${res.costBreakdown?.subtotalLaborINR}`);
    assert(res.costBreakdown!.subtotalEquipmentINR > 0, 'Equipment subtotal > 0', `Got ${res.costBreakdown?.subtotalEquipmentINR}`);
    assert(res.costBreakdown!.mobilizationSubtotalINR > 0, 'Mobilization overhead > 0', `Got ${res.costBreakdown?.mobilizationSubtotalINR}`);

    const expectedTotal = Number(
      (
        res.costBreakdown!.subtotalMaterialsINR +
        res.costBreakdown!.subtotalLaborINR +
        res.costBreakdown!.subtotalEquipmentINR +
        res.costBreakdown!.mobilizationSubtotalINR
      ).toFixed(2)
    );
    assert(res.totalEstimatedCostINR === expectedTotal, 'Total cost matches mathematical sum of components', `Total: ₹${res.totalEstimatedCostINR}`);

    // Duration check
    assert(res.durationBreakdown!.totalEstimatedHours > 0, 'Total estimated duration hours > 0');
    assert(res.durationBreakdown!.crewSize === 4, 'Standard 4-person crew used');
    assert(res.durationBreakdown!.operationalExclusions.length >= 2, 'Operational exclusions documented');
  }

  // --- TEST GROUP 4: Database Integration & Persistence ---
  console.log('\n--- TEST GROUP 4: Database Integration & Persistence ---');
  const service = new RepairEstimationService();

  {
    // Find an actual incident from database (INC-2024-001 in Delhi)
    const inc = await prisma.incident.findFirst({ where: { incidentNumber: 'INC-2024-001' } });
    assert(Boolean(inc), 'Found benchmark incident INC-2024-001 in DB');

    // 1. Calculate with physical measurement: 3.5 m2, 0.05m depth
    const measuredResult = await service.calculateEstimate({
      incidentId: inc!.id,
      measuredAreaSqMeters: 3.5,
      depthMeters: 0.05,
      measurementSource: 'field_measurement',
    });

    assert(measuredResult.estimationStatus === 'ESTIMATED', 'DB calculateEstimate returned ESTIMATED');
    assert(measuredResult.totalEstimatedCostINR! > 0, 'Saved totalEstimatedCostINR > 0', `₹${measuredResult.totalEstimatedCostINR}`);

    // 2. Query persisted record from repair_cost_estimates
    const persisted = await prisma.repairCostEstimate.findFirst({
      where: { incidentId: inc!.id },
      orderBy: { computedAt: 'desc' },
    });

    assert(Boolean(persisted), 'Estimate persisted in repair_cost_estimates table');
    assert(persisted?.estimationStatus === 'ESTIMATED', 'Persisted status is ESTIMATED');
    assert(persisted?.measuredAreaSqMeters === 3.5, 'Persisted measured area = 3.5');
    assert(persisted?.isActualCost === false, 'Persisted isActualCost is false (strictly preserved)');

    // 3. Retrieve via service getIncidentEstimate
    const retrieved = await service.getIncidentEstimate(inc!.id);
    assert(retrieved.estimationStatus === 'ESTIMATED', 'getIncidentEstimate returned ESTIMATED');
    assert(retrieved.measurement.measuredAreaSqMeters === 3.5, 'getIncidentEstimate preserved 3.5 m2');
  }

  // --- TEST GROUP 5: Express REST API Endpoints ---
  console.log('\n--- TEST GROUP 5: Express REST API Endpoints ---');
  const app = createApp();
  const PORT = 5898;
  const server = app.listen(PORT);

  try {
    // GET /summary
    const resSummary = await httpGet(PORT, '/api/v1/repair-estimates/summary');
    assert(resSummary.status === 200, 'GET /summary returned 200 OK');
    assert(resSummary.body.data.totalIncidents === 66, 'Summary reported 66 total incidents');
    assert(resSummary.body.data.supervisedRegressionReadiness.status === 'DATA_INSUFFICIENT', 'Summary flags DATA_INSUFFICIENT for regression');
    assert(resSummary.body.data.supervisedRegressionReadiness.currentCompletedRepairs === 0, 'Current completed repairs = 0');
    assert(resSummary.body.data.supervisedRegressionReadiness.canTrainRegression === false, 'canTrainRegression = false');

    // GET /rates
    const resRates = await httpGet(PORT, '/api/v1/repair-estimates/rates');
    assert(resRates.status === 200, 'GET /rates returned 200 OK');
    assert(resRates.body.data.schedules.length === 2, 'Rates returned 2 regional schedules (Bengaluru and Delhi)');
    assert(resRates.body.data.auditNotice.includes('Zero official municipal Schedule of Rates'), 'Mandatory rate audit disclosure present');

    // GET /model-status
    const resModel = await httpGet(PORT, '/api/v1/repair-estimates/model-status');
    assert(resModel.status === 200, 'GET /model-status returned 200 OK');
    assert(resModel.body.data.status === 'DATA_INSUFFICIENT', 'Model status returns DATA_INSUFFICIENT');

    // POST /calculate
    const targetInc = await prisma.incident.findFirst({ where: { incidentNumber: 'INC-2024-002' } });
    const resCalc = await httpPost(PORT, '/api/v1/repair-estimates/calculate', {
      incidentId: targetInc!.id,
      measuredAreaSqMeters: 4.0,
      depthMeters: 0.06,
      measurementSource: 'field_measurement',
    });
    assert(resCalc.status === 200, 'POST /calculate returned 200 OK');
    assert(resCalc.body.data.estimationStatus === 'ESTIMATED', 'POST /calculate returned ESTIMATED status');
    assert(resCalc.body.data.totalEstimatedCostINR > 0, 'POST /calculate returned positive estimated cost');

    // POST /calculate validation (missing incidentId)
    const resBadCalc = await httpPost(PORT, '/api/v1/repair-estimates/calculate', {});
    assert(resBadCalc.status === 400, 'POST /calculate with missing incidentId returned 400 Bad Request');
  } finally {
    server.close();
  }

  console.log('\n===============================================================');
  console.log(`  REPAIR ESTIMATION TEST SUMMARY: ${testsPassed} PASSED, ${testsFailed} FAILED`);
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


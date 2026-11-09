/**
 * CivicPulse AI — Stage 2: Severity Assessment & Experimental Pipeline Test Suite
 * Tests feature extraction, engineering rule evaluation, evidence gating,
 * inspector ground-truth separation, and Express REST APIs.
 */

import http from 'http';
import { prisma } from '../config/prisma';
import { createApp } from '../app';
import {
  extractDetectionFeatures,
  extractImageLevelFeatures,
  normalizeDamageCategory,
} from '../modules/severity/feature-extractor';
import {
  EngineeringSeverityRuleEngine,
  SEVERITY_RULES,
} from '../modules/severity/rule-engine';
import { SeverityService } from '../modules/severity/service';

async function runSeverityTestSuite() {
  console.log('\n===============================================================');
  console.log('  CivicPulse AI — Phase 3, Stage 2: Severity Assessment Test Suite');
  console.log('===============================================================\n');

  let passedTests = 0;
  let failedTests = 0;

  function assert(condition: boolean, testName: string, detail: string = '') {
    if (condition) {
      console.log(`  [PASS] ${testName} ${detail ? `(${detail})` : ''}`);
      passedTests++;
    } else {
      console.error(`  [FAIL] ${testName} - FAILED! ${detail}`);
      failedTests++;
    }
  }

  // -----------------------------------------------------------------
  // 1. Feature Extraction Tests
  // -----------------------------------------------------------------
  console.log('--- TEST GROUP 1: Feature Extraction Pipeline ---');

  assert(normalizeDamageCategory('pothole') === 'D40', 'Normalize pothole to D40');
  assert(normalizeDamageCategory('alligator_crack') === 'D20', 'Normalize alligator_crack to D20');
  assert(normalizeDamageCategory('longitudinal_crack') === 'D00', 'Normalize longitudinal_crack to D00');
  assert(normalizeDamageCategory('transverse_crack') === 'D10', 'Normalize transverse_crack to D10');
  assert(normalizeDamageCategory(null) === 'unknown', 'Normalize null category to unknown');

  // Normalized bounding box
  const detFeat1 = extractDetectionFeatures({
    label: 'D40',
    confidence: 0.92,
    width: 0.2,
    height: 0.3,
  });
  assert(detFeat1.normalizedArea === 0.06, 'Normalized area calculation', `Area=${detFeat1.normalizedArea}`);
  assert(Math.abs(detFeat1.aspectRatio - 0.667) < 0.01, 'Aspect ratio calculation', `Aspect=${detFeat1.aspectRatio}`);

  // Pixel coordinates box
  const detFeat2 = extractDetectionFeatures(
    {
      label: 'D20',
      confidence: 0.85,
      xmin: 100,
      xmax: 300,
      ymin: 50,
      ymax: 250,
    },
    1000,
    1000
  );
  assert(detFeat2.normalizedWidth === 0.2, 'Normalized pixel width (200/1000)');
  assert(detFeat2.normalizedHeight === 0.2, 'Normalized pixel height (200/1000)');
  assert(detFeat2.normalizedArea === 0.04, 'Normalized pixel area (0.2 * 0.2)');

  // Image-level feature aggregation
  const imgFeat = extractImageLevelFeatures([
    { label: 'D40', confidence: 0.9, width: 0.2, height: 0.3 },
    { label: 'D20', confidence: 0.8, width: 0.1, height: 0.2 },
  ]);
  assert(imgFeat.totalDefectCount === 2, 'Total defect count = 2');
  assert(imgFeat.categoryDiversity === 2, 'Category diversity = 2 unique classes');
  assert(imgFeat.maxNormalizedArea === 0.06, 'Max normalized area = 0.06');

  // -----------------------------------------------------------------
  // 2. Engineering Severity Rule Engine Tests
  // -----------------------------------------------------------------
  console.log('\n--- TEST GROUP 2: Engineering Severity Rule Engine ---');

  assert(SEVERITY_RULES.length >= 7, 'Defined at least 7 transparent engineering rules');

  // Case A: Major Pothole (Area >= 0.05) -> CRITICAL
  const critPothole = EngineeringSeverityRuleEngine.evaluate({
    incidentId: 'test-1',
    incidentNumber: 'T-001',
    imageCount: 1,
    totalDefects: 1,
    categoryDistribution: { D40: 1 },
    primaryDamageCategory: 'D40',
    maxDefectArea: 0.08,
    totalDefectArea: 0.08,
    meanDetectionConfidence: 0.92,
    hasInspectorVerification: false,
    verifiedCategory: null,
    verifiedSeverity: null,
    imageFeatures: [],
    evidenceSufficient: true,
    insufficiencyReasons: [],
  });
  assert(critPothole.provisionalSeverity === 'CRITICAL', 'Major Pothole evaluated as CRITICAL');
  assert(critPothole.primaryRuleTriggered === 'RULE-SEV-01', 'Rule R1 triggered');
  assert(critPothole.isProvisional === true, 'Flagged isProvisional = true');
  assert(critPothole.isInspectorVerified === false, 'Flagged isInspectorVerified = false');

  // Case B: Minor Pothole (Area < 0.05) -> HIGH
  const highPothole = EngineeringSeverityRuleEngine.evaluate({
    incidentId: 'test-2',
    incidentNumber: 'T-002',
    imageCount: 1,
    totalDefects: 1,
    categoryDistribution: { D40: 1 },
    primaryDamageCategory: 'D40',
    maxDefectArea: 0.03,
    totalDefectArea: 0.03,
    meanDetectionConfidence: 0.85,
    hasInspectorVerification: false,
    verifiedCategory: null,
    verifiedSeverity: null,
    imageFeatures: [],
    evidenceSufficient: true,
    insufficiencyReasons: [],
  });
  assert(highPothole.provisionalSeverity === 'HIGH', 'Minor Pothole evaluated as HIGH');
  assert(highPothole.primaryRuleTriggered === 'RULE-SEV-02', 'Rule R2 triggered');

  // Case C: Extensive Alligator Cracking (Area >= 0.06) -> HIGH
  const highAlligator = EngineeringSeverityRuleEngine.evaluate({
    incidentId: 'test-3',
    incidentNumber: 'T-003',
    imageCount: 1,
    totalDefects: 2,
    categoryDistribution: { D20: 2 },
    primaryDamageCategory: 'D20',
    maxDefectArea: 0.04,
    totalDefectArea: 0.07,
    meanDetectionConfidence: 0.88,
    hasInspectorVerification: false,
    verifiedCategory: null,
    verifiedSeverity: null,
    imageFeatures: [],
    evidenceSufficient: true,
    insufficiencyReasons: [],
  });
  assert(highAlligator.provisionalSeverity === 'HIGH', 'Extensive Alligator evaluated as HIGH');
  assert(highAlligator.primaryRuleTriggered === 'RULE-SEV-03', 'Rule R3 triggered');

  // Case D: Minor Linear Crack (Area < 0.05) -> LOW
  const lowCrack = EngineeringSeverityRuleEngine.evaluate({
    incidentId: 'test-4',
    incidentNumber: 'T-004',
    imageCount: 1,
    totalDefects: 1,
    categoryDistribution: { D00: 1 },
    primaryDamageCategory: 'D00',
    maxDefectArea: 0.02,
    totalDefectArea: 0.02,
    meanDetectionConfidence: 0.79,
    hasInspectorVerification: false,
    verifiedCategory: null,
    verifiedSeverity: null,
    imageFeatures: [],
    evidenceSufficient: true,
    insufficiencyReasons: [],
  });
  assert(lowCrack.provisionalSeverity === 'LOW', 'Minor Linear Crack evaluated as LOW');
  assert(lowCrack.primaryRuleTriggered === 'RULE-SEV-06', 'Rule R6 triggered');

  // Case E: Insufficient Evidence Handling
  const noEvidence = EngineeringSeverityRuleEngine.evaluate({
    incidentId: 'test-5',
    incidentNumber: 'T-005',
    imageCount: 0,
    totalDefects: 0,
    categoryDistribution: {},
    primaryDamageCategory: 'unknown',
    maxDefectArea: 0.0,
    totalDefectArea: 0.0,
    meanDetectionConfidence: 0.0,
    hasInspectorVerification: false,
    verifiedCategory: null,
    verifiedSeverity: null,
    imageFeatures: [],
    evidenceSufficient: false,
    insufficiencyReasons: ['No field evidence photographs attached to incident.'],
  });
  assert(
    noEvidence.provisionalSeverity === 'INSUFFICIENT_EVIDENCE',
    'Missing evidence returns INSUFFICIENT_EVIDENCE'
  );

  // -----------------------------------------------------------------
  // 3. Database & Ground Truth Audit Tests
  // -----------------------------------------------------------------
  console.log('\n--- TEST GROUP 3: Database & Ground Truth Audit ---');

  const summary = await SeverityService.getSummary();
  assert(summary.totalIncidents === 66, 'Total incidents census = 66');
  assert(summary.totalVerifiedIncidents === 6, 'Total verified ground truth = 6');
  assert(summary.mlReadinessGate.status === 'DATA_INSUFFICIENT', 'ML readiness gate flags DATA_INSUFFICIENT');
  assert(summary.mlReadinessGate.requiredSamples === 50, 'Required samples = 50');

  // Fetch verified incident INC-2024-001
  const inc1 = await prisma.incident.findFirst({
    where: { incidentNumber: 'INC-2024-001' },
  });
  assert(inc1 !== null, 'Found benchmark incident INC-2024-001');

  if (inc1) {
    const profile = await SeverityService.getIncidentSeverity(inc1.id);
    assert(profile !== null, 'Retrieved incident severity profile');
    if (profile) {
      assert(profile.inspectorGroundTruth.isVerified === true, 'Inspector verification status = true');
      assert(profile.inspectorGroundTruth.verifiedSeverity === 'critical', 'Inspector ground truth = critical');
      assert(
        profile.inspectorGroundTruth.inspectorName?.includes('Marcus Vance') === true,
        'Preserved inspector identity (Marcus Vance)'
      );
      assert(profile.provisionalAssessment.isProvisional === true, 'Provisional assessment kept separate from ground truth');
    }
  }

  // -----------------------------------------------------------------
  // 4. Express REST API Endpoints Integration Test
  // -----------------------------------------------------------------
  console.log('\n--- TEST GROUP 4: Express REST API Endpoints ---');

  const app = createApp();
  const PORT = 5058;
  const server = app.listen(PORT, '127.0.0.1');

  function httpGet(path: string): Promise<{ status: number; body: any }> {
    return new Promise((resolve, reject) => {
      http
        .get(`http://127.0.0.1:${PORT}${path}`, (res) => {
          let rawData = '';
          res.on('data', (chunk) => (rawData += chunk));
          res.on('end', () => {
            try {
              resolve({ status: res.statusCode || 500, body: JSON.parse(rawData) });
            } catch (e) {
              resolve({ status: res.statusCode || 500, body: rawData });
            }
          });
        })
        .on('error', reject);
    });
  }

  function httpPost(path: string, payload: any): Promise<{ status: number; body: any }> {
    return new Promise((resolve, reject) => {
      const dataStr = JSON.stringify(payload);
      const req = http.request(
        `http://127.0.0.1:${PORT}${path}`,
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
            } catch (e) {
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

  try {
    // 4.1 GET /api/v1/severity/summary
    const resSummary = await httpGet('/api/v1/severity/summary');
    assert(resSummary.status === 200, 'GET /api/v1/severity/summary returned 200');
    assert(resSummary.body.data.mlReadinessGate.status === 'DATA_INSUFFICIENT', 'Summary exposes DATA_INSUFFICIENT');

    // 4.2 GET /api/v1/severity/rules
    const resRules = await httpGet('/api/v1/severity/rules');
    assert(resRules.status === 200, 'GET /api/v1/severity/rules returned 200');
    assert(resRules.body.data.rules.length >= 7, 'Rules API returned rule definitions');

    // 4.3 GET /api/v1/severity/model-status
    const resStatus = await httpGet('/api/v1/severity/model-status');
    assert(resStatus.status === 200, 'GET /api/v1/severity/model-status returned 200');
    assert(
      resStatus.body.data.status === 'DATA_INSUFFICIENT_FOR_SUPERVISED_TRAINING',
      'Model status flags DATA_INSUFFICIENT'
    );

    // 4.4 GET /api/v1/severity/incidents/:id
    if (inc1) {
      const resIncident = await httpGet(`/api/v1/severity/incidents/${inc1.id}`);
      assert(resIncident.status === 200, 'GET /api/v1/severity/incidents/:id returned 200');
      assert(
        resIncident.body.data.inspectorGroundTruth.verifiedSeverity === 'critical',
        'Incident endpoint returned verified severity'
      );
    }

    // 4.5 POST /api/v1/severity/assess
    if (inc1) {
      const resAssess = await httpPost('/api/v1/severity/assess', { incidentId: inc1.id });
      assert(resAssess.status === 200, 'POST /api/v1/severity/assess returned 200');
      assert(
        resAssess.body.data.provisionalAssessment.isProvisional === true,
        'Assessment returned provisional evaluation'
      );
    }

    // 4.6 Error handling
    const resBadId = await httpGet('/api/v1/severity/incidents/00000000-0000-0000-0000-000000000000');
    assert(resBadId.status === 404, 'Non-existent incident returned 404');

    const resEmptyPost = await httpPost('/api/v1/severity/assess', {});
    assert(resEmptyPost.status === 400, 'Missing incidentId in body returned 400 Bad Request');
  } finally {
    server.close();
  }

  // -----------------------------------------------------------------
  // SUMMARY
  // -----------------------------------------------------------------
  console.log('\n===============================================================');
  console.log(`  SEVERITY TEST SUMMARY: ${passedTests} PASSED, ${failedTests} FAILED`);
  console.log('===============================================================\n');

  if (failedTests > 0) {
    process.exit(1);
  }
}

runSeverityTestSuite()
  .catch((e) => {
    console.error('Test suite failed with unexpected error:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

/**
 * CivicPulse AI — Phase 2.3: End-to-End API Deduplication & Incident Pipeline Test
 * Spins up Express createApp(), tests real HTTP requests against PostgreSQL and MinIO.
 */

import * as fs from 'fs';
import * as path from 'path';
import { createApp } from '../app';
import { prisma } from '../config/prisma';

async function runE2EApiTests() {
  console.log('\n===============================================================');
  console.log('  CivicPulse AI — Phase 2.3 End-to-End HTTP API Deduplication Test');
  console.log('===============================================================\n');

  const app = createApp();
  const PORT = 5055;
  const server = app.listen(PORT, '127.0.0.1');
  const baseUrl = `http://127.0.0.1:${PORT}/api/v1`;

  const createdIncidentIds: string[] = [];

  try {
    const testImgPath = path.resolve(__dirname, '../../../../datasets/raw/RDD2022_India/test/images/India_000038.jpg');
    if (!fs.existsSync(testImgPath)) {
      throw new Error(`Test image not found at ${testImgPath}`);
    }
    const imgBuffer = fs.readFileSync(testImgPath);

    // 1. Report Incident A (Canonical)
    console.log('[Step 1] Submitting Incident A via POST /api/v1/incidents/report...');
    const formDataA = new FormData();
    formDataA.append('image', new Blob([imgBuffer], { type: 'image/jpeg' }), 'India_000038.jpg');
    formDataA.append('roadName', 'Hosur Main Road Silk Board Junction');
    formDataA.append('address', 'Bengaluru South');
    formDataA.append('latitude', '12.91740');
    formDataA.append('longitude', '77.62310');

    const resA = await fetch(`${baseUrl}/incidents/report`, {
      method: 'POST',
      body: formDataA,
    });

    if (!resA.ok) {
      throw new Error(`Failed to report incident A: ${resA.status} ${await resA.text()}`);
    }

    const jsonA = (await resA.json()) as any;
    const incidentA = jsonA.data.incident;
    createdIncidentIds.push(incidentA.id);

    console.log(`  ✓ Incident A Created: ${incidentA.incidentNumber} (${incidentA.id})`);
    console.log(`  ✓ Image Hashes -> dHash: ${jsonA.data.image.dHash}, pHash: ${jsonA.data.image.perceptualHash}`);
    if (!jsonA.data.image.dHash || !jsonA.data.image.perceptualHash) {
      throw new Error('Image hashes were not saved on IncidentImage!');
    }

    // 2. Report Incident B (~4.5 meters away at 12.91744, 77.62310 with identical photo)
    console.log('\n[Step 2] Submitting Incident B (~4.5m away) via POST /api/v1/incidents/report...');
    const formDataB = new FormData();
    formDataB.append('image', new Blob([imgBuffer], { type: 'image/jpeg' }), 'India_000038_duplicate.jpg');
    formDataB.append('roadName', 'Silk Board South Approach Ramp');
    formDataB.append('address', 'Near Metro Pillar 124');
    formDataB.append('latitude', '12.91744');
    formDataB.append('longitude', '77.62310');

    const resB = await fetch(`${baseUrl}/incidents/report`, {
      method: 'POST',
      body: formDataB,
    });

    if (!resB.ok) {
      throw new Error(`Failed to report incident B: ${resB.status} ${await resB.text()}`);
    }

    const jsonB = (await resB.json()) as any;
    const incidentB = jsonB.data.incident;
    createdIncidentIds.push(incidentB.id);

    console.log(`  ✓ Incident B Created: ${incidentB.incidentNumber} (${incidentB.id})`);
    console.log(`  ✓ Duplicate Detection Auto-Flag -> Status: '${incidentB.duplicateStatus}', Confidence: ${incidentB.duplicateConfidence}`);
    console.log(`  ✓ Duplicate Info returned to client:`, jsonB.data.duplicateInfo);

    if (incidentB.duplicateStatus !== 'suspected') {
      throw new Error(`Expected Incident B to have duplicateStatus='suspected', got '${incidentB.duplicateStatus}'`);
    }

    // 3. Query Pending Duplicate Reviews
    console.log('\n[Step 3] Querying GET /api/v1/incidents/duplicates/pending-review...');
    const resPending = await fetch(`${baseUrl}/incidents/duplicates/pending-review`);
    const jsonPending = (await resPending.json()) as any;
    console.log(`  ✓ Pending duplicate count: ${jsonPending.data.length}`);
    const foundBInPending = jsonPending.data.some((inc: any) => inc.id === incidentB.id);
    if (!foundBInPending) {
      throw new Error('Incident B was not found in pending duplicate review queue');
    }

    // 4. Query Duplicate Candidates for Incident B
    console.log(`\n[Step 4] Querying GET /api/v1/incidents/${incidentB.id}/duplicate-candidates...`);
    const resCandidates = await fetch(`${baseUrl}/incidents/${incidentB.id}/duplicate-candidates?radius=15&days=30`);
    const jsonCandidates = (await resCandidates.json()) as any;

    console.log(`  ✓ Candidates Found: ${jsonCandidates.data.candidates.length}`);
    const matchedCandidateA = jsonCandidates.data.candidates.find(
      (c: any) => c.candidateIncident.id === incidentA.id
    );
    if (!matchedCandidateA) {
      throw new Error('Candidate A was not returned in duplicate candidate list');
    }
    console.log(`  ✓ Matched Candidate A: Distance = ${matchedCandidateA.candidateIncident.distanceMeters}m`);
    console.log(`  ✓ Confidence Score: ${matchedCandidateA.scoring.confidenceScore}`);
    console.log(`  ✓ Reasons:`, matchedCandidateA.scoring.reasons);

    // 5. Inspector Confirms Duplicate
    console.log(`\n[Step 5] Inspector confirms duplicate via POST /api/v1/incidents/${incidentB.id}/duplicates/confirm...`);
    const resConfirm = await fetch(`${baseUrl}/incidents/${incidentB.id}/duplicates/confirm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        canonicalIncidentId: incidentA.id,
        remarks: 'Inspector verified physical duplicate on field survey.',
      }),
    });

    if (!resConfirm.ok) {
      throw new Error(`Failed to confirm duplicate: ${resConfirm.status} ${await resConfirm.text()}`);
    }
    const jsonConfirm = (await resConfirm.json()) as any;
    console.log(`  ✓ Confirmed duplicate response:`, jsonConfirm.message);

    // Verify in database: canonical relationship & zero deletion
    const dbIncB = await prisma.incident.findUnique({
      where: { id: incidentB.id },
      include: { canonicalIncident: true, images: true },
    });
    if (!dbIncB || dbIncB.canonicalIncidentId !== incidentA.id) {
      throw new Error('Database canonicalIncidentId was not properly set!');
    }
    if (dbIncB.duplicateStatus !== 'confirmed_duplicate') {
      throw new Error(`Expected duplicateStatus='confirmed_duplicate', got '${dbIncB.duplicateStatus}'`);
    }
    if (dbIncB.images.length === 0) {
      throw new Error('Citizen submission image was deleted!');
    }
    console.log(`  ✓ Citizen submission verified intact in PostgreSQL with canonicalIncidentId=${dbIncB.canonicalIncidentId}`);

    // Verify AuditLog
    const auditLogConfirm = await prisma.auditLog.findFirst({
      where: { entityId: incidentB.id, action: 'incident_duplicate_confirmed' },
    });
    if (!auditLogConfirm) {
      throw new Error('AuditLog entry was not created for duplicate confirmation!');
    }
    console.log(`  ✓ AuditLog confirmed: id=${auditLogConfirm.id}, action=${auditLogConfirm.action}`);

    // 6. Report Incident C and Test Rejection
    console.log('\n[Step 6] Submitting Incident C and testing Rejection Workflow...');
    const formDataC = new FormData();
    formDataC.append('image', new Blob([imgBuffer], { type: 'image/jpeg' }), 'India_000038_reject.jpg');
    formDataC.append('roadName', 'Hosur Service Road');
    formDataC.append('address', 'Service Lane West');
    formDataC.append('latitude', '12.91742');
    formDataC.append('longitude', '77.62312');

    const resC = await fetch(`${baseUrl}/incidents/report`, {
      method: 'POST',
      body: formDataC,
    });
    const jsonC = (await resC.json()) as any;
    const incidentC = jsonC.data.incident;
    createdIncidentIds.push(incidentC.id);

    // Inspector rejects duplicate suggestion
    const resReject = await fetch(`${baseUrl}/incidents/${incidentC.id}/duplicates/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        candidateIncidentId: incidentA.id,
        remarks: 'Distinct defect located on separated service road.',
      }),
    });

    if (!resReject.ok) {
      throw new Error(`Failed to reject duplicate: ${resReject.status} ${await resReject.text()}`);
    }
    const jsonReject = (await resReject.json()) as any;
    console.log(`  ✓ Rejection response:`, jsonReject.message);

    const dbIncC = await prisma.incident.findUnique({ where: { id: incidentC.id } });
    if (!dbIncC || dbIncC.duplicateStatus !== 'rejected' || dbIncC.canonicalIncidentId !== null) {
      throw new Error('Incident C was not properly updated upon rejection!');
    }
    console.log(`  ✓ Incident C verified as duplicateStatus='rejected' with canonicalIncidentId=null`);

    const auditLogReject = await prisma.auditLog.findFirst({
      where: { entityId: incidentC.id, action: 'incident_duplicate_rejected' },
    });
    if (!auditLogReject) {
      throw new Error('AuditLog entry was not created for duplicate rejection!');
    }
    console.log(`  ✓ AuditLog confirmed: id=${auditLogReject.id}, action=${auditLogReject.action}`);

    console.log('\n===============================================================');
    console.log('  ALL END-TO-END HTTP API DEDUPLICATION TESTS PASSED (100%)');
    console.log('===============================================================\n');
  } finally {
    // Cleanup created test records
    if (createdIncidentIds.length > 0) {
      await prisma.incidentImage.deleteMany({
        where: { incidentId: { in: createdIncidentIds } },
      });
      await prisma.aiPrediction.deleteMany({
        where: { incidentId: { in: createdIncidentIds } },
      });
      await prisma.auditLog.deleteMany({
        where: { entityId: { in: createdIncidentIds } },
      });
      await prisma.incident.deleteMany({
        where: { id: { in: createdIncidentIds } },
      });
    }

    server.close();
    await prisma.$disconnect();
  }
}

runE2EApiTests().catch((err) => {
  console.error('\n❌ E2E API Test Suite Failed:', err);
  process.exit(1);
});


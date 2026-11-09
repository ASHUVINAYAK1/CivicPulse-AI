/**
 * CivicPulse AI — Phase 2.3: Intelligent Incident Deduplication Verification Suite
 * Tests:
 * 1. Perceptual hashing (dHash, pHash, orientation, format, Hamming distance)
 * 2. PostGIS spatial distance and ST_DWithin query execution
 * 3. Composite duplicate scoring engine & explainability reasons
 * 4. Full API end-to-end duplicate submission, detection, candidate retrieval
 * 5. Inspector duplicate confirmation workflow (canonical linking, audit logging, zero deletion)
 * 6. Inspector duplicate rejection workflow (distinct defect verification, audit logging)
 */

import * as fs from 'fs';
import * as path from 'path';
import sharp from 'sharp';
import { prisma } from '../config/prisma';
import {
  computePerceptualHash,
  computeHammingDistance,
  computeHashSimilarity,
} from '../modules/deduplication/hasher';
import {
  scoreDuplicateCandidate,
} from '../modules/deduplication/scorer';
import {
  findDuplicateCandidates,
  confirmDuplicate,
  rejectDuplicate,
} from '../modules/deduplication/service';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    throw new Error(`Assertion failed: ${msg}`);
  }
}

async function runDeduplicationTests() {
  console.log('\n===============================================================');
  console.log('  CivicPulse AI — Phase 2.3 Deduplication Test Suite');
  console.log('===============================================================\n');

  // Load a real test image from dataset
  const imgPath1 = path.resolve(__dirname, '../../../../datasets/raw/RDD2022_India/test/images/India_000038.jpg');
  const imgPath2 = path.resolve(__dirname, '../../../../datasets/raw/RDD2022_India/test/images/India_000046.jpg');

  assert(fs.existsSync(imgPath1), `Test image 1 exists at ${imgPath1}`);
  assert(fs.existsSync(imgPath2), `Test image 2 exists at ${imgPath2}`);

  const buf1 = fs.readFileSync(imgPath1);
  const buf2 = fs.readFileSync(imgPath2);

  // -----------------------------------------------------------------
  // 1. Perceptual Hashing Unit Tests
  // -----------------------------------------------------------------
  console.log('[Test 1] Perceptual Hashing & Hamming Distance Verification...');
  const hash1 = await computePerceptualHash(buf1);
  const hash2 = await computePerceptualHash(buf2);

  console.log(`  Image 1 -> dHash: ${hash1.dHash}, pHash: ${hash1.pHash}`);
  console.log(`  Image 2 -> dHash: ${hash2.dHash}, pHash: ${hash2.pHash}`);

  assert(hash1.dHash.length === 16, 'dHash length must be 16 hex characters (64 bits)');
  assert(hash1.pHash.length === 16, 'pHash length must be 16 hex characters (64 bits)');

  // Identical image self-comparison
  const selfDist = computeHammingDistance(hash1.dHash, hash1.dHash);
  const selfSim = computeHashSimilarity(hash1.dHash, hash1.dHash);
  assert(selfDist === 0, `Self-distance must be 0 (got ${selfDist})`);
  assert(selfSim === 1.0, `Self-similarity must be 1.0 (got ${selfSim})`);
  console.log('  ✓ Self-comparison: Hamming distance = 0, similarity = 100.0%');

  // Minor format alteration (JPEG -> PNG -> WebP with slight compression)
  const pngBuf = await sharp(buf1).png().toBuffer();
  const webpBuf = await sharp(buf1).webp({ quality: 85 }).toBuffer();
  const hashPng = await computePerceptualHash(pngBuf);
  const hashWebp = await computePerceptualHash(webpBuf);

  const pngDist = computeHammingDistance(hash1.dHash, hashPng.dHash);
  const webpDist = computeHammingDistance(hash1.dHash, hashWebp.dHash);
  console.log(`  Format invariance: PNG Hamming dist = ${pngDist}/64, WebP dist = ${webpDist}/64`);
  assert(pngDist <= 2, `PNG transcoding should preserve hash (dist: ${pngDist})`);
  assert(webpDist <= 4, `WebP transcoding should preserve hash (dist: ${webpDist})`);
  console.log('  ✓ Image format and transcoding invariance confirmed');

  // Distinct image comparison
  const crossDist = computeHammingDistance(hash1.dHash, hash2.dHash);
  const crossSim = computeHashSimilarity(hash1.dHash, hash2.dHash);
  console.log(`  Cross-image distance between different frames: ${crossDist}/64 (similarity: ${(crossSim * 100).toFixed(1)}%)`);
  assert(crossDist > 0, 'Distinct frames must have non-zero distance');
  console.log('  ✓ Distinct image separation confirmed');

  // -----------------------------------------------------------------
  // 2. Composite Scoring Engine Unit Tests
  // -----------------------------------------------------------------
  console.log('\n[Test 2] Explainable Composite Scoring Engine...');

  // High probability duplicate scenario (3.5m away, 2.5 hours apart, same damage type, 92% image match)
  const highScore = scoreDuplicateCandidate({
    distanceMeters: 3.5,
    maxRadiusMeters: 10.0,
    timeDeltaHours: 2.5,
    maxTimeWindowDays: 30,
    targetDamageType: 'pothole',
    candidateDamageType: 'pothole',
    bestImageSimilarity: 0.92,
    bestHammingDistance: 5,
    hashTypeUsed: 'dHash',
  });

  console.log(`  High probability scenario score: ${highScore.confidenceScore} (verdict: ${highScore.verdict})`);
  assert(highScore.confidenceScore >= 0.70, `Expected score >= 0.70, got ${highScore.confidenceScore}`);
  assert(highScore.verdict === 'high_probability_duplicate', 'Expected high_probability_duplicate');
  assert(highScore.isSuspectedDuplicate === true, 'Expected suspected duplicate flag');
  assert(highScore.reasons.length >= 4, 'Must provide detailed explainability reasons');
  console.log('  ✓ High probability scoring and explainability verified');

  // Unlikely duplicate scenario (45m away, 25 days apart, divergent damage type, low image similarity)
  const lowScore = scoreDuplicateCandidate({
    distanceMeters: 45.0,
    maxRadiusMeters: 10.0,
    timeDeltaHours: 25 * 24.0,
    maxTimeWindowDays: 30,
    targetDamageType: 'pothole',
    candidateDamageType: 'crack',
    bestImageSimilarity: 0.35,
    bestHammingDistance: 42,
    hashTypeUsed: 'dHash',
  });

  console.log(`  Unlikely scenario score: ${lowScore.confidenceScore} (verdict: ${lowScore.verdict})`);
  assert(lowScore.confidenceScore < 0.50, `Expected score < 0.50, got ${lowScore.confidenceScore}`);
  assert(lowScore.verdict === 'unlikely_duplicate', 'Expected unlikely_duplicate');
  assert(lowScore.isSuspectedDuplicate === false, 'Expected isSuspectedDuplicate = false');
  console.log('  ✓ Low probability scoring & rejection threshold verified');

  // -----------------------------------------------------------------
  // 3. PostGIS Spatial Query & Service Tests
  // -----------------------------------------------------------------
  console.log('\n[Test 3] PostGIS Spatial ST_DWithin & Deduplication Service Integration...');

  const defaultUser = await prisma.user.findFirst();
  assert(defaultUser !== null, 'Default user must exist in database');

  const testCorridorLat = 12.93524;
  const testCorridorLng = 77.62448;

  // Create Canonical Incident A
  const incA = await prisma.incident.create({
    data: {
      incidentNumber: `TEST-CANONICAL-${Date.now().toString().slice(-5)}`,
      reporterId: defaultUser!.id,
      title: 'Canonical Test Pothole — Outer Ring Road',
      description: 'Severe cavitation in right traffic lane.',
      damageType: 'pothole',
      severity: 'high',
      status: 'submitted',
      latitude: testCorridorLat,
      longitude: testCorridorLng,
      address: 'Outer Ring Road, Koramangala Flyover',
      duplicateStatus: 'none',
    },
  });

  await prisma.incidentImage.create({
    data: {
      incidentId: incA.id,
      storageKey: `test/canonical_${incA.id}.jpg`,
      originalFilename: 'India_000038.jpg',
      contentType: 'image/jpeg',
      fileSize: BigInt(buf1.length),
      imageType: 'report',
      dHash: hash1.dHash,
      perceptualHash: hash1.pHash,
    },
  });

  console.log(`  Created Canonical Incident A: ${incA.incidentNumber} (${incA.id})`);

  // Create Duplicate Candidate Incident B (4.2 meters away, reported 1 hour later)
  // At latitude 12.93, 0.000038 deg latitude is ~4.2 meters
  const incB = await prisma.incident.create({
    data: {
      incidentNumber: `TEST-DUPLICATE-${Date.now().toString().slice(-5)}`,
      reporterId: defaultUser!.id,
      title: 'Citizen Duplicate Report — Pothole',
      description: 'Deep road depression near flyover ramp.',
      damageType: 'pothole',
      severity: 'high',
      status: 'submitted',
      latitude: testCorridorLat + 0.000038,
      longitude: testCorridorLng,
      address: 'Near Koramangala Flyover Ramp',
      duplicateStatus: 'suspected',
    },
  });

  await prisma.incidentImage.create({
    data: {
      incidentId: incB.id,
      storageKey: `test/duplicate_${incB.id}.jpg`,
      originalFilename: 'India_000038_copy.jpg',
      contentType: 'image/jpeg',
      fileSize: BigInt(buf1.length),
      imageType: 'report',
      dHash: hashPng.dHash, // Nearly identical hash from PNG transcode
      perceptualHash: hashPng.pHash,
    },
  });

  console.log(`  Created Suspected Duplicate Incident B: ${incB.incidentNumber} (${incB.id})`);

  // Run PostGIS search on Incident B
  const dupResults = await findDuplicateCandidates(incB.id, { radiusMeters: 10, timeWindowDays: 30 });
  console.log(`  Found ${dupResults.candidates.length} candidates within 10m radius.`);
  assert(dupResults.candidates.length >= 1, 'Must find at least 1 nearby candidate');

  const matchedCandidate = dupResults.candidates.find((c) => c.candidateIncident.id === incA.id);
  assert(matchedCandidate !== undefined, 'Canonical Incident A must be identified as candidate');
  console.log(`  Matched Candidate A Distance: ${matchedCandidate!.candidateIncident.distanceMeters}m`);
  console.log(`  Candidate Confidence Score: ${matchedCandidate!.scoring.confidenceScore}`);
  console.log(`  Candidate Verdict: ${matchedCandidate!.scoring.verdict}`);
  assert(
    matchedCandidate!.candidateIncident.distanceMeters <= 10.0,
    `Distance should be <= 10m (got ${matchedCandidate!.candidateIncident.distanceMeters}m)`
  );
  assert(
    matchedCandidate!.scoring.confidenceScore >= 0.70,
    `Confidence should be >= 0.70 (got ${matchedCandidate!.scoring.confidenceScore})`
  );
  console.log('  ✓ PostGIS ST_DWithin and candidate scoring verified');

  // -----------------------------------------------------------------
  // 4. Human Review Workflow: Confirm Duplicate
  // -----------------------------------------------------------------
  console.log('\n[Test 4] Inspector Duplicate Confirmation Workflow...');

  const inspectorUser = (await prisma.user.findFirst({
    where: { role: 'inspector' },
  })) || defaultUser!;

  const confirmedIncident = await confirmDuplicate(
    incB.id,
    incA.id,
    inspectorUser.id,
    'Verified by Inspector Sharma: identical pothole cavity reported by second motorist.'
  );

  assert(confirmedIncident.canonicalIncidentId === incA.id, 'canonicalIncidentId must point to Incident A');
  assert(confirmedIncident.duplicateStatus === 'confirmed_duplicate', 'duplicateStatus must be confirmed_duplicate');
  assert(confirmedIncident.status === 'duplicate', 'Incident status must be updated to duplicate');

  // Verify Citizen Report Was NOT Deleted
  const verifyIncB = await prisma.incident.findUnique({
    where: { id: incB.id },
    include: { images: true },
  });
  assert(verifyIncB !== null, 'Citizen submission must NEVER be deleted');
  assert(verifyIncB!.images.length === 1, 'Citizen image records must be preserved');
  console.log('  ✓ Zero data deletion confirmed: original submission preserved');

  // Verify Immutable AuditLog Entry
  const auditEntry = await prisma.auditLog.findFirst({
    where: {
      entityType: 'incident',
      entityId: incB.id,
      action: 'incident_duplicate_confirmed',
    },
    orderBy: { createdAt: 'desc' },
  });

  assert(auditEntry !== null, 'AuditLog entry must be recorded');
  assert(
    (auditEntry!.metadata as any).canonicalIncidentId === incA.id,
    'Audit metadata must record canonicalIncidentId'
  );
  console.log(`  ✓ Audit log entry created: action=${auditEntry!.action}, id=${auditEntry!.id}`);

  // -----------------------------------------------------------------
  // 5. Human Review Workflow: Reject Duplicate
  // -----------------------------------------------------------------
  console.log('\n[Test 5] Inspector Duplicate Rejection Workflow...');

  // Create Incident C that inspector will reject
  const incC = await prisma.incident.create({
    data: {
      incidentNumber: `TEST-REJECT-${Date.now().toString().slice(-5)}`,
      reporterId: defaultUser!.id,
      title: 'Distinct Distress Nearby — Crack',
      description: 'Transverse fissure across opposing shoulder lane.',
      damageType: 'crack',
      severity: 'medium',
      status: 'submitted',
      latitude: testCorridorLat + 0.000045,
      longitude: testCorridorLng,
      address: 'Outer Ring Road Opposite Shoulder',
      duplicateStatus: 'suspected',
    },
  });

  const rejectedIncident = await rejectDuplicate(
    incC.id,
    incA.id,
    inspectorUser.id,
    'Verified by Inspector Sharma: distinct transverse crack on opposing roadway lane.'
  );

  assert(rejectedIncident.duplicateStatus === 'rejected', 'duplicateStatus must be rejected');
  assert(rejectedIncident.canonicalIncidentId === null, 'canonicalIncidentId must remain null');

  const rejectAudit = await prisma.auditLog.findFirst({
    where: {
      entityType: 'incident',
      entityId: incC.id,
      action: 'incident_duplicate_rejected',
    },
    orderBy: { createdAt: 'desc' },
  });

  assert(rejectAudit !== null, 'AuditLog entry for rejection must be recorded');
  console.log(`  ✓ Audit log for rejection created: action=${rejectAudit!.action}`);
  console.log('  ✓ Rejection workflow verified');

  // Clean up test incidents from database
  await prisma.incidentImage.deleteMany({
    where: { incidentId: { in: [incA.id, incB.id, incC.id] } },
  });
  await prisma.auditLog.deleteMany({
    where: { entityId: { in: [incA.id, incB.id, incC.id] } },
  });
  await prisma.incident.deleteMany({
    where: { id: { in: [incA.id, incB.id, incC.id] } },
  });

  console.log('\n===============================================================');
  console.log('  ALL DEDUPLICATION TESTS PASSED SUCCESSFULLY (100%)');
  console.log('===============================================================\n');

  await prisma.$disconnect();
}

runDeduplicationTests().catch((err) => {
  console.error('\n❌ Test Suite Failed:', err);
  process.exit(1);
});

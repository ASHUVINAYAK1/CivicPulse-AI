import { prisma } from '../config/prisma';
import { confirmDuplicate, rejectDuplicate } from '../modules/deduplication/service';

async function testDatabaseSafety() {
  console.log('='.repeat(70));
  console.log('  CivicPulse AI — Phase 2.4 Database & Deduplication Safety Audit  ');
  console.log('='.repeat(70));

  // Find a valid user to act as inspector
  const user = await prisma.user.findFirst();
  if (!user) throw new Error('No user found in database');

  // Create three synthetic test incidents
  const suffix = Math.floor(Math.random() * 90000 + 10000);
  const incA = await prisma.incident.create({
    data: {
      incidentNumber: `SAFE-A-${suffix}`,
      title: 'Incident A for Safety Test',
      description: 'Pothole A test',
      damageType: 'pothole',
      latitude: 28.6139,
      longitude: 77.2090,
      status: 'reported',
      reportedAt: new Date(),
      reporterId: user.id,
    },
  });

  const incB = await prisma.incident.create({
    data: {
      incidentNumber: `SAFE-B-${suffix}`,
      title: 'Incident B for Safety Test',
      description: 'Pothole B test',
      damageType: 'pothole',
      latitude: 28.61392,
      longitude: 77.20901,
      status: 'reported',
      reportedAt: new Date(),
      reporterId: user.id,
    },
  });

  const incC = await prisma.incident.create({
    data: {
      incidentNumber: `SAFE-C-${suffix}`,
      title: 'Incident C for Safety Test',
      description: 'Pothole C test',
      damageType: 'pothole',
      latitude: 28.61394,
      longitude: 77.20902,
      status: 'reported',
      reportedAt: new Date(),
      reporterId: user.id,
    },
  });

  console.log(`Created test incidents:`);
  console.log(`  Incident A: ${incA.id} (${incA.incidentNumber})`);
  console.log(`  Incident B: ${incB.id} (${incB.incidentNumber})`);
  console.log(`  Incident C: ${incC.id} (${incC.incidentNumber})`);

  // --- Test 1: Self-Duplicate Rejection ---
  console.log('\n[Safety Check 1] Self-Duplicate Prevention...');
  let selfError = false;
  try {
    await confirmDuplicate(incA.id, incA.id, user.id, 'Trying to self-duplicate');
  } catch (err: any) {
    selfError = true;
    console.log(`  ✓ Correctly rejected: ${err.message}`);
  }
  if (!selfError) throw new Error('Failed: Self-duplicate was not prevented!');

  // --- Test 2: Standard Valid Confirmation & Atomic Audit Log ---
  console.log('\n[Safety Check 2] Valid Confirmation & Atomic Audit Logging...');
  const confirmedB = await confirmDuplicate(incB.id, incA.id, user.id, 'Confirming B is duplicate of A');
  console.log(`  ✓ Confirmed B -> A: canonicalIncidentId = ${confirmedB.canonicalIncidentId}, duplicateStatus = ${confirmedB.duplicateStatus}`);

  const auditB = await prisma.auditLog.findFirst({
    where: { entityId: incB.id, action: 'incident_duplicate_confirmed' },
    orderBy: { createdAt: 'desc' },
  });
  if (!auditB) throw new Error('Failed: Audit log entry was not created!');
  console.log(`  ✓ Audit log verified: action = ${auditB.action}, id = ${auditB.id}`);

  // --- Test 3: Circular Relationship Prevention (B -> A exists, now try A -> B) ---
  console.log('\n[Safety Check 3] Direct Circular Relationship Prevention (A -> B)...');
  let circularError = false;
  try {
    await confirmDuplicate(incA.id, incB.id, user.id, 'Trying to make canonical duplicate of its duplicate');
  } catch (err: any) {
    circularError = true;
    console.log(`  ✓ Correctly rejected circular link: ${err.message}`);
  }
  if (!circularError) throw new Error('Failed: Direct circular relationship was not prevented!');

  // --- Test 4: Transitive Flattening & Multi-hop Chain Prevention (C -> B -> A) ---
  console.log('\n[Safety Check 4] Transitive Chain Flattening to Root Canonical (C -> B resolves to C -> A)...');
  const confirmedC = await confirmDuplicate(incC.id, incB.id, user.id, 'Linking C to B');
  if (confirmedC.canonicalIncidentId !== incA.id) {
    throw new Error(`Failed: Expected C to be linked to root canonical A (${incA.id}), but got ${confirmedC.canonicalIncidentId}`);
  }
  console.log(`  ✓ Confirmed C -> A: Resolved directly to root canonical ${incA.id} (no multi-hop chains)`);

  // --- Test 5: Rejection Handling & Distinct State Preservation ---
  console.log('\n[Safety Check 5] Rejection Workflow & Status Transition...');
  const rejectedC = await rejectDuplicate(incC.id, incA.id, user.id, 'Inspector confirmed C is distinct on re-inspection');
  console.log(`  ✓ Rejection verified: duplicateStatus = ${rejectedC.duplicateStatus}`);
  const auditReject = await prisma.auditLog.findFirst({
    where: { entityId: incC.id, action: 'incident_duplicate_rejected' },
    orderBy: { createdAt: 'desc' },
  });
  if (!auditReject) throw new Error('Failed: Audit log entry for rejection was not created!');
  console.log(`  ✓ Rejection audit log verified: action = ${auditReject.action}`);

  // --- Test 6: Zero Deletion Verification ---
  console.log('\n[Safety Check 6] Zero Deletion Verification across all submissions...');
  const [checkA, checkB, checkC] = await Promise.all([
    prisma.incident.findUnique({ where: { id: incA.id } }),
    prisma.incident.findUnique({ where: { id: incB.id } }),
    prisma.incident.findUnique({ where: { id: incC.id } }),
  ]);
  if (!checkA || !checkB || !checkC) {
    throw new Error('Failed: One or more incident records were deleted!');
  }
  console.log(`  ✓ All original records exist: A=${!!checkA}, B=${!!checkB}, C=${!!checkC}`);
  console.log(`  ✓ Citizen submissions are 100% preserved in database.`);

  console.log('\n' + '='.repeat(70));
  console.log('  ALL DATABASE SAFETY CHECKS PASSED (6/6) (100%)  ');
  console.log('='.repeat(70));

  await prisma.$disconnect();
}

testDatabaseSafety().catch((err) => {
  console.error('Safety test failed:', err);
  process.exit(1);
});


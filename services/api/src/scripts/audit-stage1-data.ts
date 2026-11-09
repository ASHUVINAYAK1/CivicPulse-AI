import { prisma } from '../config/prisma';

async function auditData() {
  console.log('=== CIVICPULSE AI STAGE 1: DATA READINESS & GOVERNANCE AUDIT ===\n');

  // 1. Users
  const users = await prisma.user.findMany({ select: { role: true } });
  const userRoleCounts: Record<string, number> = {};
  users.forEach((u) => {
    userRoleCounts[u.role] = (userRoleCounts[u.role] || 0) + 1;
  });
  console.log('--- 1. USERS ---');
  console.log(`Total Users: ${users.length}`);
  console.log('Roles:', JSON.stringify(userRoleCounts, null, 2));

  // 2. Incidents
  const incidents = await prisma.incident.findMany({
    select: {
      id: true,
      incidentNumber: true,
      damageType: true,
      severity: true,
      status: true,
      priority: true,
      latitude: true,
      longitude: true,
      address: true,
      reportedAt: true,
      verifiedAt: true,
      duplicateStatus: true,
      canonicalIncidentId: true,
      _count: {
        select: {
          images: true,
          predictions: true,
          verifications: true,
          tasks: true,
        },
      },
    },
  });

  console.log('\n--- 2. INCIDENTS ---');
  console.log(`Total Incidents: ${incidents.length}`);

  const statusCounts: Record<string, number> = {};
  const damageTypeCounts: Record<string, number> = {};
  const severityCounts: Record<string, number> = {};
  const priorityCounts: Record<string, number> = {};
  const dupStatusCounts: Record<string, number> = {};
  let validGpsCount = 0;
  let hasAddressCount = 0;
  let verifiedAtCount = 0;
  let minDate = new Date();
  let maxDate = new Date(0);

  incidents.forEach((inc) => {
    statusCounts[inc.status] = (statusCounts[inc.status] || 0) + 1;
    damageTypeCounts[inc.damageType] = (damageTypeCounts[inc.damageType] || 0) + 1;
    const sevKey = inc.severity || 'NULL';
    severityCounts[sevKey] = (severityCounts[sevKey] || 0) + 1;
    const prioKey = inc.priority || 'NULL';
    priorityCounts[prioKey] = (priorityCounts[prioKey] || 0) + 1;
    dupStatusCounts[inc.duplicateStatus] = (dupStatusCounts[inc.duplicateStatus] || 0) + 1;

    if (inc.latitude && inc.longitude) validGpsCount++;
    if (inc.address) hasAddressCount++;
    if (inc.verifiedAt) verifiedAtCount++;
    if (inc.reportedAt < minDate) minDate = inc.reportedAt;
    if (inc.reportedAt > maxDate) maxDate = inc.reportedAt;
  });

  console.log('Status Breakdown:', statusCounts);
  console.log('Damage Types:', damageTypeCounts);
  console.log('Severities:', severityCounts);
  console.log('Priorities:', priorityCounts);
  console.log('Duplicate Statuses:', dupStatusCounts);
  console.log(`Valid GPS Coordinates: ${validGpsCount}/${incidents.length}`);
  console.log(`Address Populated: ${hasAddressCount}/${incidents.length}`);
  console.log(`VerifiedAt Populated: ${verifiedAtCount}/${incidents.length}`);
  console.log(`Earliest Incident Reported: ${minDate.toISOString()}`);
  console.log(`Latest Incident Reported: ${maxDate.toISOString()}`);

  // 3. Incident Verifications
  const verifications = await prisma.incidentVerification.findMany();
  console.log('\n--- 3. INCIDENT VERIFICATIONS ---');
  console.log(`Total Verifications: ${verifications.length}`);
  const verifiedClassCounts: Record<string, number> = {};
  const verifiedSeverityCounts: Record<string, number> = {};
  let remarksCount = 0;

  verifications.forEach((v) => {
    verifiedClassCounts[v.verifiedClass] = (verifiedClassCounts[v.verifiedClass] || 0) + 1;
    verifiedSeverityCounts[v.verifiedSeverity] = (verifiedSeverityCounts[v.verifiedSeverity] || 0) + 1;
    if (v.remarks && v.remarks.trim().length > 0) remarksCount++;
  });
  console.log('Verified Classes:', verifiedClassCounts);
  console.log('Verified Severities:', verifiedSeverityCounts);
  console.log(`Remarks Present: ${remarksCount}/${verifications.length}`);

  // 4. Maintenance Tasks
  const tasks = await prisma.maintenanceTask.findMany({
    include: {
      updates: true,
      incident: {
        select: {
          damageType: true,
          severity: true,
          status: true,
        },
      },
    },
  });

  console.log('\n--- 4. MAINTENANCE TASKS ---');
  console.log(`Total Tasks: ${tasks.length}`);
  const taskStatusCounts: Record<string, number> = {};
  const taskPriorityCounts: Record<string, number> = {};
  let estCostCount = 0;
  let actCostCount = 0;
  let estDurCount = 0;
  let actDurCount = 0;
  let startedCount = 0;
  let completedCount = 0;

  tasks.forEach((t) => {
    taskStatusCounts[t.taskStatus] = (taskStatusCounts[t.taskStatus] || 0) + 1;
    taskPriorityCounts[t.priority] = (taskPriorityCounts[t.priority] || 0) + 1;
    if (t.estimatedCost !== null) estCostCount++;
    if (t.actualCost !== null) actCostCount++;
    if (t.estimatedDuration !== null) estDurCount++;
    if (t.actualDuration !== null) actDurCount++;
    if (t.startedAt !== null) startedCount++;
    if (t.completedAt !== null) completedCount++;
  });

  console.log('Task Statuses:', taskStatusCounts);
  console.log('Task Priorities:', taskPriorityCounts);
  console.log(`Estimated Cost Present: ${estCostCount}/${tasks.length}`);
  console.log(`Actual Cost Present: ${actCostCount}/${tasks.length}`);
  console.log(`Estimated Duration Present: ${estDurCount}/${tasks.length}`);
  console.log(`Actual Duration Present: ${actDurCount}/${tasks.length}`);
  console.log(`StartedAt Present: ${startedCount}/${tasks.length}`);
  console.log(`CompletedAt Present: ${completedCount}/${tasks.length}`);

  if (tasks.length > 0) {
    console.log('Sample Task Records:', JSON.stringify(tasks.slice(0, 3), null, 2));
  }

  // 5. Maintenance Updates
  const updateCount = await prisma.maintenanceUpdate.count();
  console.log('\n--- 5. MAINTENANCE UPDATES ---');
  console.log(`Total Updates: ${updateCount}`);

  // 6. AI Predictions
  const predictions = await prisma.aiPrediction.findMany({
    select: {
      predictionType: true,
      predictedClass: true,
      confidence: true,
      processingTimeMs: true,
    },
  });
  console.log('\n--- 6. AI PREDICTIONS ---');
  console.log(`Total AI Predictions: ${predictions.length}`);
  const predTypeCounts: Record<string, number> = {};
  predictions.forEach((p) => {
    predTypeCounts[p.predictionType] = (predTypeCounts[p.predictionType] || 0) + 1;
  });
  console.log('Prediction Types:', predTypeCounts);

  // 7. Incident Forecasts
  const forecasts = await prisma.incidentForecast.findMany();
  console.log('\n--- 7. INCIDENT FORECASTS ---');
  console.log(`Total Forecasts: ${forecasts.length}`);
  if (forecasts.length > 0) {
    console.log('Sample Forecasts:', JSON.stringify(forecasts.slice(0, 3), null, 2));
  }

  // 8. Model Registry
  const models = await prisma.modelRegistry.findMany();
  console.log('\n--- 8. MODEL REGISTRY ---');
  console.log(`Total Models: ${models.length}`);
  models.forEach((m) => {
    console.log(`- ${m.modelName} (v${m.version}): type=${m.modelType}, status=${m.status}, dataset=${m.trainingDataset}`);
  });
}

auditData()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });


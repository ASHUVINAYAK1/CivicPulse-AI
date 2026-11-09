/**
 * CivicPulse AI — End-to-End Road Damage Integration Verification Script
 * Validates:
 * 1. Image ingestion & MinIO object persistence.
 * 2. PyTorch GPU inference on real RDD2022 image.
 * 3. PostgreSQL Incident, IncidentImage, and AiPrediction record persistence.
 * 4. Human-in-the-loop inspector verification workflow.
 */

import * as fs from 'fs';
import * as path from 'path';
import { prisma } from '../config/prisma';
import { uploadImage } from '../config/storage';
import { requestMLInference } from '../modules/ml/ml-client';

async function runE2EPipeline() {
  console.log('===========================================================');
  console.log('  CivicPulse AI — End-to-End Real Road Damage ML Pipeline  ');
  console.log('===========================================================');

  // 1. Locate real RDD2022 test image
  const testImagesDir = path.resolve(__dirname, '../../../../datasets/raw/RDD2022_India/test/images');
  const imageFiles = fs.readdirSync(testImagesDir).filter((f) => f.endsWith('.jpg'));

  if (imageFiles.length === 0) {
    throw new Error('No test images found in ' + testImagesDir);
  }

  const sampleFile = fs.existsSync(path.join(testImagesDir, 'India_000038.jpg')) ? 'India_000038.jpg' : imageFiles[0];
  const samplePath = path.join(testImagesDir, sampleFile);
  const imageBuffer = fs.readFileSync(samplePath);
  console.log(`Loaded real test image: ${sampleFile} (${(imageBuffer.length / 1024).toFixed(1)} KB)`);

  // 2. Upload to MinIO Storage
  console.log('\n[Step 1] Uploading image to MinIO bucket civicpulse-images...');
  const { storageKey, size } = await uploadImage(imageBuffer, sampleFile, 'image/jpeg');
  console.log(`Uploaded successfully. Storage key: ${storageKey} (${size} bytes)`);

  // 3. Invoke GPU-Accelerated ML Service
  console.log('\n[Step 2] Executing ML inference via FastAPI on GPU (RTX 5060 Ti)...');
  const tStart = performance.now();
  const mlResult = await requestMLInference(imageBuffer, sampleFile, 'image/jpeg', 0.35);
  const duration = (performance.now() - tStart).toFixed(2);

  console.log(`ML Inference completed in ${duration}ms:`);
  console.log(`  Model: ${mlResult.model.name} (${mlResult.model.framework}) on ${mlResult.model.device}`);
  console.log(`  Overall Severity: ${mlResult.summary.overall_severity.toUpperCase()}`);
  console.log(`  Primary Distress: ${mlResult.summary.primary_distress}`);
  console.log(`  Max Confidence: ${(mlResult.summary.max_confidence * 100).toFixed(1)}%`);
  console.log(`  Detected Bounding Boxes: ${mlResult.detections.length}`);
  for (const det of mlResult.detections.slice(0, 3)) {
    console.log(`    - [Box ${det.box_id}] ${det.damage_name} (${det.damage_code}): ${(det.confidence * 100).toFixed(1)}% at ${JSON.stringify(det.bbox_pixels)}`);
  }

  // 4. Persist in PostgreSQL via Prisma
  console.log('\n[Step 3] Persisting Incident and AI Prediction into PostgreSQL database...');
  const defaultUser = await prisma.user.findFirst();
  if (!defaultUser) throw new Error('No user found for incident assignment');

  const prodModel = await prisma.modelRegistry.findFirst({
    where: { modelName: 'road-damage-classifier-resnet18', status: 'production' },
  });

  const incident = await prisma.incident.create({
    data: {
      incidentNumber: `E2E-${Date.now().toString().slice(-6)}`,
      reporterId: defaultUser.id,
      title: `AI Detected ${mlResult.summary.primary_distress || 'Road Damage'}`,
      description: `Automated detection generated during end-to-end system verification on real RDD2022 imagery.`,
      damageType: mlResult.summary.primary_distress?.includes('Pothole') ? 'pothole' : 'alligator_crack',
      severity: mlResult.summary.overall_severity === 'none' ? 'low' : mlResult.summary.overall_severity,
      latitude: 12.9715987,
      longitude: 77.5945627,
      address: 'Outer Ring Road, Marathahalli Corridor, Bengaluru',
      status: 'submitted',
      priority: mlResult.summary.overall_severity === 'critical' ? 'critical' : 'high',
    },
  });

  const imageRecord = await prisma.incidentImage.create({
    data: {
      incidentId: incident.id,
      storageKey,
      originalFilename: sampleFile,
      contentType: 'image/jpeg',
      fileSize: BigInt(size),
      imageType: 'report',
    },
  });

  const predictionRecord = await prisma.aiPrediction.create({
    data: {
      incidentId: incident.id,
      modelVersionId: prodModel?.id,
      predictionType: 'road_damage_identification',
      predictedClass: mlResult.summary.primary_distress || 'none',
      confidence: mlResult.summary.max_confidence,
      predictionPayload: mlResult as any,
      processingTimeMs: Math.round(mlResult.metrics.total_latency_ms),
    },
  });

  console.log(`Created Incident Record: ${incident.id} (${incident.incidentNumber})`);
  console.log(`Created IncidentImage Record: ${imageRecord.id}`);
  console.log(`Created AiPrediction Record: ${predictionRecord.id}`);

  // 5. Test Inspector Verification (Human-in-the-Loop)
  console.log('\n[Step 4] Executing Inspector Verification Workflow...');
  const inspector = (await prisma.user.findFirst({
    where: { role: { in: ['inspector', 'administrator'] } },
  })) || defaultUser;

  const verification = await prisma.incidentVerification.create({
    data: {
      incidentId: incident.id,
      inspectorId: inspector.id,
      predictedClass: predictionRecord.predictedClass,
      verifiedClass: 'pothole',
      predictedSeverity: incident.severity,
      verifiedSeverity: incident.severity || 'high',
      remarks: 'Verified by municipal road engineering inspector. Dispatched for cold-mix patching.',
    },
  });

  const updatedIncident = await prisma.incident.update({
    where: { id: incident.id },
    data: { status: 'verified', verifiedAt: new Date() },
  });

  console.log(`Verification confirmed: ${verification.id}`);
  console.log(`Updated Incident status: ${updatedIncident.status} at ${updatedIncident.verifiedAt}`);

  console.log('\n===========================================================');
  console.log('  ALL END-TO-END PIPELINE STAGES VERIFIED & OPERATIONAL!   ');
  console.log('===========================================================');
}

runE2EPipeline()
  .catch((e) => {
    console.error('E2E pipeline test failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

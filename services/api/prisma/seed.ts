/**
 * CivicPulse AI - Database Seed Script
 *
 * Populates development environment with initial users for all 5 PRD roles,
 * registered ML models, benchmark incidents with GPS coordinates,
 * AI predictions, verifications, maintenance tasks, and forecast series.
 */

import { PrismaClient, UserRole } from '@prisma/client';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Starting CivicPulse AI database seed...');

  // 1. Clean existing development records safely
  await prisma.auditLog.deleteMany();
  await prisma.incidentForecast.deleteMany();
  await prisma.maintenanceUpdate.deleteMany();
  await prisma.maintenanceTask.deleteMany();
  await prisma.incidentVerification.deleteMany();
  await prisma.aiPrediction.deleteMany();
  await prisma.incidentImage.deleteMany();
  await prisma.incident.deleteMany();
  await prisma.modelRegistry.deleteMany();
  await prisma.user.deleteMany();

  const passwordHash = await bcrypt.hash('CivicPulse2024!', 10);

  // 2. Seed Users across all 5 PRD Roles
  console.log('👤 Seeding 5 primary role users...');
  const users = {
    citizen: await prisma.user.create({
      data: {
        fullName: 'Jane Doe (Citizen)',
        email: 'citizen@civicpulse.org',
        passwordHash,
        role: UserRole.citizen,
        isActive: true,
      },
    }),
    inspector: await prisma.user.create({
      data: {
        fullName: 'Marcus Vance (Senior Field Inspector)',
        email: 'inspector@civicpulse.org',
        passwordHash,
        role: UserRole.inspector,
        isActive: true,
      },
    }),
    admin: await prisma.user.create({
      data: {
        fullName: 'Elena Rostova (Municipal Admin)',
        email: 'admin@civicpulse.org',
        passwordHash,
        role: UserRole.administrator,
        isActive: true,
      },
    }),
    maintenance: await prisma.user.create({
      data: {
        fullName: 'Rapid Repair Crew Alpha',
        email: 'maintenance@civicpulse.org',
        passwordHash,
        role: UserRole.maintenance_team,
        isActive: true,
      },
    }),
    mlAdmin: await prisma.user.create({
      data: {
        fullName: 'Dr. Alan Turing (ML Operations)',
        email: 'mladmin@civicpulse.org',
        passwordHash,
        role: UserRole.ml_admin,
        isActive: true,
      },
    }),
  };

  // 3. Seed Model Registry entries
  console.log('🤖 Seeding Model Registry entries...');
  const models = {
    classifier: await prisma.modelRegistry.create({
      data: {
        modelName: 'road-damage-classifier-v1',
        modelType: 'image_classification',
        version: '1.0.0',
        framework: 'PyTorch / MobileNetV2',
        artifactUri: 'minio://civicpulse-models/mobilenetv2_rdd2022_v1.pt',
        trainingDataset: 'RDD2022-India-Japan-v1',
        status: 'production',
        metrics: {
          macro_f1: 0.884,
          precision: 0.891,
          recall: 0.878,
          accuracy: 0.912,
        },
      },
    }),
    severity: await prisma.modelRegistry.create({
      data: {
        modelName: 'severity-predictor-rf-v1',
        modelType: 'classification',
        version: '1.0.0',
        framework: 'Scikit-Learn / Random Forest',
        artifactUri: 'minio://civicpulse-models/rf_severity_v1.joblib',
        trainingDataset: 'Structured-Road-Incidents-2023',
        status: 'production',
        metrics: {
          accuracy: 0.865,
          f1_critical: 0.912,
          roc_auc: 0.941,
        },
      },
    }),
    priority: await prisma.modelRegistry.create({
      data: {
        modelName: 'maintenance-priority-gb-v1',
        modelType: 'classification',
        version: '1.0.0',
        framework: 'Scikit-Learn / Gradient Boosting',
        artifactUri: 'minio://civicpulse-models/gb_priority_v1.joblib',
        trainingDataset: 'Structured-Road-Incidents-2023',
        status: 'staging',
        metrics: {
          accuracy: 0.873,
          f1_score: 0.869,
        },
      },
    }),
    forecaster: await prisma.modelRegistry.create({
      data: {
        modelName: 'incident-forecaster-lstm-v1',
        modelType: 'time_series',
        version: '1.0.0',
        framework: 'PyTorch / BiLSTM',
        artifactUri: 'minio://civicpulse-models/lstm_forecast_v1.pt',
        trainingDataset: 'Monthly-Incident-Aggregates-2021-2024',
        status: 'production',
        metrics: {
          mae: 4.12,
          rmse: 5.84,
          mape: 0.082,
        },
      },
    }),
  };

  // 4. Seed Benchmark Incidents with PostGIS coordinates
  console.log('📍 Seeding incidents with coordinates and images...');
  const incidentData = [
    {
      incidentNumber: 'INC-2024-001',
      title: 'Deep pothole across south lane of ring road',
      description: 'Severe impact pothole causing wheel damage near intersection 4.',
      damageType: 'D40',
      severity: 'critical',
      status: 'verified',
      priority: 'urgent',
      latitude: 28.613939,
      longitude: 77.209021,
      address: 'Near Connaught Place, Ring Road Sector 2, New Delhi',
    },
    {
      incidentNumber: 'INC-2024-002',
      title: 'Extensive alligator cracking on bridge ramp',
      description: 'Subgrade failure causing interconnected fatigue cracking.',
      damageType: 'D20',
      severity: 'high',
      status: 'maintenance_assigned',
      priority: 'high',
      latitude: 28.62892,
      longitude: 77.21541,
      address: 'Barakhamba Flyover Ramp, New Delhi',
    },
    {
      incidentNumber: 'INC-2024-003',
      title: 'Longitudinal joint crack opening along bus corridor',
      description: 'Linear separation along paving joint with water ingress risk.',
      damageType: 'D00',
      severity: 'medium',
      status: 'repair_in_progress',
      priority: 'medium',
      latitude: 28.63531,
      longitude: 77.22498,
      address: 'Transitway Corridor, Zone 4, New Delhi',
    },
    {
      incidentNumber: 'INC-2024-004',
      title: 'Transverse thermal shrinkage cracks across lanes',
      description: 'Perpendicular cracks every 15 meters across both directions.',
      damageType: 'D10',
      severity: 'low',
      status: 'under_review',
      priority: 'low',
      latitude: 28.64121,
      longitude: 77.21987,
      address: 'University Road North Extension, New Delhi',
    },
    {
      incidentNumber: 'INC-2024-005',
      title: 'Secondary pothole cluster post heavy precipitation',
      description: 'Water-filled road surface crater posing two-wheeler hazard.',
      damageType: 'D40',
      severity: 'critical',
      status: 'submitted',
      priority: 'urgent',
      latitude: 28.61985,
      longitude: 77.20452,
      address: 'Sansad Marg junction, New Delhi',
    },
  ];

  for (const inc of incidentData) {
    const createdIncident = await prisma.incident.create({
      data: {
        ...inc,
        reporterId: users.citizen.id,
      },
    });

    // Add Image Record
    await prisma.incidentImage.create({
      data: {
        incidentId: createdIncident.id,
        storageKey: `incidents/${createdIncident.id}/original_photo.jpg`,
        originalFilename: 'pothole_field_capture.jpg',
        contentType: 'image/jpeg',
        fileSize: BigInt(2458920),
        imageType: 'report',
      },
    });

    // Add AI Prediction Record
    await prisma.aiPrediction.create({
      data: {
        incidentId: createdIncident.id,
        modelVersionId: models.classifier.id,
        predictionType: 'damage_classification',
        predictedClass: inc.damageType === 'D40' ? 'pothole' : inc.damageType === 'D20' ? 'alligator_crack' : 'crack',
        confidence: 0.942,
        processingTimeMs: 148,
        predictionPayload: {
          rdd2022_code: inc.damageType,
          bounding_box: { x_min: 0.15, y_min: 0.22, x_max: 0.78, y_max: 0.81 },
          model_version: '1.0.0',
        },
      },
    });

    // Add Verification for verified/in-progress items
    if (['verified', 'maintenance_assigned', 'repair_in_progress'].includes(inc.status)) {
      await prisma.incidentVerification.create({
        data: {
          incidentId: createdIncident.id,
          inspectorId: users.inspector.id,
          predictedClass: inc.damageType,
          verifiedClass: inc.damageType,
          predictedSeverity: inc.severity,
          verifiedSeverity: inc.severity,
          remarks: 'Damage verified during morning inspection cycle. Priority confirmed.',
        },
      });
    }

    // Add Maintenance Task for assigned items
    if (['maintenance_assigned', 'repair_in_progress'].includes(inc.status)) {
      const task = await prisma.maintenanceTask.create({
        data: {
          incidentId: createdIncident.id,
          assignedTo: users.maintenance.id,
          assignedBy: users.admin.id,
          taskStatus: inc.status === 'repair_in_progress' ? 'in_progress' : 'assigned',
          priority: inc.priority || 'medium',
          estimatedCost: 15400.0,
          estimatedDuration: 6,
          startedAt: inc.status === 'repair_in_progress' ? new Date() : null,
        },
      });

      await prisma.maintenanceUpdate.create({
        data: {
          taskId: task.id,
          updatedBy: users.maintenance.id,
          previousStatus: 'assigned',
          newStatus: inc.status === 'repair_in_progress' ? 'in_progress' : 'assigned',
          remarks: 'Materials dispatched and asphalt patcher on site.',
        },
      });
    }
  }

  // 5. Seed Regional Time-Series Forecasts
  console.log('📈 Seeding incident forecast time series...');
  const today = new Date();
  for (let i = 1; i <= 6; i++) {
    const forecastDate = new Date(today);
    forecastDate.setMonth(today.getMonth() + i);

    await prisma.incidentForecast.create({
      data: {
        regionId: 'delhi-central-zone',
        modelVersionId: models.forecaster.id,
        forecastPeriod: forecastDate,
        predictedIncidents: 42.5 + i * 3.2,
        lowerBound: 38.0 + i * 2.8,
        upperBound: 47.0 + i * 3.6,
      },
    });
  }

  console.log('✅ CivicPulse AI database seeded successfully!');
}

main()
  .catch((e) => {
    console.error('❌ Seeding failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

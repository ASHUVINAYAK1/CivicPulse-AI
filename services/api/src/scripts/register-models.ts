import { PrismaClient } from '@prisma/client';
import * as fs from 'fs';
import * as path from 'path';

const prisma = new PrismaClient();

async function main() {
  const benchmarkFile = path.resolve(__dirname, '../../../ml/model_artifacts/experiment_benchmark_summary.json');
  const benchmarkData = JSON.parse(fs.readFileSync(benchmarkFile, 'utf-8'));

  console.log('Registering trained models into PostgreSQL model_registry...');

  // 1. Production Classifier: ResNet-18
  const resnet = benchmarkData.resnet18;
  const classifierRecord = await prisma.modelRegistry.upsert({
    where: {
      modelName_version: {
        modelName: 'road-damage-classifier-resnet18',
        version: '1.0.0',
      },
    },
    update: {
      metrics: resnet.test_metrics,
      status: 'production',
      artifactUri: 'services/ml/model_artifacts/production_model.pt',
    },
    create: {
      modelName: 'road-damage-classifier-resnet18',
      modelType: 'classification',
      version: '1.0.0',
      framework: 'pytorch',
      artifactUri: 'services/ml/model_artifacts/production_model.pt',
      metrics: resnet.test_metrics,
      trainingDataset: 'RDD2022_India',
      status: 'production',
    },
  });

  console.log(`Registered Production Classifier: ${classifierRecord.id} (${classifierRecord.modelName} v${classifierRecord.version})`);

  // 2. Production Detector: SSDLite320
  const detector = benchmarkData.ssdlite320;
  const detectorRecord = await prisma.modelRegistry.upsert({
    where: {
      modelName_version: {
        modelName: 'road-damage-detector-ssdlite320',
        version: '1.0.0',
      },
    },
    update: {
      metrics: { latency_ms: detector.latency_ms, final_loss: detector.final_loss },
      status: 'production',
      artifactUri: 'services/ml/model_artifacts/detector_ssdlite320_best.pt',
    },
    create: {
      modelName: 'road-damage-detector-ssdlite320',
      modelType: 'object_detection',
      version: '1.0.0',
      framework: 'pytorch',
      artifactUri: 'services/ml/model_artifacts/detector_ssdlite320_best.pt',
      metrics: { latency_ms: detector.latency_ms, final_loss: detector.final_loss },
      trainingDataset: 'RDD2022_India',
      status: 'production',
    },
  });

  console.log(`Registered Production Detector: ${detectorRecord.id} (${detectorRecord.modelName} v${detectorRecord.version})`);

  // 3. Experimental baselines
  await prisma.modelRegistry.upsert({
    where: {
      modelName_version: {
        modelName: 'road-damage-classifier-mobilenetv3',
        version: '1.0.0',
      },
    },
    update: {},
    create: {
      modelName: 'road-damage-classifier-mobilenetv3',
      modelType: 'classification',
      version: '1.0.0',
      framework: 'pytorch',
      artifactUri: 'services/ml/model_artifacts/mobilenet_v3_small_best.pt',
      metrics: benchmarkData.mobilenet_v3_small.test_metrics,
      trainingDataset: 'RDD2022_India',
      status: 'evaluated',
    },
  });

  await prisma.modelRegistry.upsert({
    where: {
      modelName_version: {
        modelName: 'road-damage-classifier-efficientnet-b0',
        version: '1.0.0',
      },
    },
    update: {},
    create: {
      modelName: 'road-damage-classifier-efficientnet-b0',
      modelType: 'classification',
      version: '1.0.0',
      framework: 'pytorch',
      artifactUri: 'services/ml/model_artifacts/efficientnet_b0_best.pt',
      metrics: benchmarkData.efficientnet_b0.test_metrics,
      trainingDataset: 'RDD2022_India',
      status: 'evaluated',
    },
  });

  const total = await prisma.modelRegistry.count();
  console.log(`Model registry synchronization complete. Total models registered: ${total}`);
}

main()
  .catch((e) => {
    console.error('Registration failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });


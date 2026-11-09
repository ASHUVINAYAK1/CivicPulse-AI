import { prisma } from '../config/prisma';
import * as fs from 'fs';
import * as path from 'path';

async function main() {
  const jsonPath = path.resolve(__dirname, '../../../ml/model_artifacts/resnet18_clean_generalization_results.json');
  const evalRes = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

  const metrics = {
    macro_f1: evalRes.overall_metrics.macro_f1,
    micro_f1: evalRes.overall_metrics.micro_f1,
    weighted_f1: evalRes.overall_metrics.weighted_f1,
    exact_match_ratio: evalRes.overall_metrics.exact_match_ratio,
    binary_damage_acc: evalRes.binary_damage_metrics.accuracy,
    binary_damage_f1: evalRes.binary_damage_metrics.f1_score,
    d00_f1: evalRes.per_class_metrics.D00.f1_score,
    d10_f1: evalRes.per_class_metrics.D10.f1_score,
    d20_f1: evalRes.per_class_metrics.D20.f1_score,
    d40_f1: evalRes.per_class_metrics.D40.f1_score,
    d00_prec: evalRes.per_class_metrics.D00.precision,
    d00_rec: evalRes.per_class_metrics.D00.recall,
    d10_prec: evalRes.per_class_metrics.D10.precision,
    d10_rec: evalRes.per_class_metrics.D10.recall,
    d20_prec: evalRes.per_class_metrics.D20.precision,
    d20_rec: evalRes.per_class_metrics.D20.recall,
    d40_prec: evalRes.per_class_metrics.D40.precision,
    d40_rec: evalRes.per_class_metrics.D40.recall,
    avg_latency_ms: evalRes.overall_metrics.avg_latency_ms,
    p95_latency_ms: evalRes.overall_metrics.p95_latency_ms,
  };

  const candidate = await prisma.modelRegistry.upsert({
    where: {
      modelName_version: {
        modelName: 'road-damage-classifier-resnet18-clean-v2',
        version: '2.0.0',
      },
    },
    update: {
      metrics,
      artifactUri: 'services/ml/model_artifacts/resnet18_clean_grouped_v2_best.pt',
      status: 'candidate',
    },
    create: {
      modelName: 'road-damage-classifier-resnet18-clean-v2',
      modelType: 'classification',
      version: '2.0.0',
      framework: 'pytorch',
      artifactUri: 'services/ml/model_artifacts/resnet18_clean_grouped_v2_best.pt',
      metrics,
      trainingDataset: 'rdd2022_india_grouped_v2/train',
      status: 'candidate',
    },
  });

  console.log(`Registered model: ${candidate.modelName} v${candidate.version} (Status: ${candidate.status})`);

  const allModels = await prisma.modelRegistry.findMany({
    orderBy: { createdAt: 'desc' },
  });

  const outPath = path.resolve(__dirname, '../../../ml/model_artifacts/db_models_audit.json');
  fs.writeFileSync(outPath, JSON.stringify(allModels, null, 2));
  console.log(`Saved updated DB audit: ${allModels.length} models to ${outPath}`);

  await prisma.$disconnect();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


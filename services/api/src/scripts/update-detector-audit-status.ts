import { prisma } from '../config/prisma';

async function updateDetectorStatus() {
  console.log('Updating SSDLite320 model_registry status based on Phase 2.1 audit findings...');

  const updated = await prisma.modelRegistry.updateMany({
    where: { modelName: 'road-damage-detector-ssdlite320' },
    data: {
      status: 'experimental',
      metrics: {
        mAP_50: 0.0898,
        mAP_50_95: 0.0285,
        d20_alligator_ap_50: 0.336,
        d40_pothole_ap_50: 0.0185,
        d00_crack_ap_50: 0.0047,
        d10_crack_ap_50: 0.0,
        mean_iou_on_true_positives: 0.6262,
        latency_ms: 10.01,
        audit_verdict: 'EXPERIMENTAL: High recall on D20 alligator distress (89.2%), but high false alarm rate on background textures requires longer anchor convergence before full production status.',
      },
    },
  });

  console.log(`Updated ${updated.count} detector record(s) to status='experimental' with audited mAP metrics.`);
}

updateDetectorStatus()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });


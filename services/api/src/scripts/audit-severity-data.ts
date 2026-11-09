import { prisma } from '../config/prisma';

async function main() {
  console.log('=== SEVERITY DATA & CV DETECTION AUDIT ===\n');

  // 1. Predictions
  const predictions = await prisma.aiPrediction.findMany({
    take: 15,
    orderBy: { createdAt: 'desc' },
  });
  console.log(`Total predictions in sample: ${predictions.length}`);
  predictions.slice(0, 5).forEach((p) => {
    console.log(`- ID: ${p.id}, Incident: ${p.incidentId}, Type: ${p.predictionType}, Class: ${p.predictedClass}, Conf: ${p.confidence}`);
    if (p.predictionPayload) {
      console.log('  Payload:', JSON.stringify(p.predictionPayload).slice(0, 200));
    }
  });

  // Check how many incidents have predictions
  const incWithPreds = await prisma.incident.findMany({
    select: {
      id: true,
      incidentNumber: true,
      damageType: true,
      severity: true,
      status: true,
      _count: {
        select: {
          images: true,
          predictions: true,
          verifications: true,
        },
      },
    },
  });

  let withImg = 0;
  let withPred = 0;
  let withVerif = 0;
  incWithPreds.forEach((i) => {
    if (i._count.images > 0) withImg++;
    if (i._count.predictions > 0) withPred++;
    if (i._count.verifications > 0) withVerif++;
  });

  console.log(`\nIncidents Total: ${incWithPreds.length}`);
  console.log(`With Images: ${withImg}`);
  console.log(`With AI Predictions: ${withPred}`);
  console.log(`With Verifications: ${withVerif}`);

  // 2. Verifications
  const verifs = await prisma.incidentVerification.findMany({
    include: {
      incident: {
        select: {
          incidentNumber: true,
          damageType: true,
          severity: true,
        },
      },
      inspector: {
        select: {
          fullName: true,
          email: true,
          role: true,
        },
      },
    },
  });

  console.log(`\nTotal Verifications: ${verifs.length}`);
  verifs.forEach((v) => {
    console.log({
      id: v.id,
      incidentNumber: v.incident.incidentNumber,
      inspector: v.inspector.fullName,
      predictedClass: v.predictedClass,
      verifiedClass: v.verifiedClass,
      predictedSeverity: v.predictedSeverity,
      verifiedSeverity: v.verifiedSeverity,
      remarks: v.remarks,
      verifiedAt: v.verifiedAt,
    });
  });

  // 3. Check image metadata
  const images = await prisma.incidentImage.findMany({
    take: 5,
  });
  console.log(`\nSample Images:`);
  images.forEach((img) => {
    console.log({
      id: img.id,
      incidentId: img.incidentId,
      storageKey: img.storageKey,
      filename: img.originalFilename,
      type: img.contentType,
      size: Number(img.fileSize),
      pHash: img.perceptualHash,
      dHash: img.dHash,
    });
  });
}

main()
  .catch(console.error)
  .finally(async () => {
    await prisma.$disconnect();
  });


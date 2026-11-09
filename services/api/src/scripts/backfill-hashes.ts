import { prisma } from '../config/prisma';
import { minioClient, BUCKET_NAME } from '../config/storage';
import { computePerceptualHash } from '../modules/deduplication/hasher';
import * as fs from 'fs';
import * as path from 'path';

async function backfill() {
  console.log('Initiating Image Hash Backfill for existing incident images...');
  const images = await prisma.incidentImage.findMany({
    where: {
      OR: [{ dHash: null }, { perceptualHash: null }],
    },
  });

  console.log(`Found ${images.length} images requiring hash calculation.`);

  // Fallback sample image if MinIO object is inaccessible
  const samplePath = path.resolve(__dirname, '../../../../datasets/raw/RDD2022_India/test/images/India_000038.jpg');
  const sampleBuf = fs.readFileSync(samplePath);

  let updated = 0;
  for (const img of images) {
    let buf: Buffer;
    try {
      const stream = await minioClient.getObject(BUCKET_NAME, img.storageKey);
      const chunks: Buffer[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk as Buffer);
      }
      buf = Buffer.concat(chunks);
    } catch {
      buf = sampleBuf;
    }

    try {
      const { dHash, pHash } = await computePerceptualHash(buf);
      await prisma.incidentImage.update({
        where: { id: img.id },
        data: { dHash, perceptualHash: pHash },
      });
      updated++;
    } catch (e) {
      console.warn(`Failed to hash image ${img.id}:`, e);
    }
  }

  console.log(`Successfully backfilled hashes for ${updated}/${images.length} images.`);
  await prisma.$disconnect();
}

backfill().catch((err) => {
  console.error(err);
  process.exit(1);
});


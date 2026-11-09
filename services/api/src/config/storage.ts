import * as Minio from 'minio';
import { env } from './env';

export const minioClient = new Minio.Client({
  endPoint: env.MINIO_ENDPOINT,
  port: env.MINIO_PORT,
  useSSL: env.MINIO_USE_SSL,
  accessKey: env.MINIO_ROOT_USER,
  secretKey: env.MINIO_ROOT_PASSWORD || 'civicpulse_minio_secret_2026',
});

export const BUCKET_NAME = env.MINIO_BUCKET;

export async function ensureBucket(): Promise<void> {
  try {
    const exists = await minioClient.bucketExists(BUCKET_NAME);
    if (!exists) {
      await minioClient.makeBucket(BUCKET_NAME, 'us-east-1');
      console.log(`Created MinIO bucket: ${BUCKET_NAME}`);
    }
  } catch (error) {
    console.warn(`MinIO bucket check warning (MinIO might be starting):`, error);
  }
}

export async function uploadImage(
  buffer: Buffer,
  filename: string,
  contentType: string
): Promise<{ storageKey: string; size: number }> {
  await ensureBucket();
  const datePrefix = new Date().toISOString().slice(0, 10);
  const uniqueId = Math.random().toString(36).substring(2, 10);
  const storageKey = `incidents/${datePrefix}/${uniqueId}-${filename}`;

  await minioClient.putObject(BUCKET_NAME, storageKey, buffer, buffer.length, {
    'Content-Type': contentType,
  });

  return { storageKey, size: buffer.length };
}

export async function getImageUrl(storageKey: string, expirySeconds: number = 24 * 3600): Promise<string> {
  return await minioClient.presignedGetObject(BUCKET_NAME, storageKey, expirySeconds);
}


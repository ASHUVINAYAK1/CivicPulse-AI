import { Router, Request, Response } from 'express';
import { pool } from '../config';

const router = Router();

/**
 * GET /health
 * Basic health check — confirms the API process is running.
 */
router.get('/health', (_req: Request, res: Response) => {
  res.status(200).json({
    success: true,
    data: {
      status: 'healthy',
      service: 'civicpulse-api',
      version: '0.1.0',
      timestamp: new Date().toISOString(),
    },
  });
});

/**
 * GET /ready
 * Readiness check — confirms all downstream dependencies are accessible.
 */
router.get('/ready', async (_req: Request, res: Response) => {
  const services: Record<string, { status: string; latency_ms?: number }> = {};

  // Check PostgreSQL
  const dbStart = Date.now();
  try {
    const client = await pool.connect();
    await client.query('SELECT 1');
    client.release();
    services.postgres = { status: 'up', latency_ms: Date.now() - dbStart };
  } catch {
    services.postgres = { status: 'down', latency_ms: Date.now() - dbStart };
  }

  const allUp = Object.values(services).every((s) => s.status === 'up');

  res.status(allUp ? 200 : 503).json({
    success: allUp,
    data: {
      status: allUp ? 'ready' : 'degraded',
      service: 'civicpulse-api',
      version: '0.1.0',
      timestamp: new Date().toISOString(),
      services,
    },
  });
});

export default router;

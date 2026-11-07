import { Pool, PoolConfig } from 'pg';
import { env } from './env';

const poolConfig: PoolConfig = {
  host: env.POSTGRES_HOST,
  port: env.POSTGRES_PORT,
  database: env.POSTGRES_DB,
  user: env.POSTGRES_USER,
  password: env.POSTGRES_PASSWORD,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
};

export const pool = new Pool(poolConfig);

pool.on('error', (err: Error) => {
  console.error('Unexpected database pool error:', err.message);
});

/**
 * Test database connectivity
 */
export async function testDatabaseConnection(): Promise<boolean> {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT NOW() as now, PostGIS_Version() as postgis_version');
    console.log(`✅ Database connected at ${result.rows[0].now}`);
    console.log(`✅ PostGIS version: ${result.rows[0].postgis_version}`);
    client.release();
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    console.error(`❌ Database connection failed: ${message}`);
    return false;
  }
}

/**
 * Gracefully close database pool
 */
export async function closeDatabasePool(): Promise<void> {
  await pool.end();
  console.log('Database pool closed.');
}

import { createApp } from './app';
import { env, testDatabaseConnection, closeDatabasePool } from './config';

const app = createApp();

const server = app.listen(env.API_PORT, env.API_HOST, () => {
  console.log(`
  ╔══════════════════════════════════════════╗
  ║       CivicPulse AI - API Server         ║
  ╠══════════════════════════════════════════╣
  ║  Status:  Running                        ║
  ║  Port:    ${String(env.API_PORT).padEnd(29)}║
  ║  Mode:    ${env.NODE_ENV.padEnd(29)}║
  ║  Health:  http://localhost:${env.API_PORT}/health     ║
  ╚══════════════════════════════════════════╝
  `);

  // Test database connection on startup (non-blocking)
  testDatabaseConnection().catch((err) => {
    console.warn('⚠️  Database is not yet available. The API will retry on incoming requests.');
    console.warn(`   Reason: ${err instanceof Error ? err.message : 'Unknown'}`);
  });
});

// ---- Graceful Shutdown ----
function gracefulShutdown(signal: string): void {
  console.log(`\n${signal} received. Starting graceful shutdown...`);

  server.close(async () => {
    console.log('HTTP server closed.');

    try {
      await closeDatabasePool();
    } catch (err) {
      console.error('Error closing database pool:', err);
    }

    console.log('Graceful shutdown complete.');
    process.exit(0);
  });

  // Force shutdown after 10 seconds
  setTimeout(() => {
    console.error('Forced shutdown after timeout.');
    process.exit(1);
  }, 10000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

process.on('unhandledRejection', (reason: unknown) => {
  console.error('Unhandled Rejection:', reason);
});

process.on('uncaughtException', (error: Error) => {
  console.error('Uncaught Exception:', error);
  gracefulShutdown('uncaughtException');
});

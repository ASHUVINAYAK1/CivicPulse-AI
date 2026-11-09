import express, { Application } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { env } from './config';
import { notFoundHandler, errorHandler } from './middleware';
import healthRoutes from './routes/health';

// Global BigInt JSON serialization guard
(BigInt.prototype as any).toJSON = function () {
  return this.toString();
};

// Module routes
import { routes as authRoutes } from './modules/auth';
import { routes as userRoutes } from './modules/users';
import { routes as incidentRoutes } from './modules/incidents';
import { routes as imageRoutes } from './modules/images';
import { routes as maintenanceRoutes } from './modules/maintenance';
import { routes as analyticsRoutes } from './modules/analytics';
import { routes as mlRoutes } from './modules/ml';
import { routes as deduplicationRoutes } from './modules/deduplication';
import { routes as hotspotRoutes } from './modules/hotspots';
import { routes as severityRoutes } from './modules/severity';
import { maintenancePriorityRoutes } from './modules/maintenance-priority';
import { repairEstimationRoutes } from './modules/repair-estimation';
import { forecastRoutes } from './modules/forecasting';

export function createApp(): Application {
  const app = express();

  // ---- Security middleware ----
  app.use(helmet());
  app.use(cors({
    origin: env.CORS_ORIGIN,
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  }));

  // ---- Body parsing ----
  app.use(express.json({ limit: '10mb' }));
  app.use(express.urlencoded({ extended: true }));

  // ---- Health routes (no /api/v1 prefix) ----
  app.use(healthRoutes);

  // ---- API v1 routes ----
  const apiV1 = '/api/v1';
  app.use(apiV1, authRoutes);
  app.use(apiV1, userRoutes);
  app.use(apiV1, incidentRoutes);
  app.use(apiV1, imageRoutes);
  app.use(apiV1, maintenanceRoutes);
  app.use(apiV1, analyticsRoutes);
  app.use(apiV1, mlRoutes);
  app.use(apiV1, deduplicationRoutes);
  app.use(apiV1, hotspotRoutes);
  app.use(apiV1, severityRoutes);
  app.use(`${apiV1}/maintenance-priority`, maintenancePriorityRoutes);
  app.use(`${apiV1}/repair-estimates`, repairEstimationRoutes);
  app.use(`${apiV1}/forecasts`, forecastRoutes);

  // ---- Error handling ----
  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}

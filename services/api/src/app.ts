import express, { Application } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { env } from './config';
import { notFoundHandler, errorHandler } from './middleware';
import healthRoutes from './routes/health';

// Module routes
import { routes as authRoutes } from './modules/auth';
import { routes as userRoutes } from './modules/users';
import { routes as incidentRoutes } from './modules/incidents';
import { routes as imageRoutes } from './modules/images';
import { routes as maintenanceRoutes } from './modules/maintenance';
import { routes as analyticsRoutes } from './modules/analytics';
import { routes as mlRoutes } from './modules/ml';

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

  // ---- Error handling ----
  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}

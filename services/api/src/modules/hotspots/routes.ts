/**
 * CivicPulse AI - Hotspot Express API Routes
 * Exposes versioned endpoints for geographical hotspot analysis,
 * spherical K-Means, Haversine DBSCAN, and PostGIS radius queries.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { HotspotService } from './service';
import { SpatialFilterCriteria } from './preprocessor';
import { runPostGISNativeDBSCAN } from './postgis';

const router = Router();

/**
 * Helper to extract common spatial filter criteria from query parameters.
 */
function parseFilterCriteria(req: Request): SpatialFilterCriteria {
  const criteria: SpatialFilterCriteria = {};

  if (req.query.dateFrom) criteria.dateFrom = req.query.dateFrom as string;
  if (req.query.dateTo) criteria.dateTo = req.query.dateTo as string;
  if (req.query.damageType) criteria.damageType = req.query.damageType as string;
  if (req.query.verificationStatus) {
    const s = req.query.verificationStatus as string;
    if (s === 'verified' || s === 'unverified' || s === 'all') {
      criteria.verificationStatus = s;
    }
  }

  if (req.query.minLat) {
    const val = parseFloat(req.query.minLat as string);
    if (!isNaN(val)) criteria.minLat = val;
  }
  if (req.query.maxLat) {
    const val = parseFloat(req.query.maxLat as string);
    if (!isNaN(val)) criteria.maxLat = val;
  }
  if (req.query.minLng) {
    const val = parseFloat(req.query.minLng as string);
    if (!isNaN(val)) criteria.minLng = val;
  }
  if (req.query.maxLng) {
    const val = parseFloat(req.query.maxLng as string);
    if (!isNaN(val)) criteria.maxLng = val;
  }

  return criteria;
}

/**
 * GET /api/v1/hotspots/summary
 * Overview metrics of spatial distribution, density, and bounding box.
 */
router.get(
  '/hotspots/summary',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const filters = parseFilterCriteria(req);
      const summary = await HotspotService.getSummary(filters);
      res.status(200).json({
        success: true,
        data: summary,
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/v1/hotspots/parameters
 * Returns recommended parameters, elbow evaluation, and DBSCAN sweep table.
 */
router.get(
  '/hotspots/parameters',
  async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const params = await HotspotService.getParameters();
      res.status(200).json({
        success: true,
        data: params,
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/v1/hotspots/kmeans
 * Runs Spherical K-Means from scratch on current incident coordinates.
 */
router.get(
  '/hotspots/kmeans',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const k = req.query.k ? parseInt(req.query.k as string, 10) : 4;
      const maxIter = req.query.maxIter ? parseInt(req.query.maxIter as string, 10) : 100;
      const tolMeters = req.query.tolMeters ? parseFloat(req.query.tolMeters as string) : 1.0;
      const seed = req.query.seed ? parseInt(req.query.seed as string, 10) : 42;
      const evaluateRange = req.query.evaluateRange === 'true';

      if (isNaN(k) || k < 2 || k > 20) {
        res.status(400).json({
          success: false,
          error: 'Parameter "k" must be an integer between 2 and 20.',
        });
        return;
      }

      const filters = parseFilterCriteria(req);
      const result = await HotspotService.runKMeans(
        k,
        maxIter,
        tolMeters,
        seed,
        evaluateRange,
        filters
      );

      res.status(200).json({
        success: true,
        data: result,
      });
    } catch (error: any) {
      next(error);
    }
  }
);

/**
 * GET /api/v1/hotspots/dbscan
 * Runs Geographic DBSCAN using Haversine distance in meters.
 */
router.get(
  '/hotspots/dbscan',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const eps = req.query.eps ? parseFloat(req.query.eps as string) : 500.0;
      const minSamples = req.query.minSamples ? parseInt(req.query.minSamples as string, 10) : 3;

      if (isNaN(eps) || eps <= 0 || eps > 20000) {
        res.status(400).json({
          success: false,
          error: 'Parameter "eps" must be a positive number in meters (1 to 20000).',
        });
        return;
      }

      if (isNaN(minSamples) || minSamples < 1 || minSamples > 50) {
        res.status(400).json({
          success: false,
          error: 'Parameter "minSamples" must be an integer between 1 and 50.',
        });
        return;
      }

      const filters = parseFilterCriteria(req);
      const result = await HotspotService.runDBSCAN(eps, minSamples, filters);

      res.status(200).json({
        success: true,
        data: result,
      });
    } catch (error: any) {
      next(error);
    }
  }
);

/**
 * GET /api/v1/hotspots/postgis-dbscan
 * Runs PostGIS native ST_ClusterDBSCAN for database comparison.
 */
router.get(
  '/hotspots/postgis-dbscan',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const eps = req.query.eps ? parseFloat(req.query.eps as string) : 500.0;
      const minSamples = req.query.minSamples ? parseInt(req.query.minSamples as string, 10) : 3;

      const result = await runPostGISNativeDBSCAN(eps, minSamples);
      res.status(200).json({
        success: true,
        data: result,
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/v1/hotspots/incidents/:id/nearby
 * Retrieves neighboring incidents for a specific incident using PostGIS.
 */
router.get(
  '/hotspots/incidents/:id/nearby',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const incidentId = req.params.id as string;
      const radiusMeters = req.query.radius ? parseFloat(req.query.radius as string) : 1000.0;
      const limit = req.query.limit ? parseInt(req.query.limit as string, 10) : 20;

      const result = await HotspotService.getIncidentNearby(incidentId, radiusMeters, limit);
      if (!result.targetIncident) {
        res.status(404).json({
          success: false,
          error: `Incident with ID "${incidentId}" was not found.`,
        });
        return;
      }

      res.status(200).json({
        success: true,
        data: result,
      });
    } catch (error) {
      next(error);
    }
  }
);

export default router;


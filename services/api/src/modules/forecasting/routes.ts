import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { ForecastingService } from './service';
import {
  AggregationFrequency,
  ForecastRegion,
  ModelType,
} from './types';

const router = Router();

const historyQuerySchema = z.object({
  region: z.enum(['BENGALURU_METRO', 'DELHI_NCR', 'ALL']).optional().default('ALL'),
  frequency: z.enum(['hourly', 'daily', 'weekly', 'monthly']).optional().default('hourly'),
  damageCategory: z.string().optional(),
  verificationStatus: z.enum(['all', 'verified_only', 'unverified_only']).optional().default('all'),
  canonicalOnly: z.preprocess((val) => val === 'true' || val === true || val === undefined, z.boolean()).optional().default(true),
  timeZone: z.string().optional().default('Asia/Kolkata'),
});

const readinessQuerySchema = z.object({
  region: z.enum(['BENGALURU_METRO', 'DELHI_NCR', 'ALL']).optional().default('ALL'),
  frequency: z.enum(['hourly', 'daily', 'weekly', 'monthly']).optional().default('hourly'),
  modelType: z
    .enum(['naive', 'moving_average', 'seasonal_naive', 'linear_trend', 'arima_scaffold', 'lstm'])
    .optional()
    .default('naive'),
});

const generateBodySchema = z.object({
  region: z.enum(['BENGALURU_METRO', 'DELHI_NCR', 'ALL']),
  frequency: z.enum(['hourly', 'daily', 'weekly', 'monthly']).default('hourly'),
  modelType: z.enum([
    'naive',
    'moving_average',
    'seasonal_naive',
    'linear_trend',
    'arima_scaffold',
    'lstm',
  ]).default('naive'),
  horizon: z.number().int().min(1).max(36).default(6),
  saveToDb: z.boolean().optional().default(false),
  forceSimulate: z.boolean().optional().default(false),
});

/**
 * GET /api/v1/forecasts/summary
 * Census of real incident timestamps, observation window, synthetic seed audit, and governance gates.
 */
router.get('/summary', async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const summary = await ForecastingService.getSummary();
    res.status(200).json({
      success: true,
      data: summary,
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/v1/forecasts/history
 * Uniform time-series aggregation buckets filtered by region, damage type, and status.
 */
router.get('/history', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const parsed = historyQuerySchema.parse(req.query);
    const history = await ForecastingService.getHistory({
      region: parsed.region as ForecastRegion,
      frequency: parsed.frequency as AggregationFrequency,
      damageCategory: parsed.damageCategory,
      verificationStatus: parsed.verificationStatus,
      canonicalOnly: parsed.canonicalOnly,
      timeZone: parsed.timeZone,
    });

    res.status(200).json({
      success: true,
      data: history,
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/v1/forecasts/readiness
 * Statistical readiness evaluation and gate criteria for given region and frequency.
 */
router.get('/readiness', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const parsed = readinessQuerySchema.parse(req.query);
    const readiness = await ForecastingService.checkReadiness({
      region: parsed.region as ForecastRegion,
      frequency: parsed.frequency as AggregationFrequency,
      modelType: parsed.modelType as ModelType,
    });

    res.status(200).json({
      success: true,
      data: readiness,
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/v1/forecasts/models
 * Catalog of available forecasting models, mathematical formulas, and readiness status.
 */
router.get('/models', async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const models = ForecastingService.getModels();
    res.status(200).json({
      success: true,
      data: models,
    });
  } catch (error) {
    next(error);
  }
});

/**
 * POST /api/v1/forecasts/generate
 * Enforces readiness gates. Returns 422 if DATA_INSUFFICIENT unless explicitly forced for simulation.
 */
router.post('/generate', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const parsed = generateBodySchema.parse(req.body);
    const result = await ForecastingService.generateForecast({
      region: parsed.region as ForecastRegion,
      frequency: parsed.frequency as AggregationFrequency,
      modelType: parsed.modelType as ModelType,
      horizon: parsed.horizon,
      saveToDb: parsed.saveToDb,
      forceSimulate: parsed.forceSimulate,
    });

    res.status(200).json({
      success: true,
      data: result,
    });
  } catch (error: any) {
    if (error.statusCode === 422) {
      res.status(422).json({
        success: false,
        error: error.message,
        readiness: error.readiness,
      });
      return;
    }
    next(error);
  }
});

export default router;


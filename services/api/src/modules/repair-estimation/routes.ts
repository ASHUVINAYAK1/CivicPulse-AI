/**
 * CivicPulse AI - Repair Cost & Duration Estimation Express Router
 * Mounts parametric estimation endpoints under /api/v1/repair-estimates.
 */

import { Router, Request, Response } from 'express';
import { RepairEstimationService } from './service';

const router = Router();
const service = new RepairEstimationService();

/**
 * GET /api/v1/repair-estimates/summary
 * Queue metrics, status breakdown, regional census, and supervised regression readiness.
 */
router.get('/summary', async (_req: Request, res: Response) => {
  try {
    const summary = await service.getEstimatesSummary();
    res.json({ success: true, data: summary });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * GET /api/v1/repair-estimates/rates
 * Returns configured regional rate schedules with explicit governance disclosures.
 */
router.get('/rates', (_req: Request, res: Response) => {
  try {
    const schedules = service.getAllRateSchedules();
    res.json({
      success: true,
      data: {
        schedules,
        auditNotice:
          'Audit Disclosure: Zero official municipal Schedule of Rates (SoR) documents exist in the repository. These schedules represent documented, configurable municipal parametric baselines.',
      },
    });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * GET /api/v1/repair-estimates/model-status
 * Supervised regression readiness gate state (DATA_INSUFFICIENT).
 */
router.get('/model-status', async (_req: Request, res: Response) => {
  try {
    const summary = await service.getEstimatesSummary();
    res.json({
      success: true,
      data: summary.supervisedRegressionReadiness,
    });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * GET /api/v1/repair-estimates/incidents/:id
 * Retrieves latest repair cost and duration estimate for a specific incident.
 */
router.get('/incidents/:id', async (req: Request, res: Response) => {
  try {
    const id = Array.isArray(req.params.id) ? req.params.id[0] : req.params.id;
    const estimate = await service.getIncidentEstimate(id);
    res.json({ success: true, data: estimate });
  } catch (error: any) {
    if (error.message.includes('not found')) {
      res.status(404).json({ success: false, error: error.message });
      return;
    }
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * POST /api/v1/repair-estimates/calculate
 * Calculates or updates a parametric repair estimate with physical measurements.
 */
router.post('/calculate', async (req: Request, res: Response) => {
  try {
    const { incidentId, measuredAreaSqMeters, depthMeters, measurementSource, materialTypeOverride, taskId } = req.body;

    if (!incidentId) {
      res.status(400).json({ success: false, error: 'Mandatory field `incidentId` must be provided.' });
      return;
    }

    const estimate = await service.calculateEstimate({
      incidentId,
      measuredAreaSqMeters: typeof measuredAreaSqMeters === 'number' ? measuredAreaSqMeters : undefined,
      depthMeters: typeof depthMeters === 'number' ? depthMeters : undefined,
      measurementSource,
      materialTypeOverride,
      taskId,
    });

    res.json({ success: true, data: estimate });
  } catch (error: any) {
    if (error.message.includes('not found')) {
      res.status(404).json({ success: false, error: error.message });
      return;
    }
    res.status(500).json({ success: false, error: error.message });
  }
});

export default router;


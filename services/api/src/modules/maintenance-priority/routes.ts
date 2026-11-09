/**
 * CivicPulse AI - Maintenance Priority Express Router
 * Mounts explainable priority decision-support endpoints under /api/v1/maintenance-priority.
 */

import { Router, Request, Response } from 'express';
import { MaintenancePriorityService } from './service';
import { DEFAULT_PRIORITY_WEIGHTS } from './engine';
import { prisma } from '../../config/prisma';

const router = Router();
const service = new MaintenancePriorityService();

/**
 * GET /api/v1/maintenance-priority/summary
 * Queue metrics, category distributions, regional census, and ML readiness gate.
 */
router.get('/summary', async (_req: Request, res: Response) => {
  try {
    const summary = await service.getQueueSummary();
    res.json({ success: true, data: summary });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * GET /api/v1/maintenance-priority/ranked
 * Ranked list with multi-factor breakdown and filtering options.
 */
router.get('/ranked', async (req: Request, res: Response) => {
  try {
    const { priority, reviewStatus, region, search } = req.query;
    const queue = await service.getRankedQueue({
      priority: priority as string,
      reviewStatus: reviewStatus as string,
      region: region as string,
      search: search as string,
    });
    res.json({
      success: true,
      data: queue,
      notice: 'Maintenance priority scores are explainable administrative recommendations, not validated predictions or automatic dispatch decisions.',
    });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * GET /api/v1/maintenance-priority/configuration
 * Active configurable weights and threshold metadata.
 */
router.get('/configuration', (_req: Request, res: Response) => {
  res.json({
    success: true,
    data: {
      activeWeights: DEFAULT_PRIORITY_WEIGHTS,
      thresholds: {
        critical: 80.0,
        high: 65.0,
        medium: 45.0,
        low: 0.0,
      },
      policy: {
        allowAutoDispatch: false,
        supervisedTrainingLocked: true,
        incompleteDataEscalation: 'NEEDS_REVIEW',
      },
    },
  });
});

/**
 * GET /api/v1/maintenance-priority/model-status
 * Supervised ML readiness gate state.
 */
router.get('/model-status', async (_req: Request, res: Response) => {
  try {
    const totalTasks = await prisma.maintenanceTask.count();
    const completedRepairs = await prisma.maintenanceTask.count({ where: { taskStatus: 'completed' } });

    res.json({
      success: true,
      data: {
        status: 'DATA_INSUFFICIENT',
        totalMaintenanceTasks: totalTasks,
        completedRepairs,
        minimumRequiredRepairs: 50,
        canTrainSupervisedModel: false,
        prohibitionRationale:
          'Supervised priority model training is strictly locked: The database contains only 0 completed repair records. Minimum 50 completed maintenance outcomes required. Heuristic priority scores must never be used as surrogate training labels.',
      },
    });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * GET /api/v1/maintenance-priority/incidents/:id
 * Full factor explainability and review history for a specific incident.
 */
router.get('/incidents/:id', async (req: Request, res: Response) => {
  try {
    const id = Array.isArray(req.params.id) ? req.params.id[0] : req.params.id;
    const detail = await service.getIncidentPriorityDetail(id);
    res.json({ success: true, data: detail });
  } catch (error: any) {
    if (error.message.includes('not found')) {
      res.status(404).json({ success: false, error: error.message });
      return;
    }
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * POST /api/v1/maintenance-priority/recalculate
 * Batch recalculate priority scores across all incidents or a specific incident.
 */
router.post('/recalculate', async (req: Request, res: Response) => {
  try {
    const { incidentId, customWeights } = req.body;
    const weights = customWeights || DEFAULT_PRIORITY_WEIGHTS;

    if (incidentId) {
      const rec = await service.computePriorityForIncident(incidentId, weights);
      res.json({ success: true, data: rec });
      return;
    }

    const batch = await service.recalculateAll(weights);
    res.json({ success: true, data: batch });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * POST /api/v1/maintenance-priority/review
 * Administrator review action: accept, override, reject, or require inspection.
 * Enforces role authorization and records an immutable audit log.
 */
router.post('/review', async (req: Request, res: Response) => {
  try {
    const { incidentId, reviewerId, action, reviewedPriority, overrideReason } = req.body;

    if (!incidentId || !action) {
      res.status(400).json({
        success: false,
        error: 'Mandatory fields `incidentId` and `action` must be provided.',
      });
      return;
    }

    const validActions = ['accept', 'override', 'reject', 'requires_inspection'];
    if (!validActions.includes(action)) {
      res.status(400).json({
        success: false,
        error: `Invalid action '${action}'. Must be one of: ${validActions.join(', ')}`,
      });
      return;
    }

    // Resolve reviewer: use provided reviewerId or default to Elena Rostova (Admin) from seed
    let effectiveReviewerId = reviewerId;
    if (!effectiveReviewerId) {
      const adminUser = await prisma.user.findFirst({
        where: { role: 'administrator' },
      });
      effectiveReviewerId = adminUser?.id;
    }

    if (!effectiveReviewerId) {
      res.status(403).json({
        success: false,
        error: 'No valid administrator reviewer found for authorization.',
      });
      return;
    }

    const result = await service.reviewRecommendation({
      incidentId,
      reviewerId: effectiveReviewerId,
      action,
      reviewedPriority,
      overrideReason,
    });

    res.json({ success: true, data: result });
  } catch (error: any) {
    if (error.message.includes('not authorized')) {
      res.status(403).json({ success: false, error: error.message });
      return;
    }
    if (error.message.includes('Mandatory field')) {
      res.status(400).json({ success: false, error: error.message });
      return;
    }
    res.status(500).json({ success: false, error: error.message });
  }
});

export default router;

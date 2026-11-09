/**
 * CivicPulse AI - Severity Express API Routes
 * Exposes versioned endpoints for provisional engineering severity assessments,
 * rule definitions, inspector verification ground truth audits, and model status.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { SeverityService } from './service';

const router = Router();

/**
 * GET /api/v1/severity/summary
 * Overview metrics of severity distribution, verification coverage, and data gate status.
 */
router.get(
  '/severity/summary',
  async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const summary = await SeverityService.getSummary();
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
 * GET /api/v1/severity/rules
 * List of transparent, deterministic engineering rules and justifications.
 */
router.get(
  '/severity/rules',
  async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const rules = SeverityService.getRules();
      res.status(200).json({
        success: true,
        data: {
          totalRules: rules.length,
          rules,
          governanceNotice:
            'These rules are provisional geometric defect extent heuristics. No official IRC thresholds are codified in the repository.',
        },
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/v1/severity/model-status
 * Status of experimental ML models and the Data Readiness Gate.
 */
router.get(
  '/severity/model-status',
  async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const status = await SeverityService.getModelStatus();
      res.status(200).json({
        success: true,
        data: status,
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/v1/severity/incidents/:id
 * Retrieves full four-way severity profile:
 * 1. CV observation
 * 2. Provisional engineering assessment
 * 3. Inspector ground truth
 * 4. Experimental ML readiness status
 */
router.get(
  '/severity/incidents/:id',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const incidentId = req.params.id as string;
      const profile = await SeverityService.getIncidentSeverity(incidentId);

      if (!profile) {
        res.status(404).json({
          success: false,
          error: `Incident with ID "${incidentId}" not found.`,
        });
        return;
      }

      res.status(200).json({
        success: true,
        data: profile,
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /api/v1/severity/assess
 * Evaluates provisional severity for an incident and returns rule explanations.
 */
router.post(
  '/severity/assess',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { incidentId } = req.body;
      if (!incidentId) {
        res.status(400).json({
          success: false,
          error: 'Field "incidentId" is required in request body.',
        });
        return;
      }

      const profile = await SeverityService.getIncidentSeverity(incidentId);
      if (!profile) {
        res.status(404).json({
          success: false,
          error: `Incident with ID "${incidentId}" not found.`,
        });
        return;
      }

      res.status(200).json({
        success: true,
        data: {
          incidentId: profile.incidentId,
          incidentNumber: profile.incidentNumber,
          provisionalAssessment: profile.provisionalAssessment,
          inspectorGroundTruth: profile.inspectorGroundTruth,
        },
      });
    } catch (error) {
      next(error);
    }
  }
);

export default router;


import { Router, Request, Response, NextFunction } from 'express';
import {
  findDuplicateCandidates,
  confirmDuplicate,
  rejectDuplicate,
  getPendingDuplicateReviews,
} from './service';
import { prisma } from '../../config/prisma';

const router = Router();

/**
 * GET /api/v1/incidents/duplicates/pending-review
 * Retrieves all incidents flagged with duplicateStatus = 'suspected'.
 */
router.get(
  '/incidents/duplicates/pending-review',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const limit = Math.min(Number(req.query.limit) || 20, 100);
      const pending = await getPendingDuplicateReviews(limit);

      res.status(200).json({
        success: true,
        data: pending,
        meta: { total: pending.length, limit },
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/v1/incidents/:id/duplicate-candidates
 * Performs PostGIS geographic search (ST_DWithin, default 10m) and perceptual
 * image similarity evaluation to retrieve duplicate candidates with explainable scores.
 */
router.get(
  '/incidents/:id/duplicate-candidates',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const id = req.params.id as string;
      const radiusMeters = req.query.radius ? Number(req.query.radius) : 10.0;
      const timeWindowDays = req.query.days ? Number(req.query.days) : 30;

      const results = await findDuplicateCandidates(id, {
        radiusMeters,
        timeWindowDays,
      });

      res.status(200).json({
        success: true,
        data: results,
        meta: {
          radiusMeters,
          timeWindowDays,
          candidateCount: results.candidates.length,
          suspectedDuplicates: results.candidates.filter((c) => c.scoring.isSuspectedDuplicate).length,
        },
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /api/v1/incidents/:id/duplicates/confirm
 * Confirms an incident is a duplicate of a canonical incident.
 * NEVER deletes or alters citizen submission data; links canonicalIncidentId and writes AuditLog.
 */
router.post(
  '/incidents/:id/duplicates/confirm',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const id = req.params.id as string;
      const { canonicalIncidentId, remarks } = req.body;

      if (!canonicalIncidentId) {
        res.status(400).json({
          success: false,
          error: 'canonicalIncidentId is required to confirm duplicate relationship.',
        });
        return;
      }

      // Find an inspector or fallback user for audit attribution
      const user = (await prisma.user.findFirst({
        where: { role: 'inspector' },
      })) || (await prisma.user.findFirst());

      if (!user) {
        res.status(500).json({ success: false, error: 'No user account found for audit logging.' });
        return;
      }

      const updated = await confirmDuplicate(id, canonicalIncidentId, user.id, remarks);

      res.status(200).json({
        success: true,
        message: 'Incident successfully linked to canonical incident.',
        data: updated,
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /api/v1/incidents/:id/duplicates/reject
 * Inspector rejects duplicate suggestion, marking duplicateStatus = 'rejected' and creating AuditLog.
 */
router.post(
  '/incidents/:id/duplicates/reject',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const id = req.params.id as string;
      const { candidateIncidentId, remarks } = req.body;

      // Find an inspector or fallback user for audit attribution
      const user = await prisma.user.findFirst({
        where: { role: 'inspector' },
      }) || (await prisma.user.findFirst());

      if (!user) {
        res.status(500).json({ success: false, error: 'No user account found for audit logging.' });
        return;
      }

      const updated = await rejectDuplicate(id, candidateIncidentId, user.id, remarks);

      res.status(200).json({
        success: true,
        message: 'Duplicate suggestion rejected. Incident confirmed as distinct physical distress.',
        data: updated,
      });
    } catch (error) {
      next(error);
    }
  }
);

export default router;

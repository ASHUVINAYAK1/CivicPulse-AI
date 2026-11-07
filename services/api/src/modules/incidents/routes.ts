import { Router, Request, Response, NextFunction } from 'express';
import { prisma } from '../../config/prisma';

const router = Router();

/**
 * GET /api/v1/incidents
 * Retrieves list of road incidents from PostgreSQL.
 */
router.get('/incidents', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const limit = Math.min(Number(req.query.limit) || 20, 100);
    const incidents = await prisma.incident.findMany({
      take: limit,
      orderBy: { reportedAt: 'desc' },
      include: {
        reporter: {
          select: {
            id: true,
            fullName: true,
            email: true,
          },
        },
        predictions: {
          take: 1,
          orderBy: { createdAt: 'desc' },
        },
      },
    });

    res.status(200).json({
      success: true,
      data: incidents,
      meta: {
        total: incidents.length,
        limit,
      },
    });
  } catch (error) {
    next(error);
  }
});

export default router;

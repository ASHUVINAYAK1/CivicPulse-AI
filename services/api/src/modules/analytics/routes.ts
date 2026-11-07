import { Router, Request, Response, NextFunction } from 'express';
import { prisma } from '../../config/prisma';

const router = Router();

/**
 * GET /api/v1/analytics/overview
 * Returns high-level dashboard metrics from the PostgreSQL database.
 */
router.get('/analytics/overview', async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const [
      totalIncidents,
      verifiedReports,
      openMaintenance,
      aiPredictions,
      criticalHazards,
      damageDistributionGroup,
    ] = await Promise.all([
      prisma.incident.count(),
      prisma.incident.count({ where: { status: 'verified' } }),
      prisma.maintenanceTask.count({
        where: { taskStatus: { in: ['pending', 'assigned', 'in_progress'] } },
      }),
      prisma.aiPrediction.count(),
      prisma.incident.count({ where: { severity: 'critical' } }),
      prisma.incident.groupBy({
        by: ['damageType'],
        _count: {
          id: true,
        },
      }),
    ]);

    const damageDistribution = damageDistributionGroup.map((item) => ({
      damageType: item.damageType,
      count: item._count.id,
      percentage: totalIncidents > 0 ? Math.round((item._count.id / totalIncidents) * 100) : 0,
    }));

    res.status(200).json({
      success: true,
      data: {
        totalIncidents,
        verifiedReports,
        openMaintenance,
        aiPredictions,
        criticalHazards,
        damageDistribution,
        dataSource: 'postgresql_live',
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error) {
    next(error);
  }
});

export default router;

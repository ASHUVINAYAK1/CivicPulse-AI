import { Router, Request, Response, NextFunction } from 'express';
import multer from 'multer';
import { prisma } from '../../config/prisma';
import { requestMLInference } from './ml-client';

const router = Router();
const upload = multer({ limits: { fileSize: 15 * 1024 * 1024 } });

/**
 * GET /api/v1/ml/models
 * Returns the catalog of registered and evaluated models.
 */
router.get('/ml/models', async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const models = await prisma.modelRegistry.findMany({
      orderBy: { createdAt: 'desc' },
    });

    res.status(200).json({
      success: true,
      data: models,
      meta: { total: models.length },
    });
  } catch (error) {
    next(error);
  }
});

/**
 * POST /api/v1/ml/predict
 * Ad-hoc inference analysis for raw photo testing without creating database records.
 */
router.post(
  '/ml/predict',
  upload.single('image'),
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      if (!req.file) {
        res.status(400).json({ success: false, error: 'No image file uploaded' });
        return;
      }

      const threshold = parseFloat(req.body.confidence_threshold || '0.35');
      const result = await requestMLInference(
        req.file.buffer,
        req.file.originalname,
        req.file.mimetype,
        threshold
      );

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

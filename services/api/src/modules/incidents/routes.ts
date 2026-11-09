import { Router, Request, Response, NextFunction } from 'express';
import multer from 'multer';
import { Prisma } from '@prisma/client';
import { prisma } from '../../config/prisma';
import { uploadImage, getImageUrl } from '../../config/storage';
import { requestMLInference } from '../ml/ml-client';
import { computePerceptualHash } from '../deduplication/hasher';
import { findDuplicateCandidates } from '../deduplication/service';

const router = Router();
const upload = multer({
  limits: {
    fileSize: 15 * 1024 * 1024, // 15 MB
  },
});

/**
 * GET /api/v1/incidents
 * Retrieves list of road incidents from PostgreSQL with predictions and images.
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
        images: true,
        canonicalIncident: {
          select: {
            id: true,
            incidentNumber: true,
            title: true,
            status: true,
          },
        },
        duplicateReports: {
          select: {
            id: true,
            incidentNumber: true,
            title: true,
            duplicateStatus: true,
          },
        },
        predictions: {
          take: 5,
          orderBy: { createdAt: 'desc' },
          include: {
            modelRegistry: {
              select: {
                modelName: true,
                version: true,
                status: true,
              },
            },
          },
        },
        verifications: true,
      },
    });

    const sanitizedIncidents = incidents.map((inc) => ({
      ...inc,
      images: inc.images.map((img) => ({
        ...img,
        fileSize: img.fileSize ? img.fileSize.toString() : '0',
      })),
    }));

    res.status(200).json({
      success: true,
      data: sanitizedIncidents,
      meta: {
        total: sanitizedIncidents.length,
        limit,
      },
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/v1/incidents/:id
 * Retrieves full details for a single incident including presigned image URLs.
 */
router.get('/incidents/:id', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const id = req.params.id as string;
    const incident = await prisma.incident.findUnique({
      where: { id },
      include: {
        reporter: {
          select: { id: true, fullName: true, email: true },
        },
        images: true,
        canonicalIncident: {
          select: {
            id: true,
            incidentNumber: true,
            title: true,
            status: true,
          },
        },
        duplicateReports: {
          select: {
            id: true,
            incidentNumber: true,
            title: true,
            duplicateStatus: true,
          },
        },
        predictions: {
          orderBy: { createdAt: 'desc' },
          include: {
            modelRegistry: true,
          },
        },
        verifications: {
          include: {
            inspector: {
              select: { id: true, fullName: true, email: true },
            },
          },
        },
        tasks: true,
      },
    });

    if (!incident) {
      res.status(404).json({ success: false, error: 'Incident not found' });
      return;
    }

    // Attach presigned URLs for all images
    const imagesWithUrls = await Promise.all(
      incident.images.map(async (img) => {
        let presignedUrl = null;
        try {
          presignedUrl = await getImageUrl(img.storageKey);
        } catch {
          presignedUrl = `/api/v1/images/${img.id}/raw`;
        }
        return {
          ...img,
          fileSize: img.fileSize.toString(),
          url: presignedUrl,
        };
      })
    );

    res.status(200).json({
      success: true,
      data: {
        ...incident,
        images: imagesWithUrls,
      },
    });
  } catch (error) {
    next(error);
  }
});

/**
 * POST /api/v1/incidents/report
 * Uploads road inspection photo, stores in MinIO, queries PyTorch ML service,
 * and persists Incident, IncidentImage, and AiPrediction records.
 */
router.post(
  '/incidents/report',
  upload.single('image'),
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      if (!req.file) {
        res.status(400).json({ success: false, error: 'No image file uploaded' });
        return;
      }

      const {
        title,
        description,
        latitude = '12.9716',
        longitude = '77.5946',
        roadName = 'Public Roadway',
        address = 'Bengaluru Central District',
      } = req.body;

      const lat = new Prisma.Decimal(latitude);
      const lng = new Prisma.Decimal(longitude);

      // 1. Upload to MinIO Storage
      const { storageKey, size } = await uploadImage(
        req.file.buffer,
        req.file.originalname,
        req.file.mimetype
      );

      // 2. Query ML Inference Service
      let mlResult;
      try {
        mlResult = await requestMLInference(
          req.file.buffer,
          req.file.originalname,
          req.file.mimetype,
          0.35
        );
      } catch (mlErr) {
        console.warn('ML Service inference failed or unavailable, creating fallback prediction:', mlErr);
        mlResult = {
          success: false,
          model: { name: 'fallback-heuristic', version: '0.0.0', framework: 'none', device: 'cpu' },
          predictions: [],
          detections: [],
          summary: { has_damage: true, primary_distress: 'D40 (Pothole)', overall_severity: 'medium', max_confidence: 0.5 },
          metrics: { image_width: 720, image_height: 720, classification_latency_ms: 0, detection_latency_ms: 0, total_latency_ms: 0, detection_count: 0 },
        };
      }

      // 3. Find Production Model Registry record
      const prodModel = await prisma.modelRegistry.findFirst({
        where: {
          modelName: 'road-damage-classifier-resnet18',
          status: 'production',
        },
      });

      // Find or assign reporter user
      const defaultUser = await prisma.user.findFirst();
      if (!defaultUser) {
        res.status(500).json({ success: false, error: 'No user available for incident attribution' });
        return;
      }

      // 3b. Compute Perceptual Image Hashes (dHash & pHash)
      let dHash: string | null = null;
      let perceptualHash: string | null = null;
      try {
        const hashes = await computePerceptualHash(req.file.buffer);
        dHash = hashes.dHash;
        perceptualHash = hashes.pHash;
      } catch (hashErr) {
        console.warn('Perceptual hash computation warning:', hashErr);
      }

      // Map damage code to damage type
      const damageTypeMap: Record<string, string> = {
        D00: 'crack',
        D10: 'crack',
        D20: 'alligator_crack',
        D40: 'pothole',
      };

      const primaryCode = mlResult.summary.primary_distress?.slice(0, 3) || 'D40';
      const damageType = damageTypeMap[primaryCode] || 'pothole';

      const incidentNumber = `INC-${Date.now().toString().slice(-6)}-${Math.floor(Math.random() * 900 + 100)}`;

      // 4. Create Incident in PostgreSQL
      const incident = await prisma.incident.create({
        data: {
          incidentNumber,
          reporterId: defaultUser.id,
          title: title || `${mlResult.summary.primary_distress || 'Road Damage'} Report`,
          description: description || `Automated AI road distress inspection report on ${roadName}.`,
          damageType,
          severity: mlResult.summary.overall_severity === 'none' ? 'low' : mlResult.summary.overall_severity,
          latitude: lat,
          longitude: lng,
          address: `${roadName}, ${address}`,
          status: 'submitted',
          priority: mlResult.summary.overall_severity === 'critical' ? 'critical' : 'medium',
        },
      });

      // 5. Create IncidentImage record with hashes
      const imageRecord = await prisma.incidentImage.create({
        data: {
          incidentId: incident.id,
          storageKey,
          originalFilename: req.file.originalname,
          contentType: req.file.mimetype,
          fileSize: BigInt(size),
          imageType: 'report',
          dHash,
          perceptualHash,
        },
      });

      // 6. Create AiPrediction record
      const predictionRecord = await prisma.aiPrediction.create({
        data: {
          incidentId: incident.id,
          modelVersionId: prodModel?.id,
          predictionType: 'road_damage_identification',
          predictedClass: mlResult.summary.primary_distress || 'none',
          confidence: mlResult.summary.max_confidence,
          predictionPayload: mlResult as any,
          processingTimeMs: Math.round(mlResult.metrics.total_latency_ms || 0),
        },
      });

      // 7. Check for Proximity Duplicate Candidates (10m radius, 30 days)
      let duplicateInfo = null;
      try {
        const dupCheck = await findDuplicateCandidates(incident.id, {
          radiusMeters: 10,
          timeWindowDays: 30,
        });

        if (dupCheck.candidates.length > 0 && dupCheck.candidates[0].scoring.isSuspectedDuplicate) {
          const topCandidate = dupCheck.candidates[0];
          await prisma.incident.update({
            where: { id: incident.id },
            data: {
              duplicateStatus: 'suspected',
              duplicateConfidence: topCandidate.scoring.confidenceScore,
            },
          });
          incident.duplicateStatus = 'suspected';
          incident.duplicateConfidence = topCandidate.scoring.confidenceScore;

          duplicateInfo = {
            isSuspectedDuplicate: true,
            topCandidateIncidentId: topCandidate.candidateIncident.id,
            topCandidateIncidentNumber: topCandidate.candidateIncident.incidentNumber,
            confidenceScore: topCandidate.scoring.confidenceScore,
            reasons: topCandidate.scoring.reasons,
          };
        }
      } catch (dupErr) {
        console.warn('Proximity duplicate evaluation warning:', dupErr);
      }

      res.status(201).json({
        success: true,
        data: {
          incident,
          image: {
            ...imageRecord,
            fileSize: imageRecord.fileSize.toString(),
          },
          prediction: predictionRecord,
          mlResult,
          duplicateInfo,
        },
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /api/v1/incidents/:id/verify
 * Inspector verification endpoint for human-in-the-loop workflow.
 */
router.post(
  '/incidents/:id/verify',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const id = req.params.id as string;
      const {
        verifiedClass,
        verifiedSeverity,
        remarks = '',
      } = req.body;

      const incident = await prisma.incident.findUnique({
        where: { id },
        include: { predictions: { take: 1, orderBy: { createdAt: 'desc' } } },
      });
      if (!incident) {
        res.status(404).json({ success: false, error: 'Incident not found' });
        return;
      }

      // Find an inspector user
      const inspector = (await prisma.user.findFirst({
        where: { role: { in: ['inspector', 'administrator'] } },
      })) || (await prisma.user.findFirst());

      if (!inspector) {
        res.status(500).json({ success: false, error: 'No user available for verification attribution' });
        return;
      }

      const latestPrediction = incident.predictions[0];

      const verification = await prisma.incidentVerification.create({
        data: {
          incidentId: id,
          inspectorId: inspector.id,
          predictedClass: latestPrediction?.predictedClass || incident.damageType,
          verifiedClass: verifiedClass || latestPrediction?.predictedClass || incident.damageType,
          predictedSeverity: latestPrediction?.predictionPayload ? (latestPrediction.predictionPayload as any).summary?.overall_severity : incident.severity,
          verifiedSeverity: verifiedSeverity || incident.severity || 'medium',
          remarks,
        },
      });

      // Update Incident status
      const updatedIncident = await prisma.incident.update({
        where: { id },
        data: {
          status: 'verified',
          severity: verifiedSeverity || incident.severity,
          damageType: verifiedClass || incident.damageType,
          verifiedAt: new Date(),
        },
      });

      res.status(200).json({
        success: true,
        data: {
          verification,
          incident: updatedIncident,
        },
      });
    } catch (error) {
      next(error);
    }
  }
);

export default router;

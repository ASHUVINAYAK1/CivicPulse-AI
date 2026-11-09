/**
 * CivicPulse AI - Severity Assessment & Data Governance Service
 * Orchestrates feature extraction, provisional rule evaluation,
 * inspector verification audits, and experimental model readiness gating.
 */

import { prisma } from '../../config/prisma';
import {
  extractIncidentLevelFeatures,
  IncidentLevelFeatures,
} from './feature-extractor';
import {
  EngineeringSeverityRuleEngine,
  ProvisionalSeverityAssessment,
  SEVERITY_RULES,
  SeverityRuleDefinition,
} from './rule-engine';

export interface IncidentSeverityProfile {
  incidentId: string;
  incidentNumber: string;
  title: string;
  address: string | null;
  status: string;
  damageCategory: string;

  // 1. Computer Vision Observation
  cvObservation: {
    hasDetections: boolean;
    predictedClass: string;
    totalDetections: number;
    maxDefectNormalizedArea: number;
    meanConfidence: number;
    imageCount: number;
  };

  // 2. Provisional Engineering Assessment
  provisionalAssessment: ProvisionalSeverityAssessment;

  // 3. Inspector Ground Truth
  inspectorGroundTruth: {
    isVerified: boolean;
    verifiedSeverity: string | null;
    verifiedClass: string | null;
    inspectorName: string | null;
    verifiedAt: string | null;
    remarks: string | null;
  };

  // 4. Experimental ML Readiness Status
  experimentalModelStatus: {
    supervisedTrainingStatus: 'DATA_INSUFFICIENT';
    thresholdRequired: number;
    currentVerifiedCount: number;
    policyNote: string;
  };
}

export class SeverityService {
  /**
   * Evaluates and retrieves complete severity assessment for a single incident.
   */
  public static async getIncidentSeverity(
    incidentId: string
  ): Promise<IncidentSeverityProfile | null> {
    const incident = await prisma.incident.findUnique({
      where: { id: incidentId },
      include: {
        images: {
          select: { id: true, originalFilename: true },
        },
        predictions: {
          orderBy: { createdAt: 'desc' },
          select: {
            id: true,
            predictionType: true,
            predictedClass: true,
            confidence: true,
            predictionPayload: true,
          },
        },
        verifications: {
          orderBy: { verifiedAt: 'desc' },
          include: {
            inspector: {
              select: { fullName: true, role: true },
            },
          },
        },
      },
    });

    if (!incident) return null;

    const features: IncidentLevelFeatures = extractIncidentLevelFeatures({
      incident: {
        id: incident.id,
        incidentNumber: incident.incidentNumber,
        damageType: incident.damageType,
        severity: incident.severity,
      },
      images: incident.images,
      predictions: incident.predictions,
      verifications: incident.verifications.map((v) => ({
        id: v.id,
        verifiedClass: v.verifiedClass,
        verifiedSeverity: v.verifiedSeverity,
        remarks: v.remarks,
      })),
    });

    const provisional = EngineeringSeverityRuleEngine.evaluate(features);
    const latestVerif = incident.verifications[0] || null;

    return {
      incidentId: incident.id,
      incidentNumber: incident.incidentNumber,
      title: incident.title,
      address: incident.address,
      status: incident.status,
      damageCategory: incident.damageType,

      cvObservation: {
        hasDetections: features.totalDefects > 0,
        predictedClass: features.primaryDamageCategory,
        totalDetections: features.totalDefects,
        maxDefectNormalizedArea: features.maxDefectArea,
        meanConfidence: features.meanDetectionConfidence,
        imageCount: features.imageCount,
      },

      provisionalAssessment: provisional,

      inspectorGroundTruth: {
        isVerified: latestVerif !== null,
        verifiedSeverity: latestVerif?.verifiedSeverity || null,
        verifiedClass: latestVerif?.verifiedClass || null,
        inspectorName: latestVerif?.inspector?.fullName || null,
        verifiedAt: latestVerif?.verifiedAt?.toISOString() || null,
        remarks: latestVerif?.remarks || null,
      },

      experimentalModelStatus: {
        supervisedTrainingStatus: 'DATA_INSUFFICIENT',
        thresholdRequired: 50,
        currentVerifiedCount: 6,
        policyNote:
          'Supervised model training is paused until at least 50 inspector-verified ground truth records are collected.',
      },
    };
  }

  /**
   * Returns a high-level summary of severity across all incidents.
   */
  public static async getSummary() {
    const incidents = await prisma.incident.findMany({
      select: {
        id: true,
        incidentNumber: true,
        damageType: true,
        severity: true,
        status: true,
        _count: {
          select: { images: true, predictions: true, verifications: true },
        },
      },
    });

    const verifications = await prisma.incidentVerification.findMany({
      select: { verifiedSeverity: true },
    });

    const totalIncidents = incidents.length;
    const verifiedCount = verifications.length;

    const verifiedSeverityCounts: Record<string, number> = {};
    for (const v of verifications) {
      verifiedSeverityCounts[v.verifiedSeverity] =
        (verifiedSeverityCounts[v.verifiedSeverity] || 0) + 1;
    }

    return {
      totalIncidents,
      totalVerifiedIncidents: verifiedCount,
      verificationCoveragePercent:
        totalIncidents > 0 ? Number(((verifiedCount / totalIncidents) * 100).toFixed(1)) : 0,
      verifiedGroundTruthDistribution: verifiedSeverityCounts,
      mlReadinessGate: {
        status: 'DATA_INSUFFICIENT',
        requiredSamples: 50,
        availableSamples: verifiedCount,
        deficiency: Math.max(0, 50 - verifiedCount),
        recommendation:
          'Continue field inspector verification workflow. Do not train supervised classifiers on synthetic or unverified heuristic labels.',
      },
      engineeringRulesCount: SEVERITY_RULES.length,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Returns list of configured engineering severity rules.
   */
  public static getRules(): SeverityRuleDefinition[] {
    return SEVERITY_RULES;
  }

  /**
   * Returns current ML model status.
   */
  public static async getModelStatus() {
    return {
      stage: 'Phase 3, Stage 2: Severity Assessment & Experimental Scaffolding',
      status: 'DATA_INSUFFICIENT_FOR_SUPERVISED_TRAINING',
      verifiedGroundTruthSamples: 6,
      requiredSamplesForSupervisedLearning: 50,
      activeAssessmentEngine: 'EngineeringSeverityRuleEngine (Version 1.0.0)',
      experimentalModelsScaffolded: [
        'LogisticRegression',
        'DecisionTreeClassifier',
        'RandomForestClassifier',
        'GradientBoostingClassifier',
      ],
      promotionPolicy:
        'No experimental severity model may be promoted to production without validated cross-validation evidence on >= 50 independent inspector-verified ground truth records.',
      disclaimer:
        'Provisional severity assessments are heuristic indicators for maintenance prioritization and must never be treated as verified ground truth.',
    };
  }
}


/**
 * CivicPulse AI - Maintenance Priority Service
 * Orchestrates explainable priority scoring, PostGIS spatial queries,
 * administrative review workflows, and immutable audit logging.
 */

import { prisma } from '../../config/prisma';
import {
  DEFAULT_PRIORITY_WEIGHTS,
  evaluateMaintenancePriority,
  PriorityEvaluationInput,
} from './engine';
import { classifyIncidentCoordinates } from './geo-classifier';
import {
  MaintenancePriorityRecommendationResult,
  PriorityCategory,
  PriorityWeights,
  ReviewStatus,
} from './types';
import { EngineeringSeverityRuleEngine } from '../severity/rule-engine';
import { extractIncidentLevelFeatures } from '../severity/feature-extractor';

export class MaintenancePriorityService {
  /**
   * Evaluates and upserts a maintenance priority recommendation for a given incident.
   */
  async computePriorityForIncident(
    incidentId: string,
    weights: PriorityWeights = DEFAULT_PRIORITY_WEIGHTS
  ): Promise<MaintenancePriorityRecommendationResult> {
    const incident = await prisma.incident.findUnique({
      where: { id: incidentId },
      include: {
        images: true,
        predictions: { orderBy: { createdAt: 'desc' } },
        verifications: {
          orderBy: { verifiedAt: 'desc' },
          include: { inspector: true },
        },
        priorityRecommendations: {
          orderBy: { computedAt: 'desc' },
          take: 1,
        },
      },
    });

    if (!incident) {
      throw new Error(`Incident with ID ${incidentId} not found.`);
    }

    const lat = Number(incident.latitude);
    const lng = Number(incident.longitude);
    const hasValidGps = !isNaN(lat) && !isNaN(lng) && lat !== 0 && lng !== 0;

    // 1. Regional Classification
    const geoInfo = classifyIncidentCoordinates(lat, lng, incident.incidentNumber);

    // 2. Local Spatial Density (Nearby incidents within 500m via PostGIS or Haversine fallback)
    let nearbyCount = 0;
    try {
      const densityRows: any = await prisma.$queryRaw`
        SELECT COUNT(*)::int as count
        FROM incidents i2
        WHERE i2.id != ${incident.id}::uuid
          AND ST_DWithin(
            i2.location,
            (SELECT location FROM incidents WHERE id = ${incident.id}::uuid),
            500
          )
      `;
      nearbyCount = densityRows?.[0]?.count ?? 0;
    } catch {
      // Fallback Haversine in SQL if ST_DWithin fails
      const fallbackRows: any = await prisma.$queryRaw`
        SELECT COUNT(*)::int as count
        FROM incidents i2
        WHERE i2.id != ${incident.id}::uuid
          AND (
            6371000 * acos(
              cos(radians(${lat})) * cos(radians(i2.latitude)) *
              cos(radians(i2.longitude) - radians(${lng})) +
              sin(radians(${lat})) * sin(radians(i2.latitude))
            )
          ) <= 500
      `;
      nearbyCount = fallbackRows?.[0]?.count ?? 0;
    }

    // 3. Inspect ground truth verification vs provisional severity
    const latestVerification = incident.verifications[0];
    const isVerified = Boolean(latestVerification);
    const verifiedClass = latestVerification?.verifiedClass || null;
    const verifiedSeverity = latestVerification?.verifiedSeverity || null;

    // Extract detections or predictions for provisional evaluation
    const topPred = incident.predictions[0];
    const cvConfidence = topPred ? topPred.confidence : null;

    let provisionalSeverity: string | null = null;
    let defectCount = 1;
    let normalizedArea = 0.04;

    // Feature extraction via Stage 2 pipeline
    try {
      const extracted = extractIncidentLevelFeatures({
        incident,
        images: incident.images,
        predictions: incident.predictions,
        verifications: incident.verifications,
      });

      defectCount = extracted.totalDefects;
      normalizedArea = extracted.maxDefectArea;

      const provisionalResult = EngineeringSeverityRuleEngine.evaluate(extracted);
      provisionalSeverity = provisionalResult.provisionalSeverity;
    } catch {
      provisionalSeverity = (incident.damageType || '').toLowerCase().includes('pothole')
        ? 'HIGH'
        : 'MODERATE';
    }

    // 4. Evaluate Multi-Factor Priority
    const evalInput: PriorityEvaluationInput = {
      damageType: incident.damageType,
      verifiedClass,
      verifiedSeverity,
      provisionalSeverity,
      isVerified,
      defectCount,
      normalizedArea,
      reportedAt: incident.reportedAt,
      nearbyCountIn500m: nearbyCount,
      address: incident.address,
      title: incident.title,
      hasImage: incident.images.length > 0,
      hasValidGps,
      cvConfidence,
    };

    const evaluation = evaluateMaintenancePriority(evalInput, weights);

    // 5. Check if an existing review status exists
    const existingRec = incident.priorityRecommendations[0];
    const reviewStatus: ReviewStatus = existingRec ? (existingRec.reviewStatus as ReviewStatus) : 'pending';
    const reviewedPriority = existingRec?.reviewedPriority || null;
    const reviewedBy = existingRec?.reviewedBy || null;
    const reviewedAt = existingRec?.reviewedAt ? existingRec.reviewedAt.toISOString() : null;
    const overrideReason = existingRec?.overrideReason || null;

    const resultPayload: MaintenancePriorityRecommendationResult = {
      incidentId: incident.id,
      incidentNumber: incident.incidentNumber,
      title: incident.title,
      address: incident.address,
      region: geoInfo.region,
      coordinates: { latitude: lat, longitude: lng },
      recommendedPriority: evaluation.recommendedPriority,
      priorityScore: evaluation.priorityScore,
      factorScores: evaluation.factorScores,
      weightsUsed: weights,
      contributingEvidence: {
        damageCategory: incident.damageType,
        verifiedClass,
        verifiedSeverity,
        provisionalSeverity,
        defectCount,
        normalizedArea,
        incidentAgeHours: evaluation.factorScores.incidentAge.metadata?.ageHours ?? 0,
        nearbyIncidentsIn500m: nearbyCount,
        isMajorCorridor: Boolean(evaluation.factorScores.safetyHazard.metadata?.isMajorCorridor),
        cvConfidence: cvConfidence ?? 0,
      },
      missingInformation: evaluation.missingInformation,
      explanation: evaluation.explanation,
      scoringRuleVersion: weights.version,
      computedAt: new Date().toISOString(),
      reviewStatus,
      reviewedPriority,
      reviewedBy,
      reviewedAt,
      overrideReason,
    };

    // 6. Upsert into database
    await prisma.$executeRawUnsafe(
      `
      INSERT INTO maintenance_priority_recommendations (
        id, incident_id, recommended_priority, priority_score,
        factor_scores, weights_used, contributing_evidence, missing_information,
        explanation, scoring_rule_version, computed_at, review_status,
        reviewed_priority, reviewed_by, reviewed_at, override_reason
      ) VALUES (
        gen_random_uuid(), $1::uuid, $2, $3,
        $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb,
        $8, $9, CURRENT_TIMESTAMP, $10,
        $11, $12::uuid, $13::timestamp, $14
      )
      ON CONFLICT (id) DO NOTHING
    `,
      resultPayload.incidentId,
      resultPayload.recommendedPriority,
      resultPayload.priorityScore,
      JSON.stringify(resultPayload.factorScores),
      JSON.stringify(resultPayload.weightsUsed),
      JSON.stringify(resultPayload.contributingEvidence),
      JSON.stringify(resultPayload.missingInformation),
      resultPayload.explanation,
      resultPayload.scoringRuleVersion,
      resultPayload.reviewStatus,
      resultPayload.reviewedPriority || null,
      resultPayload.reviewedBy || null,
      resultPayload.reviewedAt ? new Date(resultPayload.reviewedAt) : null,
      resultPayload.overrideReason || null
    );

    return resultPayload;
  }

  /**
   * Recalculates priority recommendations for all incidents in the database.
   */
  async recalculateAll(weights: PriorityWeights = DEFAULT_PRIORITY_WEIGHTS): Promise<{
    processedCount: number;
    timestamp: string;
    weightsUsed: PriorityWeights;
  }> {
    const incidents = await prisma.incident.findMany({ select: { id: true } });
    for (const inc of incidents) {
      await this.computePriorityForIncident(inc.id, weights);
    }

    return {
      processedCount: incidents.length,
      timestamp: new Date().toISOString(),
      weightsUsed: weights,
    };
  }

  /**
   * Returns ranked maintenance priority queue with multi-factor breakdown and filters.
   */
  async getRankedQueue(filters?: {
    priority?: string;
    reviewStatus?: string;
    region?: string;
    search?: string;
  }): Promise<{
    totalCount: number;
    items: MaintenancePriorityRecommendationResult[];
  }> {
    // Ensure all incidents have recommendations
    const incidents = await prisma.incident.findMany({
      include: {
        images: true,
        predictions: { orderBy: { createdAt: 'desc' }, take: 1 },
        verifications: { orderBy: { verifiedAt: 'desc' }, take: 1, include: { inspector: true } },
        priorityRecommendations: { orderBy: { computedAt: 'desc' }, take: 1 },
      },
    });

    const results: MaintenancePriorityRecommendationResult[] = [];

    for (const inc of incidents) {
      let rec: MaintenancePriorityRecommendationResult;
      if (inc.priorityRecommendations.length > 0) {
        const r = inc.priorityRecommendations[0];
        const lat = Number(inc.latitude);
        const lng = Number(inc.longitude);
        const geo = classifyIncidentCoordinates(lat, lng, inc.incidentNumber);

        rec = {
          id: r.id,
          incidentId: inc.id,
          incidentNumber: inc.incidentNumber,
          title: inc.title,
          address: inc.address,
          region: geo.region,
          coordinates: { latitude: lat, longitude: lng },
          recommendedPriority: r.recommendedPriority as PriorityCategory,
          priorityScore: r.priorityScore,
          factorScores: r.factorScores as any,
          weightsUsed: r.weightsUsed as any,
          contributingEvidence: r.contributingEvidence as any,
          missingInformation: r.missingInformation as any,
          explanation: r.explanation,
          scoringRuleVersion: r.scoringRuleVersion,
          computedAt: r.computedAt.toISOString(),
          reviewStatus: r.reviewStatus as ReviewStatus,
          reviewedPriority: r.reviewedPriority,
          reviewedBy: r.reviewedBy,
          reviewedAt: r.reviewedAt ? r.reviewedAt.toISOString() : null,
          overrideReason: r.overrideReason,
        };
      } else {
        rec = await this.computePriorityForIncident(inc.id);
      }
      results.push(rec);
    }

    // Apply Filters
    let filtered = results;
    if (filters?.priority && filters.priority !== 'all') {
      filtered = filtered.filter((r) => r.recommendedPriority.toLowerCase() === filters.priority!.toLowerCase());
    }
    if (filters?.reviewStatus && filters.reviewStatus !== 'all') {
      filtered = filtered.filter((r) => r.reviewStatus.toLowerCase() === filters.reviewStatus!.toLowerCase());
    }
    if (filters?.region && filters.region !== 'all') {
      filtered = filtered.filter((r) => r.region.toLowerCase() === filters.region!.toLowerCase());
    }
    if (filters?.search) {
      const q = filters.search.toLowerCase();
      filtered = filtered.filter(
        (r) =>
          r.incidentNumber.toLowerCase().includes(q) ||
          r.title.toLowerCase().includes(q) ||
          (r.address && r.address.toLowerCase().includes(q))
      );
    }

    // Order: Priority Rank (CRITICAL > HIGH > NEEDS_REVIEW > MEDIUM > LOW), then score DESC
    const priorityWeightMap: Record<PriorityCategory, number> = {
      CRITICAL: 500,
      HIGH: 400,
      NEEDS_REVIEW: 300,
      MEDIUM: 200,
      LOW: 100,
    };

    filtered.sort((a, b) => {
      const pDiff = (priorityWeightMap[b.recommendedPriority] || 0) - (priorityWeightMap[a.recommendedPriority] || 0);
      if (pDiff !== 0) return pDiff;
      return b.priorityScore - a.priorityScore;
    });

    return {
      totalCount: filtered.length,
      items: filtered,
    };
  }

  /**
   * Returns detailed factor breakdown and review history for a specific incident.
   */
  async getIncidentPriorityDetail(incidentId: string): Promise<MaintenancePriorityRecommendationResult> {
    const existing = await prisma.maintenancePriorityRecommendation.findFirst({
      where: { incidentId },
      orderBy: { computedAt: 'desc' },
    });

    if (existing) {
      const inc = await prisma.incident.findUnique({ where: { id: incidentId } });
      const lat = Number(inc?.latitude ?? 0);
      const lng = Number(inc?.longitude ?? 0);
      const geo = classifyIncidentCoordinates(lat, lng, inc?.incidentNumber);

      return {
        id: existing.id,
        incidentId: existing.incidentId,
        incidentNumber: inc?.incidentNumber || '',
        title: inc?.title || '',
        address: inc?.address || null,
        region: geo.region,
        coordinates: { latitude: lat, longitude: lng },
        recommendedPriority: existing.recommendedPriority as PriorityCategory,
        priorityScore: existing.priorityScore,
        factorScores: existing.factorScores as any,
        weightsUsed: existing.weightsUsed as any,
        contributingEvidence: existing.contributingEvidence as any,
        missingInformation: existing.missingInformation as any,
        explanation: existing.explanation,
        scoringRuleVersion: existing.scoringRuleVersion,
        computedAt: existing.computedAt.toISOString(),
        reviewStatus: existing.reviewStatus as ReviewStatus,
        reviewedPriority: existing.reviewedPriority,
        reviewedBy: existing.reviewedBy,
        reviewedAt: existing.reviewedAt ? existing.reviewedAt.toISOString() : null,
        overrideReason: existing.overrideReason,
      };
    }

    return this.computePriorityForIncident(incidentId);
  }

  /**
   * Human Administrative Review Action (Accept, Override, Reject, Require Inspection).
   * Enforces role authorization and records immutable AuditLog entry.
   * NEVER automatically dispatches maintenance tasks.
   */
  async reviewRecommendation(params: {
    incidentId: string;
    reviewerId: string;
    action: 'accept' | 'override' | 'reject' | 'requires_inspection';
    reviewedPriority?: string;
    overrideReason?: string;
  }): Promise<{
    success: boolean;
    incidentId: string;
    reviewStatus: ReviewStatus;
    reviewedPriority: string | null;
    auditLogId: string;
    message: string;
  }> {
    const { incidentId, reviewerId, action, reviewedPriority, overrideReason } = params;

    // Verify reviewer identity and role
    const reviewer = await prisma.user.findUnique({ where: { id: reviewerId } });
    if (!reviewer) {
      throw new Error(`Reviewer user with ID ${reviewerId} not found.`);
    }

    const authorizedRoles = ['administrator', 'maintenance_team', 'inspector'];
    if (!authorizedRoles.includes(reviewer.role)) {
      throw new Error(`User with role '${reviewer.role}' is not authorized to review maintenance priorities.`);
    }

    if (action === 'override' && !reviewedPriority) {
      throw new Error('Mandatory field `reviewedPriority` must be specified when overriding priority.');
    }
    if (action === 'override' && (!overrideReason || overrideReason.trim().length < 5)) {
      throw new Error('Mandatory field `overrideReason` (at least 5 characters) must be provided when overriding priority.');
    }

    // Determine final review status
    let status: ReviewStatus = 'accepted';
    let finalPriority: string | null = null;

    if (action === 'accept') {
      status = 'accepted';
    } else if (action === 'override') {
      status = 'overridden';
      finalPriority = reviewedPriority!.toUpperCase();
    } else if (action === 'reject') {
      status = 'rejected';
    } else if (action === 'requires_inspection') {
      status = 'requires_inspection';
    }

    const reviewedAt = new Date();

    // Update recommendation table
    await prisma.$executeRawUnsafe(
      `
      UPDATE maintenance_priority_recommendations
      SET review_status = $1,
          reviewed_priority = $2,
          reviewed_by = $3::uuid,
          reviewed_at = $4::timestamp,
          override_reason = $5
      WHERE incident_id = $6::uuid
    `,
      status,
      finalPriority,
      reviewerId,
      reviewedAt,
      overrideReason || null,
      incidentId
    );

    // If accepted or overridden, update incident priority field (without dispatching task)
    if (status === 'accepted' || status === 'overridden') {
      const rec = await prisma.maintenancePriorityRecommendation.findFirst({
        where: { incidentId },
        orderBy: { computedAt: 'desc' },
      });
      const priorityToSet = status === 'overridden' ? finalPriority : rec?.recommendedPriority;
      if (priorityToSet) {
        await prisma.incident.update({
          where: { id: incidentId },
          data: { priority: priorityToSet.toLowerCase() },
        });
      }
    }

    // Record Immutable AuditLog Entry
    const auditLog = await prisma.auditLog.create({
      data: {
        userId: reviewerId,
        entityType: 'maintenance_priority',
        entityId: incidentId,
        action: `MAINTENANCE_PRIORITY_${action.toUpperCase()}`,
        metadata: {
          reviewerName: reviewer.fullName,
          reviewerRole: reviewer.role,
          action,
          reviewedPriority: finalPriority,
          overrideReason,
          reviewedAt: reviewedAt.toISOString(),
          policyNotice: 'Administrative triage review. Automatic dispatch strictly prohibited.',
        },
      },
    });

    return {
      success: true,
      incidentId,
      reviewStatus: status,
      reviewedPriority: finalPriority,
      auditLogId: auditLog.id,
      message: `Priority recommendation successfully marked as '${status}'.`,
    };
  }

  /**
   * Queue Summary Metrics.
   */
  async getQueueSummary(): Promise<{
    totalIncidents: number;
    priorityBreakdown: Record<PriorityCategory, number>;
    reviewBreakdown: {
      pending: number;
      accepted: number;
      overridden: number;
      rejected: number;
      requiresInspection: number;
    };
    regionalBreakdown: {
      bengaluru: number;
      delhi: number;
      ambiguous: number;
    };
    supervisedMlReadiness: {
      status: 'DATA_INSUFFICIENT';
      currentCompletedRepairs: number;
      requiredCompletedRepairs: number;
      rationale: string;
    };
    governanceNotice: string;
  }> {
    const queue = await this.getRankedQueue();

    const priorityBreakdown: Record<PriorityCategory, number> = {
      CRITICAL: 0,
      HIGH: 0,
      MEDIUM: 0,
      LOW: 0,
      NEEDS_REVIEW: 0,
    };

    const reviewBreakdown = {
      pending: 0,
      accepted: 0,
      overridden: 0,
      rejected: 0,
      requiresInspection: 0,
    };

    const regionalBreakdown = {
      bengaluru: 0,
      delhi: 0,
      ambiguous: 0,
    };

    for (const item of queue.items) {
      priorityBreakdown[item.recommendedPriority] = (priorityBreakdown[item.recommendedPriority] || 0) + 1;

      if (item.reviewStatus === 'accepted') reviewBreakdown.accepted++;
      else if (item.reviewStatus === 'overridden') reviewBreakdown.overridden++;
      else if (item.reviewStatus === 'rejected') reviewBreakdown.rejected++;
      else if (item.reviewStatus === 'requires_inspection') reviewBreakdown.requiresInspection++;
      else reviewBreakdown.pending++;

      if (item.region === 'BENGALURU_METRO') regionalBreakdown.bengaluru++;
      else if (item.region === 'DELHI_NCR') regionalBreakdown.delhi++;
      else regionalBreakdown.ambiguous++;
    }

    // Check completed repairs in database
    const completedRepairsCount = await prisma.maintenanceTask.count({
      where: { taskStatus: 'completed' },
    });

    return {
      totalIncidents: queue.totalCount,
      priorityBreakdown,
      reviewBreakdown,
      regionalBreakdown,
      supervisedMlReadiness: {
        status: 'DATA_INSUFFICIENT',
        currentCompletedRepairs: completedRepairsCount,
        requiredCompletedRepairs: 50,
        rationale:
          'Supervised priority model training is administratively locked: Only 0 completed repair records exist. Minimum 50 completed maintenance outcomes required.',
      },
      governanceNotice:
        'Maintenance priority scores are explainable administrative recommendations, not validated predictions or automatic dispatch decisions.',
    };
  }
}

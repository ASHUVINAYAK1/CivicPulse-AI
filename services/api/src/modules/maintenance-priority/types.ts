/**
 * CivicPulse AI - Phase 3, Stage 3: Maintenance Priority Types
 * Configurable, explainable, administrative decision-support schema.
 */

export type PriorityCategory = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'NEEDS_REVIEW';

export type ReviewStatus = 'pending' | 'accepted' | 'overridden' | 'rejected' | 'requires_inspection';

export type SeverityProvenance = 'inspector_verified' | 'provisional_heuristic' | 'unverified_missing';

export interface FactorDetail {
  score: number; // 0 to 100
  weight: number; // 0.0 to 1.0
  weightedScore: number; // score * weight
  provenance?: SeverityProvenance;
  rationale: string;
  metadata?: Record<string, any>;
}

export interface FactorBreakdown {
  severity: FactorDetail;
  damageExtent: FactorDetail;
  incidentAge: FactorDetail;
  localDensity: FactorDetail;
  safetyHazard: FactorDetail;
  evidenceQuality: FactorDetail;
}

export interface PriorityWeights {
  version: string;
  severity: number;
  damageExtent: number;
  incidentAge: number;
  localDensity: number;
  safetyHazard: number;
  evidenceQuality: number;
}

export interface MissingInformation {
  hasMissingSeverity: boolean;
  hasMissingImage: boolean;
  hasMissingCoordinates: boolean;
  hasUnverifiedStatus: boolean;
  hasLowConfidenceCV: boolean;
  hasConflictingEvidence: boolean;
  warnings: string[];
}

export interface MaintenancePriorityRecommendationResult {
  id?: string;
  incidentId: string;
  incidentNumber: string;
  title: string;
  address: string | null;
  region: 'BENGALURU_METRO' | 'DELHI_NCR' | 'OUT_OF_BOUNDS_AMBIGUOUS';
  coordinates: {
    latitude: number;
    longitude: number;
  };
  recommendedPriority: PriorityCategory;
  priorityScore: number; // 0 to 100
  factorScores: FactorBreakdown;
  weightsUsed: PriorityWeights;
  contributingEvidence: {
    damageCategory: string;
    verifiedClass?: string | null;
    verifiedSeverity?: string | null;
    provisionalSeverity?: string | null;
    defectCount: number;
    normalizedArea: number;
    incidentAgeHours: number;
    nearbyIncidentsIn500m: number;
    isMajorCorridor: boolean;
    cvConfidence: number;
  };
  missingInformation: MissingInformation;
  explanation: string;
  scoringRuleVersion: string;
  computedAt: string;
  reviewStatus: ReviewStatus;
  reviewedPriority?: string | null;
  reviewedBy?: string | null;
  reviewedAt?: string | null;
  overrideReason?: string | null;
}


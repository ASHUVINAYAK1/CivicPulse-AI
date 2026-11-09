/**
 * CivicPulse AI - Phase 3, Stage 3: Explainable Maintenance Priority Scoring Engine
 * 
 * Implements transparent, configurable, weighted multi-factor priority scoring.
 * Enforces strict separation between human ground truth and automated proxies.
 */

import {
  FactorBreakdown,
  FactorDetail,
  MissingInformation,
  PriorityCategory,
  PriorityWeights,
  SeverityProvenance,
} from './types';

export const DEFAULT_PRIORITY_WEIGHTS: PriorityWeights = {
  version: 'v1.0.0-heuristic',
  severity: 0.30,
  damageExtent: 0.20,
  incidentAge: 0.15,
  localDensity: 0.15,
  safetyHazard: 0.15,
  evidenceQuality: 0.05,
};

export interface PriorityEvaluationInput {
  damageType: string;
  verifiedClass?: string | null;
  verifiedSeverity?: string | null;
  provisionalSeverity?: string | null;
  isVerified: boolean;
  defectCount: number;
  normalizedArea: number;
  reportedAt: Date | string;
  currentDate?: Date;
  nearbyCountIn500m: number;
  address?: string | null;
  title?: string;
  hasImage: boolean;
  hasValidGps: boolean;
  cvConfidence?: number | null;
}

/**
 * Evaluates individual factors and computes overall priority recommendation.
 */
export function evaluateMaintenancePriority(
  input: PriorityEvaluationInput,
  weights: PriorityWeights = DEFAULT_PRIORITY_WEIGHTS
): {
  recommendedPriority: PriorityCategory;
  priorityScore: number;
  factorScores: FactorBreakdown;
  missingInformation: MissingInformation;
  explanation: string;
  isHeuristicOnly: true;
} {
  const now = input.currentDate || new Date();
  const reportedDate = new Date(input.reportedAt);
  const ageHours = Math.max(0, (now.getTime() - reportedDate.getTime()) / (1000 * 60 * 60));
  const ageDays = ageHours / 24;

  const warnings: string[] = [];

  // 1. FACTOR: Severity Score (Range: 0 - 100)
  let severityScore = 30;
  let severityProvenance: SeverityProvenance = 'unverified_missing';
  let severityRationale = '';

  const cleanVerified = input.verifiedSeverity?.toLowerCase().trim();
  const cleanProvisional = input.provisionalSeverity?.toUpperCase().trim();

  if (input.isVerified && cleanVerified) {
    severityProvenance = 'inspector_verified';
    switch (cleanVerified) {
      case 'critical':
        severityScore = 100;
        severityRationale = 'Verified Critical: Authoritative human inspector confirmed severe pavement defect.';
        break;
      case 'high':
        severityScore = 75;
        severityRationale = 'Verified High: Inspector confirmed substantial structural distress requiring near-term repair.';
        break;
      case 'medium':
        severityScore = 50;
        severityRationale = 'Verified Medium: Inspector confirmed moderate deterioration suitable for scheduled queue.';
        break;
      case 'low':
        severityScore = 25;
        severityRationale = 'Verified Low: Inspector confirmed superficial cosmetic distress.';
        break;
      default:
        severityScore = 50;
        severityRationale = `Verified Custom (${cleanVerified}): Field rating assigned.`;
    }
  } else if (cleanProvisional && cleanProvisional !== 'INSUFFICIENT_EVIDENCE') {
    severityProvenance = 'provisional_heuristic';
    switch (cleanProvisional) {
      case 'CRITICAL':
        severityScore = 90;
        severityRationale = 'Provisional Critical: Triggered engineering rule for high-extent cavity; pending inspector confirmation.';
        break;
      case 'HIGH':
        severityScore = 70;
        severityRationale = 'Provisional High: Triggered engineering rule for structural fatigue or multiple defects.';
        break;
      case 'MODERATE':
        severityScore = 45;
        severityRationale = 'Provisional Moderate: Surface crack extent indicates early pavement wear.';
        break;
      case 'LOW':
        severityScore = 20;
        severityRationale = 'Provisional Low: Minor hairline crack or single defect.';
        break;
    }
    warnings.push('Severity is provisional engineering heuristic; not yet confirmed by human inspector.');
  } else {
    severityProvenance = 'unverified_missing';
    severityScore = 35;
    severityRationale = 'Unverified: No inspector ground truth or provisional rule assessment available.';
    warnings.push('Incident lacks authoritative severity rating.');
  }

  const severityDetail: FactorDetail = {
    score: severityScore,
    weight: weights.severity,
    weightedScore: Number((severityScore * weights.severity).toFixed(2)),
    provenance: severityProvenance,
    rationale: severityRationale,
  };

  // 2. FACTOR: Damage Extent (Range: 0 - 100)
  let extentScore = 30;
  let extentRationale = '';

  if (input.normalizedArea > 0 || input.defectCount > 0) {
    const areaContribution = Math.min(70, Math.round(input.normalizedArea * 500));
    const defectContribution = Math.min(30, Math.max(0, (input.defectCount - 1) * 15));
    extentScore = Math.min(100, Math.max(20, areaContribution + defectContribution));
    extentRationale = `Optical extent: Area ratio ${(input.normalizedArea * 100).toFixed(2)}%, defect count ${input.defectCount}.`;
  } else {
    // Fallback based on damage category
    const cat = (input.damageType || '').toLowerCase();
    if (cat.includes('pothole') || cat.includes('d40')) {
      extentScore = 50;
      extentRationale = 'Category proxy: Pothole default extent score 50 (optical bbox unavailable).';
    } else if (cat.includes('alligator') || cat.includes('d20')) {
      extentScore = 45;
      extentRationale = 'Category proxy: Alligator cracking default extent score 45.';
    } else {
      extentScore = 30;
      extentRationale = 'Category proxy: General crack default extent score 30.';
    }
    warnings.push('Exact bounding box pixel extent unavailable; using category proxy.');
  }

  const extentDetail: FactorDetail = {
    score: extentScore,
    weight: weights.damageExtent,
    weightedScore: Number((extentScore * weights.damageExtent).toFixed(2)),
    rationale: extentRationale,
  };

  // 3. FACTOR: Incident Age (Range: 0 - 100)
  let ageScore = 10;
  let ageRationale = '';

  if (ageHours >= 336) {
    // > 14 days
    ageScore = 100;
    ageRationale = `Overdue backlog: Incident reported ${ageDays.toFixed(1)} days ago (>14 days).`;
  } else if (ageHours >= 168) {
    // > 7 days
    ageScore = 75;
    ageRationale = `Extended wait: Reported ${ageDays.toFixed(1)} days ago (>7 days).`;
  } else if (ageHours >= 72) {
    // > 3 days
    ageScore = 50;
    ageRationale = `Standard aging: Reported ${ageHours.toFixed(0)} hours ago (>3 days).`;
  } else if (ageHours >= 24) {
    // > 24 hours
    ageScore = 30;
    ageRationale = `Recent: Reported ${ageHours.toFixed(0)} hours ago (>1 day).`;
  } else {
    // < 24 hours
    ageScore = Math.max(10, Math.min(30, Math.round(ageHours * 1.25)));
    ageRationale = `Fresh submission: Reported within the last ${Math.max(1, Math.round(ageHours))} hours.`;
  }

  const ageDetail: FactorDetail = {
    score: ageScore,
    weight: weights.incidentAge,
    weightedScore: Number((ageScore * weights.incidentAge).toFixed(2)),
    rationale: ageRationale,
    metadata: { ageHours: Number(ageHours.toFixed(1)), ageDays: Number(ageDays.toFixed(1)) },
  };

  // 4. FACTOR: Local Density / Spatial Clustering (Range: 0 - 100)
  let densityScore = 10;
  let densityRationale = '';
  const nNeighbors = input.nearbyCountIn500m;

  if (nNeighbors >= 10) {
    densityScore = 100;
    densityRationale = `High spatial concentration: ${nNeighbors} nearby distress incidents within 500m corridor.`;
  } else if (nNeighbors >= 5) {
    densityScore = 75;
    densityRationale = `Moderate cluster: ${nNeighbors} nearby incidents within 500m.`;
  } else if (nNeighbors >= 2) {
    densityScore = 50;
    densityRationale = `Localized group: ${nNeighbors} nearby incidents within 500m.`;
  } else if (nNeighbors === 1) {
    densityScore = 25;
    densityRationale = 'Paired incident: 1 nearby incident within 500m.';
  } else {
    densityScore = 10;
    densityRationale = 'Isolated defect: Zero nearby incidents identified within 500m.';
  }

  const densityDetail: FactorDetail = {
    score: densityScore,
    weight: weights.localDensity,
    weightedScore: Number((densityScore * weights.localDensity).toFixed(2)),
    rationale: densityRationale,
    metadata: { nearbyIncidentsIn500m: nNeighbors },
  };

  // 5. FACTOR: Safety Hazard Assessment (Range: 0 - 100)
  const locText = `${input.address || ''} ${input.title || ''}`.toLowerCase();
  const isMajorCorridor =
    locText.includes('ring road') ||
    locText.includes('flyover') ||
    locText.includes('transitway') ||
    locText.includes('expressway') ||
    locText.includes('highway') ||
    locText.includes('junction') ||
    locText.includes('bridge') ||
    locText.includes('arterial') ||
    locText.includes('corridor');

  const damageLower = (input.damageType || '').toLowerCase();
  const isPothole = damageLower.includes('pothole') || damageLower.includes('d40');
  const isAlligator = damageLower.includes('alligator') || damageLower.includes('d20');

  let hazardScore = 20;
  let hazardRationale = '';

  if (isPothole && isMajorCorridor) {
    hazardScore = 100;
    hazardRationale = 'Critical safety risk: Pothole cavity situated on high-speed / heavy-traffic arterial corridor.';
  } else if (isPothole) {
    hazardScore = 75;
    hazardRationale = 'Substantial safety risk: Pothole defect capable of causing vehicle rim damage or cyclist loss of control.';
  } else if (isAlligator && isMajorCorridor) {
    hazardScore = 65;
    hazardRationale = 'Structural risk: Base course fatigue on major transit corridor risking localized subsidence.';
  } else if (isAlligator) {
    hazardScore = 45;
    hazardRationale = 'Structural wear: Interconnected alligator cracking without imminent traffic obstruction.';
  } else if (isMajorCorridor) {
    hazardScore = 40;
    hazardRationale = 'Corridor risk: Surface defect located on priority thoroughfare.';
  } else {
    hazardScore = 20;
    hazardRationale = 'Low immediate safety risk: Minor longitudinal or transverse crack on secondary roadway.';
  }

  const hazardDetail: FactorDetail = {
    score: hazardScore,
    weight: weights.safetyHazard,
    weightedScore: Number((hazardScore * weights.safetyHazard).toFixed(2)),
    rationale: hazardRationale,
    metadata: { isMajorCorridor, isPothole, isAlligator },
  };

  // 6. FACTOR: Evidence Quality (Range: 0 - 100)
  let evidenceScore = 50;
  let evidenceRationale = '';
  const conf = input.cvConfidence ?? 0.75;

  if (input.isVerified) {
    evidenceScore = 100;
    evidenceRationale = 'Authoritative: Inspector verified with field ground truth.';
  } else if (input.hasImage && input.hasValidGps && conf >= 0.85) {
    evidenceScore = 75;
    evidenceRationale = `High-confidence automated report: Valid imagery, GPS coordinates, and ${(conf * 100).toFixed(1)}% CV confidence.`;
  } else if (input.hasImage && input.hasValidGps && conf >= 0.60) {
    evidenceScore = 55;
    evidenceRationale = `Moderate-confidence report: Image and GPS verified, ${(conf * 100).toFixed(1)}% CV confidence.`;
  } else if (!input.hasImage) {
    evidenceScore = 25;
    evidenceRationale = 'Weak evidence: Photographic image missing from storage.';
    warnings.push('Missing inspection photograph.');
  } else if (!input.hasValidGps) {
    evidenceScore = 20;
    evidenceRationale = 'Weak evidence: Geolocation coordinates missing or malformed.';
    warnings.push('Missing valid geographic coordinates.');
  } else {
    evidenceScore = 35;
    evidenceRationale = `Marginal evidence: Low CV confidence (${(conf * 100).toFixed(1)}%).`;
    warnings.push('Low visual detection confidence.');
  }

  const evidenceDetail: FactorDetail = {
    score: evidenceScore,
    weight: weights.evidenceQuality,
    weightedScore: Number((evidenceScore * weights.evidenceQuality).toFixed(2)),
    rationale: evidenceRationale,
    metadata: { cvConfidence: conf, hasImage: input.hasImage, hasValidGps: input.hasValidGps },
  };

  // Combine Factor Scores
  const factorScores: FactorBreakdown = {
    severity: severityDetail,
    damageExtent: extentDetail,
    incidentAge: ageDetail,
    localDensity: densityDetail,
    safetyHazard: hazardDetail,
    evidenceQuality: evidenceDetail,
  };

  const totalScore = Number(
    (
      severityDetail.weightedScore +
      extentDetail.weightedScore +
      ageDetail.weightedScore +
      densityDetail.weightedScore +
      hazardDetail.weightedScore +
      evidenceDetail.weightedScore
    ).toFixed(2)
  );

  // Check Conflicting Evidence: only when both verified ground truth and provisional rule have substantive conflicting ratings
  const hasSubstantiveProvisional =
    Boolean(cleanProvisional) && cleanProvisional !== 'INSUFFICIENT_EVIDENCE';

  const provisionalScoreValue =
    cleanProvisional === 'CRITICAL'
      ? 90
      : cleanProvisional === 'HIGH'
      ? 70
      : cleanProvisional === 'MODERATE'
      ? 45
      : cleanProvisional === 'LOW'
      ? 20
      : 50;

  // A genuine conflict occurs when ratings diverge by more than one tier (difference > 35 points, e.g. Low vs High/Critical)
  const hasConflictingEvidence =
    Boolean(input.verifiedSeverity && hasSubstantiveProvisional) &&
    cleanVerified !== cleanProvisional?.toLowerCase() &&
    Math.abs(severityScore - provisionalScoreValue) > 35;

  if (hasConflictingEvidence) {
    warnings.push(`Assessment discrepancy: Inspector assigned '${input.verifiedSeverity}', while provisional rule indicated '${input.provisionalSeverity}'.`);
  }

  const missingInfo: MissingInformation = {
    hasMissingSeverity: severityProvenance === 'unverified_missing',
    hasMissingImage: !input.hasImage,
    hasMissingCoordinates: !input.hasValidGps,
    hasUnverifiedStatus: !input.isVerified,
    hasLowConfidenceCV: conf < 0.60,
    hasConflictingEvidence,
    warnings,
  };

  // Determine Priority Category
  let recommendedPriority: PriorityCategory = 'LOW';

  // Rule 1: Safety & Inadequate/Conflicting Data Gate
  if (missingInfo.hasMissingCoordinates || missingInfo.hasMissingImage || (missingInfo.hasMissingSeverity && !input.hasImage) || hasConflictingEvidence) {
    recommendedPriority = 'NEEDS_REVIEW';
  } else if (totalScore >= 80.0 || (input.isVerified && cleanVerified === 'critical' && isMajorCorridor)) {
    recommendedPriority = 'CRITICAL';
  } else if (totalScore >= 65.0) {
    recommendedPriority = 'HIGH';
  } else if (totalScore >= 45.0) {
    recommendedPriority = 'MEDIUM';
  } else {
    recommendedPriority = 'LOW';
  }

  // Generate plain-language explanation
  const topFactors = [
    { name: 'Severity', weighted: severityDetail.weightedScore, rationale: severityRationale },
    { name: 'Safety Hazard', weighted: hazardDetail.weightedScore, rationale: hazardRationale },
    { name: 'Damage Extent', weighted: extentDetail.weightedScore, rationale: extentRationale },
    { name: 'Local Density', weighted: densityDetail.weightedScore, rationale: densityRationale },
    { name: 'Incident Age', weighted: ageDetail.weightedScore, rationale: ageRationale },
  ].sort((a, b) => b.weighted - a.weighted);

  const explanation = [
    `Recommended Priority: ${recommendedPriority} (Composite Score: ${totalScore.toFixed(1)}/100).`,
    `Primary Driver: ${topFactors[0].name} (${topFactors[0].weighted.toFixed(1)} pts) — ${topFactors[0].rationale}`,
    `Secondary Driver: ${topFactors[1].name} (${topFactors[1].weighted.toFixed(1)} pts) — ${topFactors[1].rationale}`,
    missingInfo.warnings.length > 0 ? `Governance Notice: ${missingInfo.warnings.join(' ')}` : '',
  ]
    .filter(Boolean)
    .join(' ');

  return {
    recommendedPriority,
    priorityScore: totalScore,
    factorScores,
    missingInformation: missingInfo,
    explanation,
    isHeuristicOnly: true,
  };
}

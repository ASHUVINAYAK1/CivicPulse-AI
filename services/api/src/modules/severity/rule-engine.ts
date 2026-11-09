/**
 * CivicPulse AI - Engineering Severity Rule Assessment Engine
 * Implements transparent, deterministic provisional severity scoring.
 *
 * GOVERNANCE NOTICE:
 * 1. Outputs are PROVISIONAL ENGINEERING ASSESSMENTS, never inspector ground truth.
 * 2. No authoritative IRC standard thresholds exist in the repository;
 *    these rules are heuristic pavement triage criteria based on geometric defect extent.
 * 3. Returns INSUFFICIENT_EVIDENCE when empirical observations are absent.
 */

import { IncidentLevelFeatures } from './feature-extractor';

export type ProvisionalSeverityLevel =
  | 'LOW'
  | 'MODERATE'
  | 'HIGH'
  | 'CRITICAL'
  | 'INSUFFICIENT_EVIDENCE';

export interface SeverityRuleDefinition {
  ruleId: string;
  name: string;
  version: string;
  inputFeatures: string[];
  conditionDescription: string;
  severityContribution: ProvisionalSeverityLevel;
  rationale: string;
  engineeringJustification: string;
}

export interface RuleEvaluationResult {
  ruleId: string;
  name: string;
  triggered: boolean;
  severityContribution: ProvisionalSeverityLevel;
  rationale: string;
  matchedFeatureValues: Record<string, any>;
}

export interface ProvisionalSeverityAssessment {
  provisionalSeverity: ProvisionalSeverityLevel;
  isProvisional: true;
  isInspectorVerified: false;
  confidenceScore: number; // 0.0 to 1.0 (heuristic certainty based on evidence coverage)
  assessmentTimestamp: string;
  primaryRuleTriggered: string;
  contributingRules: RuleEvaluationResult[];
  governanceNotices: string[];
  featuresUsed: {
    primaryDamageCategory: string;
    totalDefects: number;
    maxDefectArea: number;
    totalDefectArea: number;
    meanDetectionConfidence: number;
    imageCount: number;
  };
}

export const SEVERITY_RULES: SeverityRuleDefinition[] = [
  {
    ruleId: 'RULE-SEV-01',
    name: 'Severe Cavity Hazard (D40 / Major Pothole)',
    version: '1.0.0',
    inputFeatures: ['primaryDamageCategory', 'maxDefectArea', 'totalDefects'],
    conditionDescription:
      'Damage category is D40 (pothole) AND (maxNormalizedArea >= 0.05 OR totalDefects >= 3)',
    severityContribution: 'CRITICAL',
    rationale:
      'Large or clustered pavement cavities create immediate tire/rim failure, vehicle swerve hazards, and severe danger to two-wheeler riders.',
    engineeringJustification:
      'Empirical municipal pavement triage heuristic. (Note: Project repository contains no codified IRC thresholds; this is a transparent geometric extent proxy).',
  },
  {
    ruleId: 'RULE-SEV-02',
    name: 'Isolated Surface Cavity (D40 / Moderate Pothole)',
    version: '1.0.0',
    inputFeatures: ['primaryDamageCategory', 'maxDefectArea'],
    conditionDescription:
      'Damage category is D40 (pothole) AND maxNormalizedArea < 0.05',
    severityContribution: 'HIGH',
    rationale:
      'Localized asphalt loss subject to rapid moisture infiltration and progressive expansion under traffic loading.',
    engineeringJustification:
      'Empirical municipal maintenance prioritization for localized asphalt loss.',
  },
  {
    ruleId: 'RULE-SEV-03',
    name: 'Structural Base Fatigue (D20 / Alligator Cracking)',
    version: '1.0.0',
    inputFeatures: ['primaryDamageCategory', 'totalDefectArea', 'totalDefects'],
    conditionDescription:
      'Damage category is D20 (alligator cracking) AND (totalDefectArea >= 0.06 OR totalDefects >= 2)',
    severityContribution: 'HIGH',
    rationale:
      'Interconnected fatigue cracking signifies sub-base layer failure requiring mill-and-inlay resurfacing rather than superficial patching.',
    engineeringJustification:
      'Flexible pavement fatigue failure heuristic based on interconnected crack network extent.',
  },
  {
    ruleId: 'RULE-SEV-04',
    name: 'Incipient Fatigue Cracking (D20 / Minor Alligator)',
    version: '1.0.0',
    inputFeatures: ['primaryDamageCategory', 'totalDefectArea'],
    conditionDescription:
      'Damage category is D20 (alligator cracking) AND totalDefectArea < 0.06',
    severityContribution: 'MODERATE',
    rationale:
      'Early-stage fatigue network without deep base depression; candidate for preventive crack seal coat.',
    engineeringJustification:
      'Preventive pavement maintenance triage.',
  },
  {
    ruleId: 'RULE-SEV-05',
    name: 'Extensive Linear Separation (D00/D10 / Major Longitudinal or Transverse Crack)',
    version: '1.0.0',
    inputFeatures: ['primaryDamageCategory', 'maxDefectArea'],
    conditionDescription:
      'Damage category is D00 or D10 AND maxNormalizedArea >= 0.05',
    severityContribution: 'MODERATE',
    rationale:
      'Significant longitudinal joint separation or transverse thermal cracking requiring routing and sealant to stop water ingress.',
    engineeringJustification:
      'Pavement joint opening heuristic based on linear defect width/area.',
  },
  {
    ruleId: 'RULE-SEV-06',
    name: 'Minor Linear Separation (D00/D10 / Surface Hairline Crack)',
    version: '1.0.0',
    inputFeatures: ['primaryDamageCategory', 'maxDefectArea'],
    conditionDescription:
      'Damage category is D00 or D10 AND maxNormalizedArea < 0.05',
    severityContribution: 'LOW',
    rationale:
      'Superficial hairline cracking with minimal ride quality or structural risk. Recommended for routine monitoring.',
    engineeringJustification:
      'Non-critical pavement crack preservation tier.',
  },
  {
    ruleId: 'RULE-SEV-07',
    name: 'Multi-Defect Compound Hazard Escalation',
    version: '1.0.0',
    inputFeatures: ['totalDefects', 'categoryDiversity'],
    conditionDescription:
      'Total defects >= 4 OR categoryDiversity >= 2 distinct defect types',
    severityContribution: 'HIGH',
    rationale:
      'Co-occurrence of multiple defect types or heavy defect density accelerates road degradation beyond individual defect severities.',
    engineeringJustification:
      'Compound pavement distress risk factor.',
  },
];

export class EngineeringSeverityRuleEngine {
  /**
   * Evaluates an incident feature profile against the transparent rule system.
   */
  public static evaluate(features: IncidentLevelFeatures): ProvisionalSeverityAssessment {
    const governanceNotices = [
      'PROVISIONAL ENGINEERING ASSESSMENT: This rating is generated by automated geometric rules and CV detections. It is NOT inspector-verified ground truth.',
      'NO AUTHORITATIVE IRC THRESHOLDS FOUND: The repository contains no codified IRC pavement threshold standards; these rules are transparent engineering proxies.',
    ];

    // Check evidence sufficiency
    if (!features.evidenceSufficient || features.totalDefects === 0) {
      return {
        provisionalSeverity: 'INSUFFICIENT_EVIDENCE',
        isProvisional: true,
        isInspectorVerified: false,
        confidenceScore: 0.0,
        assessmentTimestamp: new Date().toISOString(),
        primaryRuleTriggered: 'NONE_INSUFFICIENT_EVIDENCE',
        contributingRules: [],
        governanceNotices: [
          ...governanceNotices,
          `INSUFFICIENT EVIDENCE: ${features.insufficiencyReasons.join(' ')}`,
        ],
        featuresUsed: {
          primaryDamageCategory: features.primaryDamageCategory,
          totalDefects: features.totalDefects,
          maxDefectArea: features.maxDefectArea,
          totalDefectArea: features.totalDefectArea,
          meanDetectionConfidence: features.meanDetectionConfidence,
          imageCount: features.imageCount,
        },
      };
    }

    const evaluations: RuleEvaluationResult[] = [];
    const cat = features.primaryDamageCategory;
    const maxArea = features.maxDefectArea;
    const totalArea = features.totalDefectArea;
    const totalDefects = features.totalDefects;

    // Evaluate R1
    const r1Triggered = cat === 'D40' && (maxArea >= 0.05 || totalDefects >= 3);
    evaluations.push({
      ruleId: 'RULE-SEV-01',
      name: 'Severe Cavity Hazard (D40 / Major Pothole)',
      triggered: r1Triggered,
      severityContribution: 'CRITICAL',
      rationale:
        'Large or clustered pavement cavities create immediate tire/rim failure, vehicle swerve hazards, and severe danger.',
      matchedFeatureValues: { category: cat, maxArea, totalDefects },
    });

    // Evaluate R2
    const r2Triggered = cat === 'D40' && !r1Triggered;
    evaluations.push({
      ruleId: 'RULE-SEV-02',
      name: 'Isolated Surface Cavity (D40 / Moderate Pothole)',
      triggered: r2Triggered,
      severityContribution: 'HIGH',
      rationale:
        'Localized asphalt loss subject to rapid moisture infiltration and progressive expansion under traffic loading.',
      matchedFeatureValues: { category: cat, maxArea },
    });

    // Evaluate R3
    const r3Triggered = cat === 'D20' && (totalArea >= 0.06 || totalDefects >= 2);
    evaluations.push({
      ruleId: 'RULE-SEV-03',
      name: 'Structural Base Fatigue (D20 / Alligator Cracking)',
      triggered: r3Triggered,
      severityContribution: 'HIGH',
      rationale:
        'Interconnected fatigue cracking signifies sub-base layer failure requiring mill-and-inlay resurfacing.',
      matchedFeatureValues: { category: cat, totalArea, totalDefects },
    });

    // Evaluate R4
    const r4Triggered = cat === 'D20' && !r3Triggered;
    evaluations.push({
      ruleId: 'RULE-SEV-04',
      name: 'Incipient Fatigue Cracking (D20 / Minor Alligator)',
      triggered: r4Triggered,
      severityContribution: 'MODERATE',
      rationale:
        'Early-stage fatigue network without deep base depression; candidate for preventive crack seal coat.',
      matchedFeatureValues: { category: cat, totalArea },
    });

    // Evaluate R5
    const r5Triggered = (cat === 'D00' || cat === 'D10') && maxArea >= 0.05;
    evaluations.push({
      ruleId: 'RULE-SEV-05',
      name: 'Extensive Linear Separation (D00/D10 / Major Crack)',
      triggered: r5Triggered,
      severityContribution: 'MODERATE',
      rationale:
        'Significant longitudinal joint separation or transverse thermal cracking requiring routing and sealant.',
      matchedFeatureValues: { category: cat, maxArea },
    });

    // Evaluate R6
    const r6Triggered = (cat === 'D00' || cat === 'D10') && !r5Triggered;
    evaluations.push({
      ruleId: 'RULE-SEV-06',
      name: 'Minor Linear Separation (D00/D10 / Surface Hairline Crack)',
      triggered: r6Triggered,
      severityContribution: 'LOW',
      rationale:
        'Superficial hairline cracking with minimal ride quality or structural risk.',
      matchedFeatureValues: { category: cat, maxArea },
    });

    // Evaluate R7 (Multi-defect escalation)
    const r7Triggered = totalDefects >= 4 || Object.keys(features.categoryDistribution).length >= 2;
    evaluations.push({
      ruleId: 'RULE-SEV-07',
      name: 'Multi-Defect Compound Hazard Escalation',
      triggered: r7Triggered,
      severityContribution: 'HIGH',
      rationale:
        'Co-occurrence of multiple defect types or heavy defect density accelerates road degradation.',
      matchedFeatureValues: { totalDefects, categories: features.categoryDistribution },
    });

    // Determine highest contributed severity level
    const priorityOrder: Record<ProvisionalSeverityLevel, number> = {
      CRITICAL: 4,
      HIGH: 3,
      MODERATE: 2,
      LOW: 1,
      INSUFFICIENT_EVIDENCE: 0,
    };

    let highestSeverity: ProvisionalSeverityLevel = 'LOW';
    let primaryRule = 'RULE-SEV-06';

    const triggeredRules = evaluations.filter((e) => e.triggered);

    for (const rule of triggeredRules) {
      if (priorityOrder[rule.severityContribution] > priorityOrder[highestSeverity]) {
        highestSeverity = rule.severityContribution;
        primaryRule = rule.ruleId;
      }
    }

    // Heuristic confidence based on detection confidence and image count
    const evidenceConfidence = Math.min(
      0.95,
      features.meanDetectionConfidence * 0.7 + (features.imageCount > 1 ? 0.2 : 0.1)
    );

    return {
      provisionalSeverity: highestSeverity,
      isProvisional: true,
      isInspectorVerified: false,
      confidenceScore: Number(evidenceConfidence.toFixed(2)),
      assessmentTimestamp: new Date().toISOString(),
      primaryRuleTriggered: primaryRule,
      contributingRules: triggeredRules,
      governanceNotices,
      featuresUsed: {
        primaryDamageCategory: features.primaryDamageCategory,
        totalDefects: features.totalDefects,
        maxDefectArea: features.maxDefectArea,
        totalDefectArea: features.totalDefectArea,
        meanDetectionConfidence: features.meanDetectionConfidence,
        imageCount: features.imageCount,
      },
    };
  }
}


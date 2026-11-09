export interface ScoringWeights {
  imageWeight: number;
  geoWeight: number;
  timeWeight: number;
  typeWeight: number;
}

export const DEFAULT_SCORING_WEIGHTS: ScoringWeights = {
  imageWeight: 0.40,
  geoWeight: 0.30,
  timeWeight: 0.15,
  typeWeight: 0.15,
};

export interface DuplicateScoringInput {
  distanceMeters: number;
  maxRadiusMeters?: number;
  timeDeltaHours: number;
  maxTimeWindowDays?: number;
  targetDamageType: string;
  candidateDamageType: string;
  bestImageSimilarity: number;
  bestHammingDistance: number;
  hashTypeUsed: 'dHash' | 'pHash' | 'none';
  metadataMatch?: {
    sameDevice?: boolean;
    timestampWithin5Min?: boolean;
  };
}

export interface DuplicateScoreResult {
  confidenceScore: number; // 0.0 to 1.0
  verdict: 'high_probability_duplicate' | 'potential_duplicate' | 'unlikely_duplicate';
  isSuspectedDuplicate: boolean; // confidenceScore >= 0.50
  components: {
    imageScore: number;
    geoScore: number;
    timeScore: number;
    typeScore: number;
    metadataBonus: number;
  };
  reasons: string[];
}

/**
 * Computes damage category compatibility score [0.0, 1.0].
 */
function getDamageTypeCompatibility(t1: string, t2: string): number {
  const norm1 = t1?.toLowerCase().trim() || '';
  const norm2 = t2?.toLowerCase().trim() || '';

  if (norm1 === norm2) return 1.0;

  // Compatible distress families
  const isCrack1 = norm1.includes('crack');
  const isCrack2 = norm2.includes('crack');
  if (isCrack1 && isCrack2) return 0.65;

  return 0.20;
}

/**
 * Rule-based, explainable scoring engine combining perceptual image hashing,
 * PostGIS spatial distance, temporal delta, and distress classification.
 */
export function scoreDuplicateCandidate(
  input: DuplicateScoringInput,
  weights: ScoringWeights = DEFAULT_SCORING_WEIGHTS
): DuplicateScoreResult {
  const maxRadius = input.maxRadiusMeters || 10.0;
  const maxWindowHours = (input.maxTimeWindowDays || 30) * 24.0;
  const reasons: string[] = [];

  // 1. Geographic Proximity Score [0.0, 1.0]
  const clampedDistance = Math.min(Math.max(input.distanceMeters, 0), maxRadius);
  const geoScore = Math.max(0, 1.0 - clampedDistance / maxRadius);
  reasons.push(
    `GPS Distance: ${input.distanceMeters.toFixed(1)}m from incident (within ${maxRadius.toFixed(0)}m radius; geo-score: ${geoScore.toFixed(2)})`
  );

  // 2. Perceptual Image Similarity Score [0.0, 1.0]
  let imageScore = input.bestImageSimilarity;
  if (input.hashTypeUsed !== 'none') {
    reasons.push(
      `Visual Similarity: ${(input.bestImageSimilarity * 100).toFixed(1)}% match (${input.hashTypeUsed} Hamming distance: ${input.bestHammingDistance}/64 bits)`
    );
  } else {
    // If no images are available to compare, fallback gracefully
    imageScore = 0.5;
    reasons.push(`Visual Similarity: No comparable images available (neutral baseline applied)`);
  }

  // 3. Temporal Proximity Score [0.0, 1.0]
  let timeScore = 0.0;
  if (input.timeDeltaHours <= 24.0) {
    // Same day report
    timeScore = 1.0;
    reasons.push(`Temporal Delta: ${input.timeDeltaHours.toFixed(1)} hours apart (reported same day; time-score: 1.00)`);
  } else if (input.timeDeltaHours <= maxWindowHours) {
    // Linear decay across the time window
    timeScore = Math.max(0, 1.0 - (input.timeDeltaHours - 24.0) / (maxWindowHours - 24.0));
    reasons.push(
      `Temporal Delta: ${(input.timeDeltaHours / 24.0).toFixed(1)} days apart (time-score: ${timeScore.toFixed(2)})`
    );
  } else {
    timeScore = 0.0;
    reasons.push(`Temporal Delta: ${(input.timeDeltaHours / 24.0).toFixed(1)} days exceeds ${input.maxTimeWindowDays} day window`);
  }

  // 4. Damage Category Score [0.0, 1.0]
  const typeScore = getDamageTypeCompatibility(input.targetDamageType, input.candidateDamageType);
  if (typeScore === 1.0) {
    reasons.push(`Damage Category: Exact match ('${input.targetDamageType}')`);
  } else if (typeScore >= 0.5) {
    reasons.push(`Damage Category: Compatible defect types ('${input.targetDamageType}' vs '${input.candidateDamageType}')`);
  } else {
    reasons.push(`Damage Category: Divergent defect classifications ('${input.targetDamageType}' vs '${input.candidateDamageType}')`);
  }

  // 5. Metadata Bonus (EXIF hardware / capture timestamp)
  let metadataBonus = 0.0;
  if (input.metadataMatch?.sameDevice) {
    metadataBonus += 0.03;
    reasons.push(`Image Metadata: Identical capture device hardware identified`);
  }
  if (input.metadataMatch?.timestampWithin5Min) {
    metadataBonus += 0.02;
    reasons.push(`Image Metadata: Camera capture timestamps within 5 minutes`);
  }

  // Calculate weighted composite score
  const compositeScore = Math.min(
    1.0,
    weights.imageWeight * imageScore +
      weights.geoWeight * geoScore +
      weights.timeWeight * timeScore +
      weights.typeWeight * typeScore +
      metadataBonus
  );

  const roundedScore = Math.round(compositeScore * 1000) / 1000;

  let verdict: 'high_probability_duplicate' | 'potential_duplicate' | 'unlikely_duplicate';
  if (roundedScore >= 0.70) {
    verdict = 'high_probability_duplicate';
  } else if (roundedScore >= 0.50) {
    verdict = 'potential_duplicate';
  } else {
    verdict = 'unlikely_duplicate';
  }

  return {
    confidenceScore: roundedScore,
    verdict,
    isSuspectedDuplicate: roundedScore >= 0.50,
    components: {
      imageScore: Math.round(imageScore * 1000) / 1000,
      geoScore: Math.round(geoScore * 1000) / 1000,
      timeScore: Math.round(timeScore * 1000) / 1000,
      typeScore: Math.round(typeScore * 1000) / 1000,
      metadataBonus: Math.round(metadataBonus * 1000) / 1000,
    },
    reasons,
  };
}


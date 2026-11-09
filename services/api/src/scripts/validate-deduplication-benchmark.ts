import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import {
  computePerceptualHash,
  computeHammingDistance,
  computeHashSimilarity,
} from '../modules/deduplication/hasher';
import {
  scoreDuplicateCandidate,
  DuplicateScoringInput,
  DuplicateScoreResult,
} from '../modules/deduplication/scorer';

interface BenchmarkPair {
  id: string;
  category:
  | 'format_conversion'
  | 'resize_compression'
  | 'crop_transformation'
  | 'rotation_transformation'
  | 'same_defect_different_photo'
  | 'similar_defect_distant'
  | 'different_defect_same_road'
  | 'nearby_unrelated';
  description: string;
  isDuplicateGroundTruth: boolean;
  image1Buffer: Buffer;
  image2Buffer: Buffer;
  distanceMeters: number;
  timeDeltaHours: number;
  targetDamageType: string;
  candidateDamageType: string;
  metadataMatch?: {
    sameDevice?: boolean;
    timestampWithin5Min?: boolean;
  };
}

interface EvaluationMetrics {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
  fpr: number;
  fnr: number;
  accuracy: number;
}

function calculateMetrics(predictions: boolean[], groundTruths: boolean[]): EvaluationMetrics {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (let i = 0; i < predictions.length; i++) {
    const pred = predictions[i];
    const gt = groundTruths[i];

    if (pred && gt) tp++;
    else if (pred && !gt) fp++;
    else if (!pred && !gt) tn++;
    else if (!pred && gt) fn++;
  }

  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
  const fnr = tp + fn > 0 ? fn / (tp + fn) : 0;
  const accuracy = (tp + tn) / Math.max(tp + fp + tn + fn, 1);

  return {
    tp,
    fp,
    tn,
    fn,
    precision: Math.round(precision * 10000) / 10000,
    recall: Math.round(recall * 10000) / 10000,
    f1: Math.round(f1 * 10000) / 10000,
    fpr: Math.round(fpr * 10000) / 10000,
    fnr: Math.round(fnr * 10000) / 10000,
    accuracy: Math.round(accuracy * 10000) / 10000,
  };
}

async function buildBenchmarkDataset(): Promise<{
  calibrationSet: BenchmarkPair[];
  evaluationSet: BenchmarkPair[];
}> {
  const imagesDir = path.resolve(__dirname, '../../../../datasets/splits/rdd2022_india_grouped_v2/test/images');
  if (!fs.existsSync(imagesDir)) {
    throw new Error(`Images directory not found: ${imagesDir}`);
  }

  // Load real source images
  const sample1 = fs.readFileSync(path.join(imagesDir, 'India_000000.jpg'));
  const sample2 = fs.readFileSync(path.join(imagesDir, 'India_000001.jpg'));
  const sample3 = fs.readFileSync(path.join(imagesDir, 'India_000002.jpg'));
  const sample4 = fs.readFileSync(path.join(imagesDir, 'India_000003.jpg'));
  const sample5 = fs.readFileSync(path.join(imagesDir, 'India_000005.jpg'));
  const sample7 = fs.readFileSync(path.join(imagesDir, 'India_000007.jpg'));

  // Generate synthetic and transformed versions using sharp
  // 1. Format conversions
  const png1 = await sharp(sample1).png().toBuffer();
  const webp1 = await sharp(sample1).webp().toBuffer();
  const png2 = await sharp(sample2).png().toBuffer();
  const webp3 = await sharp(sample3).webp().toBuffer();

  // 2. Resized and compressed copies
  const compressed1 = await sharp(sample1).jpeg({ quality: 35 }).toBuffer();
  const resized1 = await sharp(sample1).resize(480, 270).jpeg({ quality: 50 }).toBuffer();
  const compressed2 = await sharp(sample2).jpeg({ quality: 40 }).toBuffer();
  const resized3 = await sharp(sample3).resize(320, 180).jpeg({ quality: 50 }).toBuffer();

  // 3. Cropped versions
  const meta1 = await sharp(sample1).metadata();
  const crop1 = await sharp(sample1)
    .extract({
      left: Math.floor(meta1.width! * 0.05),
      top: Math.floor(meta1.height! * 0.05),
      width: Math.floor(meta1.width! * 0.90),
      height: Math.floor(meta1.height! * 0.90),
    })
    .toBuffer();

  const meta2 = await sharp(sample2).metadata();
  const crop2 = await sharp(sample2)
    .extract({
      left: Math.floor(meta2.width! * 0.08),
      top: Math.floor(meta2.height! * 0.08),
      width: Math.floor(meta2.width! * 0.84),
      height: Math.floor(meta2.height! * 0.84),
    })
    .toBuffer();

  // 4. Slightly rotated versions
  const rot1 = await sharp(sample1).rotate(2).toBuffer();
  const rot2 = await sharp(sample2).rotate(1.5).toBuffer();
  const rot3 = await sharp(sample3).rotate(3).toBuffer();

  // Construct curated positive pairs (Ground Truth: Duplicate)
  const positivePairs: BenchmarkPair[] = [
    // Format conversion
    {
      id: 'POS-01',
      category: 'format_conversion',
      description: 'Same photograph converted to PNG, GPS 0.5m, time 10m',
      isDuplicateGroundTruth: true,
      image1Buffer: sample1,
      image2Buffer: png1,
      distanceMeters: 0.5,
      timeDeltaHours: 0.15,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Longitudinal Crack',
      metadataMatch: { sameDevice: true, timestampWithin5Min: true },
    },
    {
      id: 'POS-02',
      category: 'format_conversion',
      description: 'Same photograph converted to WebP, GPS 1.2m, time 2h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample1,
      image2Buffer: webp1,
      distanceMeters: 1.2,
      timeDeltaHours: 2.0,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Longitudinal Crack',
    },
    {
      id: 'POS-03',
      category: 'format_conversion',
      description: 'Photo 2 in PNG, GPS 0.8m, time 5h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample2,
      image2Buffer: png2,
      distanceMeters: 0.8,
      timeDeltaHours: 5.0,
      targetDamageType: 'Alligator Crack',
      candidateDamageType: 'Alligator Crack',
    },
    {
      id: 'POS-04',
      category: 'format_conversion',
      description: 'Photo 3 in WebP, GPS 2.0m, time 24h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample3,
      image2Buffer: webp3,
      distanceMeters: 2.0,
      timeDeltaHours: 24.0,
      targetDamageType: 'Pothole',
      candidateDamageType: 'Pothole',
    },
    // Resize & compression
    {
      id: 'POS-05',
      category: 'resize_compression',
      description: 'JPEG compressed (Q=35), GPS 1.5m, time 1h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample1,
      image2Buffer: compressed1,
      distanceMeters: 1.5,
      timeDeltaHours: 1.0,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Longitudinal Crack',
    },
    {
      id: 'POS-06',
      category: 'resize_compression',
      description: 'Resized 480x270 (Q=50), GPS 3.0m, time 4h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample1,
      image2Buffer: resized1,
      distanceMeters: 3.0,
      timeDeltaHours: 4.0,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Longitudinal Crack',
    },
    {
      id: 'POS-07',
      category: 'resize_compression',
      description: 'Photo 2 compressed (Q=40), GPS 2.5m, time 12h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample2,
      image2Buffer: compressed2,
      distanceMeters: 2.5,
      timeDeltaHours: 12.0,
      targetDamageType: 'Alligator Crack',
      candidateDamageType: 'Alligator Crack',
    },
    {
      id: 'POS-08',
      category: 'resize_compression',
      description: 'Photo 3 resized 320x180, GPS 4.0m, time 18h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample3,
      image2Buffer: resized3,
      distanceMeters: 4.0,
      timeDeltaHours: 18.0,
      targetDamageType: 'Pothole',
      candidateDamageType: 'Pothole',
    },
    // Crop transformations
    {
      id: 'POS-09',
      category: 'crop_transformation',
      description: 'Central 90% crop, GPS 2.2m, time 3h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample1,
      image2Buffer: crop1,
      distanceMeters: 2.2,
      timeDeltaHours: 3.0,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Longitudinal Crack',
    },
    {
      id: 'POS-10',
      category: 'crop_transformation',
      description: 'Central 84% crop of photo 2, GPS 3.8m, time 6h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample2,
      image2Buffer: crop2,
      distanceMeters: 3.8,
      timeDeltaHours: 6.0,
      targetDamageType: 'Alligator Crack',
      candidateDamageType: 'Alligator Crack',
    },
    // Rotations
    {
      id: 'POS-11',
      category: 'rotation_transformation',
      description: '2 degree rotation, GPS 1.0m, time 0.5h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample1,
      image2Buffer: rot1,
      distanceMeters: 1.0,
      timeDeltaHours: 0.5,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Longitudinal Crack',
      metadataMatch: { sameDevice: true, timestampWithin5Min: true },
    },
    {
      id: 'POS-12',
      category: 'rotation_transformation',
      description: '1.5 degree rotation of photo 2, GPS 2.0m, time 2h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample2,
      image2Buffer: rot2,
      distanceMeters: 2.0,
      timeDeltaHours: 2.0,
      targetDamageType: 'Alligator Crack',
      candidateDamageType: 'Alligator Crack',
    },
    {
      id: 'POS-13',
      category: 'rotation_transformation',
      description: '3 degree rotation of photo 3, GPS 3.5m, time 8h',
      isDuplicateGroundTruth: true,
      image1Buffer: sample3,
      image2Buffer: rot3,
      distanceMeters: 3.5,
      timeDeltaHours: 8.0,
      targetDamageType: 'Pothole',
      candidateDamageType: 'Pothole',
    },
    // Same physical defect, different photograph (adjacent corridor video frames 1-3m apart)
    {
      id: 'POS-14',
      category: 'same_defect_different_photo',
      description: 'Consecutive video frames (India_000000 & India_000001) showing same crack',
      isDuplicateGroundTruth: true,
      image1Buffer: sample1,
      image2Buffer: sample2,
      distanceMeters: 2.8,
      timeDeltaHours: 0.05,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Longitudinal Crack',
      metadataMatch: { sameDevice: true, timestampWithin5Min: true },
    },
    {
      id: 'POS-15',
      category: 'same_defect_different_photo',
      description: 'Consecutive video frames (India_000002 & India_000003) showing same distress',
      isDuplicateGroundTruth: true,
      image1Buffer: sample3,
      image2Buffer: sample4,
      distanceMeters: 3.2,
      timeDeltaHours: 0.08,
      targetDamageType: 'Pothole',
      candidateDamageType: 'Pothole',
      metadataMatch: { sameDevice: true, timestampWithin5Min: true },
    },
    {
      id: 'POS-16',
      category: 'same_defect_different_photo',
      description: 'Nearby video frames (India_000005 & India_000007) of defect zone',
      isDuplicateGroundTruth: true,
      image1Buffer: sample5,
      image2Buffer: sample7,
      distanceMeters: 4.5,
      timeDeltaHours: 0.12,
      targetDamageType: 'Alligator Crack',
      candidateDamageType: 'Alligator Crack',
      metadataMatch: { sameDevice: true, timestampWithin5Min: true },
    },
  ];

  // Construct curated negative pairs (Ground Truth: Not a duplicate)
  const negativePairs: BenchmarkPair[] = [
    // Visually similar road defects on distant corridors (e.g. 15 km apart)
    {
      id: 'NEG-01',
      category: 'similar_defect_distant',
      description: 'Similar longitudinal crack, but in distant corridor 14.5 km away',
      isDuplicateGroundTruth: false,
      image1Buffer: sample1,
      image2Buffer: crop1, // visually very similar
      distanceMeters: 14500.0,
      timeDeltaHours: 48.0,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Longitudinal Crack',
    },
    {
      id: 'NEG-02',
      category: 'similar_defect_distant',
      description: 'Similar pothole appearance, but located 8.2 km away',
      isDuplicateGroundTruth: false,
      image1Buffer: sample3,
      image2Buffer: webp3,
      distanceMeters: 8200.0,
      timeDeltaHours: 72.0,
      targetDamageType: 'Pothole',
      candidateDamageType: 'Pothole',
    },
    {
      id: 'NEG-03',
      category: 'similar_defect_distant',
      description: 'Similar alligator cracking, located 22 km away',
      isDuplicateGroundTruth: false,
      image1Buffer: sample2,
      image2Buffer: png2,
      distanceMeters: 22000.0,
      timeDeltaHours: 120.0,
      targetDamageType: 'Alligator Crack',
      candidateDamageType: 'Alligator Crack',
    },
    // Different defects on the same road (e.g. 45-80m apart)
    {
      id: 'NEG-04',
      category: 'different_defect_same_road',
      description: 'Pothole vs longitudinal crack 45m apart on same road corridor',
      isDuplicateGroundTruth: false,
      image1Buffer: sample1,
      image2Buffer: sample3,
      distanceMeters: 45.0,
      timeDeltaHours: 2.0,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Pothole',
    },
    {
      id: 'NEG-05',
      category: 'different_defect_same_road',
      description: 'Alligator crack vs pothole 65m apart along same route',
      isDuplicateGroundTruth: false,
      image1Buffer: sample2,
      image2Buffer: sample3,
      distanceMeters: 65.0,
      timeDeltaHours: 3.5,
      targetDamageType: 'Alligator Crack',
      candidateDamageType: 'Pothole',
    },
    {
      id: 'NEG-06',
      category: 'different_defect_same_road',
      description: 'Two distinct potholes 50m apart on same highway',
      isDuplicateGroundTruth: false,
      image1Buffer: sample3,
      image2Buffer: sample5,
      distanceMeters: 50.0,
      timeDeltaHours: 1.0,
      targetDamageType: 'Pothole',
      candidateDamageType: 'Pothole',
    },
    {
      id: 'NEG-07',
      category: 'different_defect_same_road',
      description: 'Crack vs pothole 35m apart',
      isDuplicateGroundTruth: false,
      image1Buffer: sample1,
      image2Buffer: sample7,
      distanceMeters: 35.0,
      timeDeltaHours: 4.0,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Pothole',
    },
    // Nearby but completely unrelated incidents (e.g. within 6-9m, but totally different visual defect)
    {
      id: 'NEG-08',
      category: 'nearby_unrelated',
      description: 'Pothole vs Longitudinal crack 6.5m apart on opposite side of dual carriageway',
      isDuplicateGroundTruth: false,
      image1Buffer: sample1,
      image2Buffer: sample3,
      distanceMeters: 6.5,
      timeDeltaHours: 14.0,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Pothole',
    },
    {
      id: 'NEG-09',
      category: 'nearby_unrelated',
      description: 'Manhole / defect vs alligator cracking 8.2m apart',
      isDuplicateGroundTruth: false,
      image1Buffer: sample2,
      image2Buffer: sample5,
      distanceMeters: 8.2,
      timeDeltaHours: 36.0,
      targetDamageType: 'Alligator Crack',
      candidateDamageType: 'Pothole',
    },
    {
      id: 'NEG-10',
      category: 'nearby_unrelated',
      description: 'Different defect 9.1m apart reported 20 days later',
      isDuplicateGroundTruth: false,
      image1Buffer: sample1,
      image2Buffer: sample5,
      distanceMeters: 9.1,
      timeDeltaHours: 480.0,
      targetDamageType: 'Longitudinal Crack',
      candidateDamageType: 'Transverse Crack',
    },
    {
      id: 'NEG-11',
      category: 'nearby_unrelated',
      description: 'Nearby incident 5.5m apart with distinct imagery and conflicting defect',
      isDuplicateGroundTruth: false,
      image1Buffer: sample3,
      image2Buffer: sample7,
      distanceMeters: 5.5,
      timeDeltaHours: 52.0,
      targetDamageType: 'Pothole',
      candidateDamageType: 'Longitudinal Crack',
    },
    {
      id: 'NEG-12',
      category: 'nearby_unrelated',
      description: 'Opposite lane defect 8.8m apart, completely distinct scene',
      isDuplicateGroundTruth: false,
      image1Buffer: sample2,
      image2Buffer: sample7,
      distanceMeters: 8.8,
      timeDeltaHours: 120.0,
      targetDamageType: 'Alligator Crack',
      candidateDamageType: 'Longitudinal Crack',
    },
    {
      id: 'NEG-13',
      category: 'similar_defect_distant',
      description: 'Distant road photo 50 km away',
      isDuplicateGroundTruth: false,
      image1Buffer: sample4,
      image2Buffer: sample7,
      distanceMeters: 50000.0,
      timeDeltaHours: 200.0,
      targetDamageType: 'Pothole',
      candidateDamageType: 'Pothole',
    },
    {
      id: 'NEG-14',
      category: 'different_defect_same_road',
      description: 'Defect 120m away on same route',
      isDuplicateGroundTruth: false,
      image1Buffer: sample4,
      image2Buffer: sample5,
      distanceMeters: 120.0,
      timeDeltaHours: 8.0,
      targetDamageType: 'Pothole',
      candidateDamageType: 'Alligator Crack',
    },
    {
      id: 'NEG-15',
      category: 'nearby_unrelated',
      description: '7.0m apart, different distress, reported 25 days apart',
      isDuplicateGroundTruth: false,
      image1Buffer: sample4,
      image2Buffer: sample1,
      distanceMeters: 7.0,
      timeDeltaHours: 600.0,
      targetDamageType: 'Pothole',
      candidateDamageType: 'Longitudinal Crack',
    },
    {
      id: 'NEG-16',
      category: 'similar_defect_distant',
      description: 'Similar crack pattern 18 km away',
      isDuplicateGroundTruth: false,
      image1Buffer: sample5,
      image2Buffer: rot1,
      distanceMeters: 18000.0,
      timeDeltaHours: 90.0,
      targetDamageType: 'Alligator Crack',
      candidateDamageType: 'Longitudinal Crack',
    },
  ];

  // Partition into Calibration Set (40%) and Held-out Evaluation Set (60%)
  // Calibration: 6 positive, 6 negative (12 pairs)
  // Held-out: 10 positive, 10 negative (20 pairs)
  const calibrationSet: BenchmarkPair[] = [
    positivePairs[0],
    positivePairs[4],
    positivePairs[8],
    positivePairs[10],
    positivePairs[13],
    positivePairs[14],
    negativePairs[0],
    negativePairs[3],
    negativePairs[7],
    negativePairs[8],
    negativePairs[12],
    negativePairs[14],
  ];

  const calibIds = new Set(calibrationSet.map((p) => p.id));
  const evalPositive = positivePairs.filter((p) => !calibIds.has(p.id));
  const evalNegative = negativePairs.filter((p) => !calibIds.has(p.id));

  const evaluationSet: BenchmarkPair[] = [...evalPositive, ...evalNegative];

  return { calibrationSet, evaluationSet };
}

async function runSignalEvaluation(dataset: BenchmarkPair[]) {
  const groundTruths = dataset.map((p) => p.isDuplicateGroundTruth);

  // Predictions under different individual signals and combined composite score
  const predsDHash: boolean[] = [];
  const predsPHash: boolean[] = [];
  const predsGPS: boolean[] = [];
  const predsTime: boolean[] = [];
  const predsCategory: boolean[] = [];
  const predsComposite: boolean[] = [];

  const latencies: {
    hashComputeMs: number[];
    scoringMs: number[];
  } = {
    hashComputeMs: [],
    scoringMs: [],
  };

  const detailedResults: any[] = [];

  for (const pair of dataset) {
    // 1. Hash compute & timing
    const t0 = performance.now();
    const [h1, h2] = await Promise.all([
      computePerceptualHash(pair.image1Buffer),
      computePerceptualHash(pair.image2Buffer),
    ]);
    const hashTime = performance.now() - t0;
    latencies.hashComputeMs.push(hashTime);

    // Distances
    const dDist = computeHammingDistance(h1.dHash, h2.dHash);
    const dSim = computeHashSimilarity(h1.dHash, h2.dHash);
    const pDist = computeHammingDistance(h1.pHash, h2.pHash);
    const pSim = computeHashSimilarity(h1.pHash, h2.pHash);

    // 2. Individual Signal Decisions
    // dHash alone: duplicate if Hamming distance <= 10 (similarity >= 0.6875)
    predsDHash.push(dDist <= 10);

    // pHash alone: duplicate if Hamming distance <= 10 (similarity >= 0.6875)
    predsPHash.push(pDist <= 10);

    // GPS alone: duplicate if within 10 meters
    predsGPS.push(pair.distanceMeters <= 10.0);

    // Time alone: duplicate if within 24 hours
    predsTime.push(pair.timeDeltaHours <= 24.0);

    // Category alone: duplicate if damage category matches
    predsCategory.push(pair.targetDamageType.toLowerCase() === pair.candidateDamageType.toLowerCase());

    // 3. Composite Score
    const tScoring0 = performance.now();
    const bestSim = Math.max(dSim, pSim);
    const bestDist = bestSim === pSim ? pDist : dDist;
    const hashUsed = bestSim === pSim ? 'pHash' : 'dHash';

    const scoringInput: DuplicateScoringInput = {
      distanceMeters: pair.distanceMeters,
      maxRadiusMeters: 10.0,
      timeDeltaHours: pair.timeDeltaHours,
      maxTimeWindowDays: 30,
      targetDamageType: pair.targetDamageType,
      candidateDamageType: pair.candidateDamageType,
      bestImageSimilarity: bestSim,
      bestHammingDistance: bestDist,
      hashTypeUsed: hashUsed,
      metadataMatch: pair.metadataMatch,
    };

    const scoreResult: DuplicateScoreResult = scoreDuplicateCandidate(scoringInput);
    const scoreTime = performance.now() - tScoring0;
    latencies.scoringMs.push(scoreTime);

    // Deduplication engine decision threshold: score >= 0.50
    predsComposite.push(scoreResult.isSuspectedDuplicate);

    detailedResults.push({
      id: pair.id,
      category: pair.category,
      groundTruth: pair.isDuplicateGroundTruth,
      compositeScore: scoreResult.confidenceScore,
      verdict: scoreResult.verdict,
      prediction: scoreResult.isSuspectedDuplicate,
      dHashDist: dDist,
      pHashDist: pDist,
      distanceMeters: pair.distanceMeters,
      timeDeltaHours: pair.timeDeltaHours,
      reasons: scoreResult.reasons,
    });
  }

  // Calculate metrics for each signal
  const metrics = {
    dHash_only: calculateMetrics(predsDHash, groundTruths),
    pHash_only: calculateMetrics(predsPHash, groundTruths),
    gps_distance_only: calculateMetrics(predsGPS, groundTruths),
    time_delta_only: calculateMetrics(predsTime, groundTruths),
    damage_category_only: calculateMetrics(predsCategory, groundTruths),
    composite_score: calculateMetrics(predsComposite, groundTruths),
  };

  const avgHashLatency =
    latencies.hashComputeMs.reduce((a, b) => a + b, 0) / latencies.hashComputeMs.length;
  const avgScoringLatency =
    latencies.scoringMs.reduce((a, b) => a + b, 0) / latencies.scoringMs.length;

  return {
    metrics,
    latencies: {
      avgHashComputeMs: Math.round(avgHashLatency * 100) / 100,
      avgScoringMs: Math.round(avgScoringLatency * 100) / 100,
      totalPipelineMs: Math.round((avgHashLatency + avgScoringLatency) * 100) / 100,
    },
    detailedResults,
  };
}

async function main() {
  console.log('='.repeat(75));
  console.log('  CivicPulse AI — Phase 2.4 Deduplication Curated Benchmark Validation  ');
  console.log('='.repeat(75));

  const { calibrationSet, evaluationSet } = await buildBenchmarkDataset();
  console.log(`Curated Calibration Set: ${calibrationSet.length} pairs (Positive: ${calibrationSet.filter(p => p.isDuplicateGroundTruth).length}, Negative: ${calibrationSet.filter(p => !p.isDuplicateGroundTruth).length})`);
  console.log(`Curated Held-out Evaluation Set: ${evaluationSet.length} pairs (Positive: ${evaluationSet.filter(p => p.isDuplicateGroundTruth).length}, Negative: ${evaluationSet.filter(p => !p.isDuplicateGroundTruth).length})`);

  console.log('\n[1] Running Calibration Set Evaluation...');
  const calibOutput = await runSignalEvaluation(calibrationSet);

  console.log('[2] Running Held-out Evaluation Set Validation...');
  const evalOutput = await runSignalEvaluation(evaluationSet);

  const report = {
    benchmark_metadata: {
      generated_at: new Date().toISOString(),
      calibration_pairs_count: calibrationSet.length,
      heldout_evaluation_pairs_count: evaluationSet.length,
      decision_threshold: 0.50,
      weights: {
        image: 0.40,
        distance: 0.30,
        time: 0.15,
        type: 0.15,
        metadata_bonus_max: 0.05,
      },
    },
    calibration_metrics: calibOutput.metrics,
    heldout_evaluation_metrics: evalOutput.metrics,
    pipeline_latencies: evalOutput.latencies,
    heldout_pair_results: evalOutput.detailedResults,
  };

  const outputPath = path.resolve(__dirname, '../../../../docs/deduplication_benchmark_results.json');
  fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));

  console.log('\n' + '='.repeat(75));
  console.log('  HELD-OUT EVALUATION SET BENCHMARK RESULTS  ');
  console.log('='.repeat(75));
  console.log('| Signal | Precision | Recall | F1 Score | FPR | FNR | Accuracy |');
  console.log('|---|---|---|---|---|---|---|');

  for (const [signal, m] of Object.entries(evalOutput.metrics)) {
    console.log(
      `| ${signal.padEnd(20)} | ${(m.precision * 100).toFixed(1)}% | ${(m.recall * 100).toFixed(1)}% | ${m.f1.toFixed(4)} | ${(m.fpr * 100).toFixed(1)}% | ${(m.fnr * 100).toFixed(1)}% | ${(m.accuracy * 100).toFixed(1)}% |`
    );
  }

  console.log('\nLatency Performance:');
  console.log(`  Dual Hash Compute (dHash + pHash): ${evalOutput.latencies.avgHashComputeMs} ms`);
  console.log(`  Composite Scoring:                 ${evalOutput.latencies.avgScoringMs} ms`);
  console.log(`  Total Processing Latency / Pair:   ${evalOutput.latencies.totalPipelineMs} ms`);
  console.log(`\nDetailed validation report saved to: ${outputPath}`);
}

main().catch((err) => {
  console.error('Benchmark validation failed:', err);
  process.exit(1);
});

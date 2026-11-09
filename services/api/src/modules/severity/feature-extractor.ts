/**
 * CivicPulse AI - Road Damage Severity Feature Extraction Pipeline
 * Extracts detection-level, image-level, and incident-level spatial-geometric features
 * from computer vision bounding boxes and incident telemetry.
 *
 * NOTE: Bounding-box normalized area represents image sensor coverage ratio [0.0, 1.0],
 * NOT calibrated physical ground area in square meters.
 */

export interface BoundingBoxDetection {
  label: string; // e.g. "D40", "pothole", "D00", "D20", "D10"
  confidence: number;
  // Normalized coordinates in [0.0, 1.0]
  xCenter?: number;
  yCenter?: number;
  width?: number;
  height?: number;
  // Pixel coordinates if raw
  xmin?: number;
  ymin?: number;
  xmax?: number;
  ymax?: number;
}

export interface DetectionLevelFeatures {
  category: string;
  normalizedWidth: number;
  normalizedHeight: number;
  normalizedArea: number; // width * height [0.0, 1.0]
  aspectRatio: number; // width / height
  confidence: number;
}

export interface ImageLevelFeatures {
  totalDefectCount: number;
  categoryCounts: Record<string, number>;
  sumNormalizedArea: number;
  maxNormalizedArea: number;
  meanNormalizedArea: number;
  categoryDiversity: number;
  spatialConcentration: number; // variance of defect centers
  detections: DetectionLevelFeatures[];
}

export interface IncidentLevelFeatures {
  incidentId: string;
  incidentNumber: string;
  imageCount: number;
  totalDefects: number;
  categoryDistribution: Record<string, number>;
  primaryDamageCategory: string;
  maxDefectArea: number;
  totalDefectArea: number;
  meanDetectionConfidence: number;
  hasInspectorVerification: boolean;
  verifiedCategory: string | null;
  verifiedSeverity: string | null;
  imageFeatures: ImageLevelFeatures[];
  evidenceSufficient: boolean;
  insufficiencyReasons: string[];
}

/**
 * Normalizes damage category names to canonical codes (D00, D10, D20, D40, other).
 */
export function normalizeDamageCategory(cat: string | null | undefined): string {
  if (!cat) return 'unknown';
  const c = cat.toLowerCase().trim();
  if (c.includes('40') || c.includes('pothole')) return 'D40';
  if (c.includes('20') || c.includes('alligator')) return 'D20';
  if (c.includes('10') || c.includes('transverse')) return 'D10';
  if (c.includes('00') || c.includes('longitudinal') || c.includes('crack')) return 'D00';
  return 'other';
}

/**
 * Extracts normalized detection-level features from a raw bounding box.
 */
export function extractDetectionFeatures(
  det: BoundingBoxDetection,
  imageWidth: number = 720,
  imageHeight: number = 720
): DetectionLevelFeatures {
  let normW = 0.0;
  let normH = 0.0;

  if (det.width !== undefined && det.height !== undefined) {
    // Already normalized if <= 1.0
    normW = det.width > 1.0 ? det.width / Math.max(imageWidth, 1) : det.width;
    normH = det.height > 1.0 ? det.height / Math.max(imageHeight, 1) : det.height;
  } else if (
    det.xmin !== undefined &&
    det.xmax !== undefined &&
    det.ymin !== undefined &&
    det.ymax !== undefined
  ) {
    const rawW = Math.max(0, det.xmax - det.xmin);
    const rawH = Math.max(0, det.ymax - det.ymin);
    normW = rawW > 1.0 ? rawW / Math.max(imageWidth, 1) : rawW;
    normH = rawH > 1.0 ? rawH / Math.max(imageHeight, 1) : rawH;
  }

  normW = Math.min(Math.max(normW, 0.0), 1.0);
  normH = Math.min(Math.max(normH, 0.0), 1.0);

  const normArea = Math.min(Math.max(normW * normH, 0.0), 1.0);
  const aspect = normH > 0 ? normW / normH : 1.0;
  const conf = Math.min(Math.max(det.confidence || 0.0, 0.0), 1.0);

  return {
    category: normalizeDamageCategory(det.label),
    normalizedWidth: Number(normW.toFixed(4)),
    normalizedHeight: Number(normH.toFixed(4)),
    normalizedArea: Number(normArea.toFixed(5)),
    aspectRatio: Number(aspect.toFixed(3)),
    confidence: Number(conf.toFixed(4)),
  };
}

/**
 * Aggregates detection features into image-level summary features.
 */
export function extractImageLevelFeatures(
  detections: BoundingBoxDetection[],
  imageWidth: number = 720,
  imageHeight: number = 720
): ImageLevelFeatures {
  if (!detections || detections.length === 0) {
    return {
      totalDefectCount: 0,
      categoryCounts: {},
      sumNormalizedArea: 0.0,
      maxNormalizedArea: 0.0,
      meanNormalizedArea: 0.0,
      categoryDiversity: 0,
      spatialConcentration: 0.0,
      detections: [],
    };
  }

  const extractedList: DetectionLevelFeatures[] = [];
  const catCounts: Record<string, number> = {};
  let sumArea = 0.0;
  let maxArea = 0.0;

  const centersX: number[] = [];
  const centersY: number[] = [];

  for (const rawDet of detections) {
    const feat = extractDetectionFeatures(rawDet, imageWidth, imageHeight);
    extractedList.push(feat);

    catCounts[feat.category] = (catCounts[feat.category] || 0) + 1;
    sumArea += feat.normalizedArea;
    if (feat.normalizedArea > maxArea) maxArea = feat.normalizedArea;

    if (rawDet.xCenter !== undefined && rawDet.yCenter !== undefined) {
      centersX.push(rawDet.xCenter);
      centersY.push(rawDet.yCenter);
    }
  }

  const n = extractedList.length;
  const meanArea = n > 0 ? sumArea / n : 0.0;
  const uniqueCats = Object.keys(catCounts).length;

  // Spatial concentration: variance of centers
  let concentration = 0.0;
  if (centersX.length > 1) {
    const meanX = centersX.reduce((a, b) => a + b, 0) / centersX.length;
    const meanY = centersY.reduce((a, b) => a + b, 0) / centersY.length;
    const varX = centersX.reduce((acc, x) => acc + (x - meanX) ** 2, 0) / centersX.length;
    const varY = centersY.reduce((acc, y) => acc + (y - meanY) ** 2, 0) / centersY.length;
    concentration = Math.sqrt(varX + varY);
  }

  return {
    totalDefectCount: n,
    categoryCounts: catCounts,
    sumNormalizedArea: Number(sumArea.toFixed(4)),
    maxNormalizedArea: Number(maxArea.toFixed(4)),
    meanNormalizedArea: Number(meanArea.toFixed(4)),
    categoryDiversity: uniqueCats,
    spatialConcentration: Number(concentration.toFixed(4)),
    detections: extractedList,
  };
}

/**
 * Constructs an incident-level feature profile by rolling up image and prediction records.
 */
export function extractIncidentLevelFeatures(params: {
  incident: {
    id: string;
    incidentNumber: string;
    damageType: string;
    severity?: string | null;
  };
  images: { id: string; originalFilename: string }[];
  predictions: {
    id: string;
    predictionType: string;
    predictedClass: string;
    confidence: number;
    predictionPayload?: any;
  }[];
  verifications: {
    id: string;
    verifiedClass: string;
    verifiedSeverity: string;
    remarks?: string | null;
  }[];
}): IncidentLevelFeatures {
  const { incident, images, predictions, verifications } = params;
  const insufficiencyReasons: string[] = [];

  const imageFeatList: ImageLevelFeatures[] = [];
  const aggregatedCatDist: Record<string, number> = {};
  let totalDetectionsCount = 0;
  let totalArea = 0.0;
  let maxArea = 0.0;
  let sumConfidence = 0.0;
  let confCount = 0;

  // 1. Process AI predictions and extract bounding boxes
  for (const pred of predictions) {
    sumConfidence += pred.confidence;
    confCount++;

    const payload = pred.predictionPayload as any;
    let detections: BoundingBoxDetection[] = [];

    if (payload && Array.isArray(payload.boxes)) {
      // Detected bounding boxes
      detections = payload.boxes.map((b: any) => ({
        label: b.label || b.class || pred.predictedClass,
        confidence: b.score || b.confidence || pred.confidence,
        xmin: b.xmin,
        ymin: b.ymin,
        xmax: b.xmax,
        ymax: b.ymax,
        width: b.width,
        height: b.height,
        xCenter: b.xCenter,
        yCenter: b.yCenter,
      }));
    } else if (payload && Array.isArray(payload.detections)) {
      detections = payload.detections;
    } else {
      // Fallback: single classification prediction represented as a default frame box
      detections = [
        {
          label: pred.predictedClass,
          confidence: pred.confidence,
          width: 0.15,
          height: 0.15,
          xCenter: 0.5,
          yCenter: 0.5,
        },
      ];
    }

    const imgFeats = extractImageLevelFeatures(detections);
    imageFeatList.push(imgFeats);

    totalDetectionsCount += imgFeats.totalDefectCount;
    totalArea += imgFeats.sumNormalizedArea;
    if (imgFeats.maxNormalizedArea > maxArea) maxArea = imgFeats.maxNormalizedArea;

    for (const [cat, count] of Object.entries(imgFeats.categoryCounts)) {
      aggregatedCatDist[cat] = (aggregatedCatDist[cat] || 0) + count;
    }
  }

  // 2. Evaluate evidence sufficiency
  if (images.length === 0) {
    insufficiencyReasons.push('No field evidence photographs attached to incident.');
  }
  if (predictions.length === 0 && verifications.length === 0) {
    insufficiencyReasons.push('Zero computer vision predictions or field inspector observations.');
  }

  // Determine predominant damage category
  let predominantCategory = normalizeDamageCategory(incident.damageType);
  if (Object.keys(aggregatedCatDist).length > 0) {
    predominantCategory = Object.entries(aggregatedCatDist).sort(
      (a, b) => b[1] - a[1]
    )[0][0];
  }

  const latestVerif = verifications.length > 0 ? verifications[0] : null;

  return {
    incidentId: incident.id,
    incidentNumber: incident.incidentNumber,
    imageCount: images.length,
    totalDefects: totalDetectionsCount,
    categoryDistribution: aggregatedCatDist,
    primaryDamageCategory: predominantCategory,
    maxDefectArea: Number(maxArea.toFixed(4)),
    totalDefectArea: Number(totalArea.toFixed(4)),
    meanDetectionConfidence: confCount > 0 ? Number((sumConfidence / confCount).toFixed(4)) : 0.0,
    hasInspectorVerification: latestVerif !== null,
    verifiedCategory: latestVerif ? latestVerif.verifiedClass : null,
    verifiedSeverity: latestVerif ? latestVerif.verifiedSeverity : null,
    imageFeatures: imageFeatList,
    evidenceSufficient: insufficiencyReasons.length === 0,
    insufficiencyReasons,
  };
}


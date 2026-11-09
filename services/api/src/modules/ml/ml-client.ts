import { env } from '../../config/env';

export interface MLPredictionItem {
  damage_code: string;
  damage_name: string;
  confidence: number;
  severity: string;
  detected: boolean;
  description: string;
}

export interface MLDetectionItem {
  box_id: number;
  damage_code: string;
  damage_name: string;
  confidence: number;
  bbox_normalized: [number, number, number, number];
  bbox_pixels: [number, number, number, number];
}

export interface MLInferenceResult {
  success: boolean;
  model: {
    name: string;
    version: string;
    framework: string;
    device: string;
  };
  predictions: MLPredictionItem[];
  detections: MLDetectionItem[];
  summary: {
    has_damage: boolean;
    primary_distress: string | null;
    overall_severity: string;
    max_confidence: number;
  };
  metrics: {
    image_width: number;
    image_height: number;
    classification_latency_ms: number;
    detection_latency_ms: number;
    total_latency_ms: number;
    detection_count: number;
  };
}

export async function requestMLInference(
  imageBuffer: Buffer,
  filename: string,
  contentType: string,
  confidenceThreshold: number = 0.35
): Promise<MLInferenceResult> {
  const formData = new FormData();
  const blob = new Blob([imageBuffer], { type: contentType });
  formData.append('image', blob, filename);
  formData.append('confidence_threshold', confidenceThreshold.toString());

  const mlUrl = `${env.ML_SERVICE_URL}/api/v1/ml/predict/road-damage`;

  const response = await fetch(mlUrl, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`ML Service responded with status ${response.status}: ${errorText}`);
  }

  return (await response.json()) as MLInferenceResult;
}

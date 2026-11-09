"""
CivicPulse AI — Real Road Damage ML Inference API
Provides GPU-accelerated road distress classification and bounding-box localization.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import io

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn

from preprocessing.image_pipeline import ImagePreprocessor, IMAGENET_MEAN, IMAGENET_STD
from models.transfer_learning.classifier import RoadDamageClassifier
from models.detection.detector import RoadDamageDetector

router = APIRouter(prefix="/predict", tags=["Inference"])

# Canonical Damage Taxonomy
DAMAGE_MAP = {
    "D00": {"name": "Longitudinal Crack", "severity": "medium", "description": "Pavement joint separation along travel direction"},
    "D10": {"name": "Transverse Crack", "severity": "low", "description": "Thermal expansion fracture across lane"},
    "D20": {"name": "Alligator Crack", "severity": "high", "description": "Structural fatigue cracking indicating base failure"},
    "D40": {"name": "Pothole", "severity": "critical", "description": "Asphalt cavity creating immediate vehicular hazard"},
}
CLASS_NAMES = ["D00", "D10", "D20", "D40"]

# Global Model Singletons
_classifier: Optional[RoadDamageClassifier] = None
_detector: Optional[RoadDamageDetector] = None
_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_preprocessor = ImagePreprocessor(target_size=(224, 224), normalize=True)


def get_models():
    """Lazily loads and caches GPU model checkpoints."""
    global _classifier, _detector
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / "model_artifacts"
    cls_ckpt = artifacts_dir / "production_model.pt"
    det_ckpt = artifacts_dir / "detector_ssdlite320_best.pt"

    if _classifier is None:
        if cls_ckpt.exists():
            checkpoint = torch.load(cls_ckpt, map_location=_device)
            backbone = checkpoint.get("backbone", "resnet18")
            _classifier = RoadDamageClassifier(backbone_name=backbone, num_classes=4, pretrained=False).to(_device)
            _classifier.load_state_dict(checkpoint["state_dict"])
            _classifier.eval()
            print(f"Loaded production classifier ({backbone}) on {_device}")
        else:
            # Fallback initialized weights if artifact missing
            _classifier = RoadDamageClassifier(backbone_name="resnet18", num_classes=4, pretrained=True).to(_device)
            _classifier.eval()

    if _detector is None:
        if det_ckpt.exists():
            checkpoint = torch.load(det_ckpt, map_location=_device)
            _detector = RoadDamageDetector(pretrained=True, num_classes=5).to(_device)
            _detector.load_state_dict(checkpoint["state_dict"])
            _detector.eval()
            _detector.model.eval()
            print(f"Loaded production detector (SSDLite320) on {_device}")
        else:
            _detector = RoadDamageDetector(pretrained=True, num_classes=5).to(_device)
            _detector.eval()
            _detector.model.eval()

    return _classifier, _detector


class BoundingBoxOut(BaseModel):
    box_id: int
    damage_code: str
    damage_name: str
    confidence: float
    bbox_normalized: List[float] = Field(description="[ymin, xmin, ymax, xmax] in 0..1")
    bbox_pixels: List[int] = Field(description="[ymin, xmin, ymax, xmax] in image pixel space")


class DamagePredictionOut(BaseModel):
    damage_code: str
    damage_name: str
    confidence: float
    severity: str
    detected: bool
    description: str


class PredictionSummaryOut(BaseModel):
    has_damage: bool
    primary_distress: Optional[str]
    overall_severity: str
    max_confidence: float


class ModelInfoOut(BaseModel):
    name: str
    version: str
    framework: str
    device: str


class RoadDamageResponse(BaseModel):
    success: bool
    model: ModelInfoOut
    predictions: List[DamagePredictionOut]
    detections: List[BoundingBoxOut]
    summary: PredictionSummaryOut
    metrics: Dict[str, Any]


@router.post("/road-damage", response_model=RoadDamageResponse)
async def predict_road_damage(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.35),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
):
    """
    Analyzes an uploaded road inspection image and returns detected pavement distress,
    confidence probabilities, bounding-box localizations, and overall severity assessment.
    """
    # 1. Validate File Metadata
    if image.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type: {image.content_type}. Must be JPEG, PNG, or WebP.",
        )

    image_bytes = await image.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum threshold of 15MB.",
        )

    t_total_start = time.perf_counter()

    # 2. Decode and Validate PIL Image
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        orig_w, orig_h = pil_img.size
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Corrupted or unreadable image data: {str(e)}",
        )

    classifier, detector = get_models()

    # 3. Classification Forward Pass
    t_cls_start = time.perf_counter()
    tensor_np = _preprocessor.preprocess_image(pil_img, augment=False)
    input_tensor = torch.from_numpy(tensor_np).unsqueeze(0).to(_device)

    with torch.no_grad():
        cls_logits = classifier(input_tensor)
        cls_probs = torch.sigmoid(cls_logits)[0].cpu().numpy()
        torch.cuda.synchronize() if _device.type == "cuda" else None

    t_cls_ms = (time.perf_counter() - t_cls_start) * 1000.0

    # 4. Object Detection Forward Pass
    t_det_start = time.perf_counter()
    det_img = pil_img.resize((320, 320))
    det_tensor = (
        torch.from_numpy(np.asarray(det_img, dtype=np.float32) / 255.0)
        .permute(2, 0, 1)
        .to(_device)
    )

    with torch.no_grad():
        det_output = detector([det_tensor])[0]
        torch.cuda.synchronize() if _device.type == "cuda" else None

    t_det_ms = (time.perf_counter() - t_det_start) * 1000.0

    # 5. Parse Detections
    boxes = det_output["boxes"].cpu().numpy()
    scores = det_output["scores"].cpu().numpy()
    labels = det_output["labels"].cpu().numpy()

    detections: List[BoundingBoxOut] = []
    box_id = 1
    det_classes_found = set()

    for box, score, lbl in zip(boxes, scores, labels):
        if score >= confidence_threshold and 1 <= lbl <= 4:
            code = CLASS_NAMES[lbl - 1]
            det_classes_found.add(code)
            # Box in 320x320 space -> normalize to [0..1]
            xmin_norm = max(0.0, min(1.0, float(box[0] / 320.0)))
            ymin_norm = max(0.0, min(1.0, float(box[1] / 320.0)))
            xmax_norm = max(0.0, min(1.0, float(box[2] / 320.0)))
            ymax_norm = max(0.0, min(1.0, float(box[3] / 320.0)))

            px_xmin = int(xmin_norm * orig_w)
            px_ymin = int(ymin_norm * orig_h)
            px_xmax = int(xmax_norm * orig_w)
            px_ymax = int(ymax_norm * orig_h)

            detections.append(
                BoundingBoxOut(
                    box_id=box_id,
                    damage_code=code,
                    damage_name=DAMAGE_MAP[code]["name"],
                    confidence=round(float(score), 4),
                    bbox_normalized=[round(ymin_norm, 4), round(xmin_norm, 4), round(ymax_norm, 4), round(xmax_norm, 4)],
                    bbox_pixels=[px_ymin, px_xmin, px_ymax, px_xmax],
                )
            )
            box_id += 1

    # 6. Build Multi-Label Predictions
    predictions: List[DamagePredictionOut] = []
    has_damage = False
    max_conf = 0.0
    detected_codes = []

    for i, code in enumerate(CLASS_NAMES):
        prob = float(cls_probs[i])
        # If detector found this class with higher confidence, boost
        if code in det_classes_found:
            matching_scores = [d.confidence for d in detections if d.damage_code == code]
            if matching_scores:
                prob = max(prob, max(matching_scores))

        is_detected = prob >= confidence_threshold
        if is_detected:
            has_damage = True
            detected_codes.append(code)

        if prob > max_conf:
            max_conf = prob

        info = DAMAGE_MAP[code]
        predictions.append(
            DamagePredictionOut(
                damage_code=code,
                damage_name=info["name"],
                confidence=round(prob, 4),
                severity=info["severity"],
                detected=is_detected,
                description=info["description"],
            )
        )

    # 7. Assess Overall Severity
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
    current_sev = "none"

    if has_damage:
        for c in detected_codes:
            cand = DAMAGE_MAP[c]["severity"]
            if severity_order[cand] > severity_order[current_sev]:
                current_sev = cand
    else:
        current_sev = "none"

    primary = None
    if detected_codes:
        # Sort by confidence
        sorted_codes = sorted(detected_codes, key=lambda c: cls_probs[CLASS_NAMES.index(c)], reverse=True)
        primary = f"{sorted_codes[0]} ({DAMAGE_MAP[sorted_codes[0]]['name']})"

    total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0

    device_name = f"{_device} ({torch.cuda.get_device_name(0)})" if _device.type == "cuda" else str(_device)

    return RoadDamageResponse(
        success=True,
        model=ModelInfoOut(
            name="road-damage-classifier-resnet18",
            version="1.0.0",
            framework=f"PyTorch {torch.__version__}",
            device=device_name,
        ),
        predictions=predictions,
        detections=detections,
        summary=PredictionSummaryOut(
            has_damage=has_damage,
            primary_distress=primary,
            overall_severity=current_sev,
            max_confidence=round(max_conf, 4),
        ),
        metrics={
            "image_width": orig_w,
            "image_height": orig_h,
            "classification_latency_ms": round(t_cls_ms, 2),
            "detection_latency_ms": round(t_det_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "detection_count": len(detections),
            "latitude": latitude,
            "longitude": longitude,
        },
    )

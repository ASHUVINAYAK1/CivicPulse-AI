"""
CivicPulse AI — Phase 2.1 Inference and Latency Independent Audit Script
Accurately benchmarks every layer of the CV and API stack:
1. Image Preprocessing
2. Classification Inference (GPU synchronized)
3. Detection Inference (GPU synchronized)
4. FastAPI Direct HTTP Request Latency
5. Express-to-FastAPI Request Latency
6. Complete Upload -> MinIO -> GPU ML -> PostgreSQL Persistence Latency
Runs multiple iterations and reports median and p95 latencies.
"""

import sys
import time
import json
import urllib.request
import io
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.image_pipeline import ImagePreprocessor
from models.transfer_learning.classifier import RoadDamageClassifier
from models.detection.detector import RoadDamageDetector


def benchmark_preprocessing(preprocessor: ImagePreprocessor, image_bytes: bytes, n_iters: int = 50) -> List[float]:
    # Warm-up
    for _ in range(5):
        _ = preprocessor.preprocess_image(image_bytes)

    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = preprocessor.preprocess_image(image_bytes)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def benchmark_gpu_classification(model: RoadDamageClassifier, input_tensor: torch.Tensor, device: torch.device, n_iters: int = 50) -> List[float]:
    model.eval()
    tensor_gpu = input_tensor.to(device)

    # Warm-up (10 iterations)
    with torch.no_grad():
        for _ in range(10):
            _ = model(tensor_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(tensor_gpu)
            _ = torch.sigmoid(logits)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def benchmark_gpu_detection(detector: RoadDamageDetector, det_tensor: torch.Tensor, device: torch.device, n_iters: int = 50) -> List[float]:
    detector.eval()
    tensor_gpu = det_tensor.to(device)

    # Warm-up (10 iterations)
    with torch.no_grad():
        for _ in range(10):
            _ = detector([tensor_gpu])
            if device.type == "cuda":
                torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = detector([tensor_gpu])
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def post_multipart(url: str, filename: str, file_bytes: bytes, fields: Dict[str, str] = {}) -> Tuple[int, bytes, float]:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()

    for k, v in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        body.extend(f"{v}\r\n".encode())

    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode())
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        content = resp.read()
        status_code = resp.status
    latency = (time.perf_counter() - t0) * 1000.0
    return status_code, content, latency


def benchmark_http_endpoint(url: str, filename: str, file_bytes: bytes, fields: Dict[str, str], n_iters: int = 30) -> List[float]:
    # Warm-up (2 iterations)
    for _ in range(2):
        try:
            post_multipart(url, filename, file_bytes, fields)
        except Exception as e:
            print(f"Warmup warning for {url}: {e}")

    latencies = []
    for _ in range(n_iters):
        _, _, lat = post_multipart(url, filename, file_bytes, fields)
        latencies.append(lat)
    return latencies


def main():
    print("=" * 70)
    print("  CivicPulse AI — Phase 2.1 Latency & Performance Audit  ")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load real test sample
    sample_path = Path("datasets/raw/RDD2022_India/test/images/India_000038.jpg")
    if not sample_path.exists():
        sample_path = list(Path("datasets/raw/RDD2022_India/test/images").glob("*.jpg"))[0]

    with open(sample_path, "rb") as f:
        image_bytes = f.read()

    print(f"Benchmark input file: {sample_path.name} ({len(image_bytes) / 1024:.1f} KB)")

    # 1. Image Preprocessing
    preprocessor = ImagePreprocessor(target_size=(224, 224), normalize=True)
    prep_lats = benchmark_preprocessing(preprocessor, image_bytes, n_iters=50)

    # 2. PyTorch GPU Classification (ResNet-18)
    artifacts_dir = Path("services/ml/model_artifacts")
    cls_ckpt = torch.load(artifacts_dir / "production_model.pt", map_location=device)
    classifier = RoadDamageClassifier(backbone_name="resnet18", num_classes=4, pretrained=False).to(device)
    classifier.load_state_dict(cls_ckpt["state_dict"])

    raw_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_np = preprocessor.preprocess_image(raw_pil)
    input_tensor = torch.from_numpy(tensor_np).unsqueeze(0)
    cls_lats = benchmark_gpu_classification(classifier, input_tensor, device, n_iters=50)

    # 3. PyTorch GPU Detection (SSDLite320)
    det_ckpt = torch.load(artifacts_dir / "detector_ssdlite320_best.pt", map_location=device)
    detector = RoadDamageDetector(pretrained=True, num_classes=5).to(device)
    detector.load_state_dict(det_ckpt["state_dict"])

    det_pil = raw_pil.resize((320, 320))
    det_tensor = torch.from_numpy(np.asarray(det_pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
    det_lats = benchmark_gpu_detection(detector, det_tensor, device, n_iters=50)

    # 4. FastAPI Direct HTTP Request Latency
    print("Benchmarking FastAPI direct endpoint via 127.0.0.1 (IPv4 direct)...")
    fastapi_lats = benchmark_http_endpoint(
        "http://127.0.0.1:8000/api/v1/ml/predict/road-damage",
        sample_path.name,
        image_bytes,
        {"confidence_threshold": "0.35"},
        n_iters=30,
    )

    # Also benchmark localhost to record the Windows IPv6 fallback penalty
    print("Benchmarking FastAPI endpoint via localhost (Windows IPv6 fallback check)...")
    fastapi_localhost_lats = benchmark_http_endpoint(
        "http://localhost:8000/api/v1/ml/predict/road-damage",
        sample_path.name,
        image_bytes,
        {"confidence_threshold": "0.35"},
        n_iters=5,
    )

    # 5. Express-to-FastAPI Proxy Latency
    print("Benchmarking Express proxy endpoint via 127.0.0.1...")
    express_proxy_lats = benchmark_http_endpoint(
        "http://127.0.0.1:5000/api/v1/ml/predict",
        sample_path.name,
        image_bytes,
        {"confidence_threshold": "0.35"},
        n_iters=30,
    )

    # 6. Complete Upload -> MinIO -> GPU Inference -> PostgreSQL Persistence
    print("Benchmarking Complete Incident Reporting & Persistence via 127.0.0.1...")
    e2e_lats = benchmark_http_endpoint(
        "http://127.0.0.1:5000/api/v1/incidents/report",
        sample_path.name,
        image_bytes,
        {
            "roadName": "Benchmark Corridor",
            "address": "Ring Road Sector 4",
            "latitude": "12.9716",
            "longitude": "77.5946",
        },
        n_iters=25,
    )

    def stats(arr):
        return {
            "median_ms": round(float(np.median(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
            "std_ms": round(float(np.std(arr)), 2),
            "iterations": len(arr),
        }

    summary = {
        "device": f"{device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})",
        "gpu_warmup_applied": True,
        "cuda_synchronization_applied": True,
        "breakdown": {
            "1_image_preprocessing": stats(prep_lats),
            "2_classification_inference_resnet18": stats(cls_lats),
            "3_detection_inference_ssdlite320": stats(det_lats),
            "4_fastapi_direct_request_ipv4": stats(fastapi_lats),
            "4b_fastapi_direct_request_localhost_ipv6_penalty": stats(fastapi_localhost_lats),
            "5_express_to_fastapi_proxy": stats(express_proxy_lats),
            "6_complete_pipeline_minio_and_postgres": stats(e2e_lats),
        },
    }

    out_file = artifacts_dir / "latency_audit_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nLatency audit complete. Results saved to {out_file}")
    for k, v in summary["breakdown"].items():
        print(f"  {k}: median={v['median_ms']}ms | p95={v['p95_ms']}ms")


if __name__ == "__main__":
    main()

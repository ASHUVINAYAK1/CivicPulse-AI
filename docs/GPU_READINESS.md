# CivicPulse AI — GPU Readiness & Hardware Audit Report

**Date of Audit**: 2026-11-07  
**Auditor**: Senior Computer Vision Engineer & PyTorch Specialist  
**Verification Result**: **ACCELERATION VERIFIED & FUNCTIONAL**

---

## 1. Hardware Specifications

| Attribute | Detected Value | Verification Method |
|---|---|---|
| **GPU Model** | **NVIDIA GeForce RTX 5060 Ti** | `nvidia-smi` / `torch.cuda.get_device_name(0)` |
| **GPU Architecture** | **Blackwell** (`sm_120`) | `torch.cuda.get_device_capability(0)` -> `(12, 0)` |
| **Total Dedicated VRAM** | **8,151 MiB (8.0 GB GDDR7)** | `nvidia-smi` |
| **Available Free VRAM** | **~7,280 MiB (>7.1 GB Free)** | `nvidia-smi` at baseline |
| **Power Target / Cap** | 180 Watts (Current idle: 9W) | `nvidia-smi` |
| **Temperature** | 35°C (Idle) | `nvidia-smi` |
| **Host System RAM** | 16.0 GB (3.14 GB free at start of session) | `Win32_OperatingSystem` |
| **Host Disk Space (C:)** | 74.4 GB free space available | `PSDrive C:` |

---

## 2. Driver & CUDA Compatibility

| Component | Version | Compatibility Assessment |
|---|---|---|
| **NVIDIA Display Driver** | `610.88` (WDDM) | Fully supports Blackwell GeForce RTX 50-series |
| **CUDA Driver / UMD** | `13.3` | Backwards-compatible with CUDA 12.x runtimes |
| **WSL2 GPU Support** | `docker-desktop` WSL2 v2 | Direct Windows host access via native WDDM driver |
| **Python Version** | Python `3.11.9` (64-bit) | Supported across PyTorch 2.x wheels |
| **PyTorch Runtime** | **`2.11.0+cu128`** | Bundles CUDA 12.8 runtime (`cu128`) supporting `sm_120` |
| **TorchVision** | **`0.26.0+cu128`** | Vision models and CUDA C++ extensions operational |

---

## 3. Direct GPU Execution Benchmark

A live matrix multiplication benchmark (`torch.matmul(x, x)` for a 1,000 × 1,000 float32 tensor) was executed directly on `device='cuda:0'`:

```
Device Name: NVIDIA GeForce RTX 5060 Ti
Compute Capability: (12, 0)
CUDA Kernel Execution: SUCCESS
Output Shape: torch.Size([1000, 1000])
Allocated Memory: 15.75 MB
Cached Memory: 20.0 MB
Status: FUNCTIONAL & VERIFIED
```

---

## 4. Workload Sizing & VRAM Recommendations

For the 8.0 GB GDDR7 VRAM budget on the RTX 5060 Ti:
- **MobileNetV3-Small / EfficientNet-B0 (Classification)**:
  - Recommended Batch Size: `32` to `64`
  - Estimated VRAM Footprint: ~1.2 GB to 2.4 GB
  - Training Throughput: ~180-250 images/second
- **ResNet-18 (Classification)**:
  - Recommended Batch Size: `32`
  - Estimated VRAM Footprint: ~1.8 GB
- **SSDLite320 / Lightweight Object Detection**:
  - Recommended Batch Size: `16`
  - Estimated VRAM Footprint: ~2.8 GB to 3.5 GB


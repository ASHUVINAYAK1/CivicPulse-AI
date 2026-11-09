'use client';

import React, { useState } from 'react';
import { Cpu, Activity, CheckCircle2, Zap, Layers, BarChart3, Clock, AlertCircle } from 'lucide-react';

const BENCHMARK_MODELS = [
  {
    name: 'road-damage-classifier-resnet18',
    architecture: 'ResNet-18 (Deep Residual Learning)',
    status: 'production',
    parameters: '11,178,564',
    testMacroF1: '0.4529',
    exactMatch: '71.70%',
    binaryAcc: '81.99%',
    latency: '0.66 ms',
    potholeF1: '0.5000',
    alligatorF1: '0.7885',
    crackF1: '0.5230',
    vram: '1.58 GB',
    notes: 'Selected Winner: Best critical pothole recall and overall distress F1 on Indian road geometry.',
  },
  {
    name: 'road-damage-classifier-efficientnet-b0',
    architecture: 'EfficientNet-B0 (Compound Scaling)',
    status: 'evaluated',
    parameters: '4,012,672',
    testMacroF1: '0.4394',
    exactMatch: '70.50%',
    binaryAcc: '81.65%',
    latency: '0.94 ms',
    potholeF1: '0.4509',
    alligatorF1: '0.8032',
    crackF1: '0.5036',
    vram: '5.49 GB',
    notes: 'Strongest alligator crack precision, but higher VRAM footprint during multi-scale backprop.',
  },
  {
    name: 'road-damage-classifier-mobilenetv3',
    architecture: 'MobileNetV3-Small (Inverted Residuals)',
    status: 'evaluated',
    parameters: '1,521,956',
    testMacroF1: '0.3545',
    exactMatch: '66.38%',
    binaryAcc: '78.39%',
    latency: '0.48 ms',
    potholeF1: '0.2883',
    alligatorF1: '0.7390',
    crackF1: '0.3908',
    vram: '1.05 GB',
    notes: 'Fastest latency (0.48ms), lowest VRAM. Suitable for low-power edge IoT dashcam units.',
  },
  {
    name: 'road-damage-detector-ssdlite320',
    architecture: 'SSDLite320 (MobileNetV3-Large)',
    status: 'experimental',
    parameters: '~3,400,000',
    testMacroF1: 'mAP@50: 8.98%',
    exactMatch: 'mAP@95: 2.85%',
    binaryAcc: 'IoU: 0.626',
    latency: '10.01 ms',
    potholeF1: 'AP: 0.018',
    alligatorF1: 'AP: 0.336',
    crackF1: 'AP: 0.005',
    vram: '2.84 GB',
    notes: 'Audited Status: EXPERIMENTAL. High recall on D20 alligator distress (89.2%), but coarse proposals and high false alarms on background textures require further epoch convergence before full production status.',
  },
];

export default function ModelsPage() {
  const [selectedModel, setSelectedModel] = useState(BENCHMARK_MODELS[0]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Cpu className="h-7 w-7 text-emerald-400" />
            AI Model Registry & GPU Benchmarks
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Validated road distress models trained on NVIDIA GeForce RTX 5060 Ti with RDD2022 India Ground Truth.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            PyTorch 2.11.0+cu128 Live
          </span>
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
            MLflow 3.18 Tracking
          </span>
        </div>
      </div>

      {/* Hardware Specs Banner */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <p className="text-xs text-slate-400 font-medium">Hardware Accelerator</p>
          <p className="mt-1 text-lg font-bold text-white">RTX 5060 Ti</p>
          <p className="text-[11px] text-slate-500">Blackwell Architecture (sm_120)</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <p className="text-xs text-slate-400 font-medium">Dedicated GDDR7 VRAM</p>
          <p className="mt-1 text-lg font-bold text-emerald-400">8,151 MiB</p>
          <p className="text-[11px] text-slate-500">High-Bandwidth Bus</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <p className="text-xs text-slate-400 font-medium">Production Latency</p>
          <p className="mt-1 text-lg font-bold text-cyan-400">0.66 ms</p>
          <p className="text-[11px] text-slate-500">~1,515 images/sec GPU throughput</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <p className="text-xs text-slate-400 font-medium">Ground-Truth Dataset</p>
          <p className="mt-1 text-lg font-bold text-amber-400">RDD2022 India</p>
          <p className="text-[11px] text-slate-500">7,706 real dashcam road images</p>
        </div>
      </div>

      {/* Model Benchmark Comparison Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-emerald-400" />
            <h2 className="text-sm font-semibold text-white">Comparative Architecture Evaluation (RDD2022 Test Split)</h2>
          </div>
          <span className="text-xs text-slate-400">1,166 Unseen Test Images</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="px-4 py-3 font-semibold">Model Name & Architecture</th>
                <th className="px-4 py-3 font-semibold">Status</th>
                <th className="px-4 py-3 font-semibold">Test Macro F1</th>
                <th className="px-4 py-3 font-semibold">D40 Pothole F1</th>
                <th className="px-4 py-3 font-semibold">D20 Alligator F1</th>
                <th className="px-4 py-3 font-semibold">Exact Match</th>
                <th className="px-4 py-3 font-semibold">Binary Damage Acc</th>
                <th className="px-4 py-3 font-semibold">Inference Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {BENCHMARK_MODELS.map((m) => (
                <tr
                  key={m.name}
                  onClick={() => setSelectedModel(m)}
                  className={`cursor-pointer transition-colors ${
                    selectedModel.name === m.name
                      ? 'bg-emerald-500/10'
                      : 'hover:bg-slate-800/40'
                  }`}
                >
                  <td className="px-4 py-3">
                    <div className="font-semibold text-white">{m.name}</div>
                    <div className="text-[11px] text-slate-400">{m.architecture}</div>
                  </td>
                  <td className="px-4 py-3">
                    {m.status === 'production' ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                        <CheckCircle2 className="h-3 w-3" /> PRODUCTION
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-300">
                        EVALUATED
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono font-bold text-white text-sm">{m.testMacroF1}</td>
                  <td className="px-4 py-3 font-mono text-amber-300 font-semibold">{m.potholeF1}</td>
                  <td className="px-4 py-3 font-mono text-cyan-300">{m.alligatorF1}</td>
                  <td className="px-4 py-3 font-mono text-slate-300">{m.exactMatch}</td>
                  <td className="px-4 py-3 font-mono text-slate-300">{m.binaryAcc}</td>
                  <td className="px-4 py-3 font-mono text-emerald-400 font-semibold">{m.latency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Model Detail Panel */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
              <Layers className="h-6 w-6" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-white">{selectedModel.name}</h3>
              <p className="text-xs text-slate-400">{selectedModel.architecture}</p>
            </div>
          </div>
          <span className="text-xs text-slate-400">
            Parameters: <strong className="text-white">{selectedModel.parameters}</strong>
          </span>
        </div>

        <div className="p-3.5 rounded-lg bg-slate-800/40 border border-slate-800 text-xs text-slate-300">
          <span className="font-semibold text-emerald-400">Engineering Rationale:</span> {selectedModel.notes}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-800/60">
            <p className="text-slate-400">D00 Longitudinal Crack</p>
            <p className="mt-1 text-sm font-bold text-white">{selectedModel.crackF1}</p>
          </div>
          <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-800/60">
            <p className="text-slate-400">D10 Transverse Crack</p>
            <p className="mt-1 text-sm font-bold text-white">0.0000 (Rare)</p>
          </div>
          <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-800/60">
            <p className="text-slate-400">D20 Alligator Crack</p>
            <p className="mt-1 text-sm font-bold text-white">{selectedModel.alligatorF1}</p>
          </div>
          <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-800/60">
            <p className="text-slate-400">D40 Critical Pothole</p>
            <p className="mt-1 text-sm font-bold text-amber-400">{selectedModel.potholeF1}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

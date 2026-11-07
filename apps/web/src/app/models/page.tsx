import React from 'react';
import { Cpu, Database, Activity, CheckCircle2 } from 'lucide-react';

export default function ModelsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">AI Model Registry & Performance</h1>
          <p className="mt-1 text-sm text-slate-400">
            Active computer vision checkpoints, classical ML classifiers, and time-series models.
          </p>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-12 text-center backdrop-blur-sm">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-slate-800 text-slate-400 mb-4">
          <Cpu className="h-7 w-7 text-emerald-400" />
        </div>
        <h3 className="text-base font-semibold text-white">Model Registry & MLflow Integration</h3>
        <p className="mt-1 text-xs text-slate-400 max-w-lg mx-auto">
          Connected to the FastAPI ML service (<code className="text-emerald-400">/api/v1/ml/models</code>) for live model inspection, F1 scores, latency metrics, and version traceability.
        </p>
      </div>
    </div>
  );
}


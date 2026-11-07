import React from 'react';
import { AlertTriangle, Filter, Plus, Search } from 'lucide-react';

export default function IncidentsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Incident Registry</h1>
          <p className="mt-1 text-sm text-slate-400">
            Citizen reports, field inspections, and AI damage verification queue.
          </p>
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white hover:bg-emerald-500 transition-colors shadow-sm"
        >
          <Plus className="h-4 w-4" />
          Report Incident
        </button>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-8 text-center backdrop-blur-sm">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-slate-800 text-slate-400 mb-4">
          <AlertTriangle className="h-6 w-6 text-amber-400" />
        </div>
        <h3 className="text-base font-semibold text-white">Incident Management Module Ready</h3>
        <p className="mt-1 text-xs text-slate-400 max-w-md mx-auto">
          Full incident filtering, status progression (Submitted → AI Processing → Verified → Repaired), and inspector review controls will be connected in Epic 3.
        </p>
      </div>
    </div>
  );
}


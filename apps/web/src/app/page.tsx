'use client';

import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Wrench,
  Cpu,
  TrendingUp,
  MapPin,
  BarChart3,
  Sparkles,
  Info,
  Calendar,
  Layers,
  ArrowUpRight,
} from 'lucide-react';
import { StatCard } from '@/components/ui/stat-card';
import { DAMAGE_TYPES, SEVERITY_LEVELS } from '@/lib/constants';

// Explicitly marked synthetic benchmark data for initial dashboard demonstration
const BENCHMARK_METRICS = {
  totalIncidents: 1247,
  verifiedReports: 892,
  openMaintenanceTasks: 156,
  aiPredictionsRun: 1180,
  criticalHazards: 43,
  avgResolutionDays: 4.2,
};

const DAMAGE_DISTRIBUTION = [
  { code: 'D40', label: 'Potholes', count: 412, percentage: 33, color: 'bg-red-500' },
  { code: 'D00', label: 'Longitudinal Cracks', count: 312, percentage: 25, color: 'bg-amber-500' },
  { code: 'D10', label: 'Transverse Cracks', count: 287, percentage: 23, color: 'bg-orange-500' },
  { code: 'D20', label: 'Alligator Cracks', count: 236, percentage: 19, color: 'bg-purple-500' },
];

const RECENT_INCIDENTS = [
  {
    id: 'INC-2024-001',
    road: 'Main Arterial Road — Km 14.2',
    damageType: 'D40 (Pothole)',
    severity: 'critical',
    aiConfidence: '96.4%',
    status: 'Verified',
    reported: '18 mins ago',
  },
  {
    id: 'INC-2024-002',
    road: 'North Ring Road — Sector 8',
    damageType: 'D20 (Alligator Crack)',
    severity: 'high',
    aiConfidence: '91.2%',
    status: 'Under Review',
    reported: '42 mins ago',
  },
  {
    id: 'INC-2024-003',
    road: 'Industrial Corridor Expressway',
    damageType: 'D00 (Longitudinal)',
    severity: 'medium',
    aiConfidence: '88.5%',
    status: 'In Progress',
    reported: '2 hours ago',
  },
  {
    id: 'INC-2024-004',
    road: 'Central Boulevard & 4th Cross',
    damageType: 'D10 (Transverse)',
    severity: 'low',
    aiConfidence: '94.0%',
    status: 'Scheduled',
    reported: '3 hours ago',
  },
];

export default function DashboardPage() {
  return (
    <div className="space-y-8">
      {/* Page Title & Environment Notice */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Infrastructure Intelligence Overview
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Real-time urban road condition monitoring, computer vision triage, and maintenance operations.
          </p>
        </div>

        {/* Data Origin Disclaimer Badge */}
        <div className="flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-950/20 px-3.5 py-2 text-xs text-amber-300">
          <Info className="h-4 w-4 shrink-0 text-amber-400" />
          <span>
            <strong className="font-semibold">Notice:</strong> Displaying calibrated baseline demonstration data. Connects to PostgreSQL & PostGIS once populated.
          </span>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Incidents"
          value={BENCHMARK_METRICS.totalIncidents.toLocaleString()}
          subtitle="All reported road defects"
          icon={AlertTriangle}
          variant="default"
          trend={{ value: 12.4, isPositive: false, period: 'vs last month' }}
          isPlaceholder
        />
        <StatCard
          title="Verified Reports"
          value={BENCHMARK_METRICS.verifiedReports.toLocaleString()}
          subtitle="Human-validated defects"
          icon={CheckCircle2}
          variant="emerald"
          trend={{ value: 8.1, isPositive: true, period: 'vs last month' }}
          isPlaceholder
        />
        <StatCard
          title="Open Maintenance"
          value={BENCHMARK_METRICS.openMaintenanceTasks}
          subtitle="Tasks assigned & in repair"
          icon={Wrench}
          variant="amber"
          trend={{ value: 4.5, isPositive: false, period: 'active load' }}
          isPlaceholder
        />
        <StatCard
          title="AI Predictions"
          value={BENCHMARK_METRICS.aiPredictionsRun.toLocaleString()}
          subtitle="CNN damage inferences"
          icon={Cpu}
          variant="indigo"
          trend={{ value: 15.2, isPositive: true, period: 'inference rate' }}
          isPlaceholder
        />
      </div>

      {/* Main Analysis Sections */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Geographical Intelligence Placeholder Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 lg:col-span-2 backdrop-blur-sm">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="rounded-lg bg-emerald-500/10 p-2 text-emerald-400 border border-emerald-500/20">
                <MapPin className="h-4 w-4" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-white">Geographical Intelligence & Hotspots</h2>
                <p className="text-xs text-slate-400">PostGIS spatial clustering & incident density map</p>
              </div>
            </div>
            <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-slate-400 border border-slate-700">
              Interactive Map Component
            </span>
          </div>

          <div className="mt-6 flex h-80 flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 bg-slate-950/50 p-6 text-center">
            <div className="rounded-full bg-slate-900 p-4 border border-slate-800 text-slate-500 mb-3">
              <Layers className="h-8 w-8 text-emerald-400/80" />
            </div>
            <h3 className="text-sm font-semibold text-white">Leaflet & PostGIS Map Engine Ready</h3>
            <p className="mt-1 max-w-md text-xs text-slate-400">
              Spatial coordinate rendering will plot live road incidents with color-coded severity markers and K-Means hotspot cluster boundaries.
            </p>
            <div className="mt-4 flex gap-2">
              <span className="rounded bg-slate-900 px-2 py-1 font-mono text-[11px] text-slate-400 border border-slate-800">
                EPSG:4326 PostGIS Geometry
              </span>
              <span className="rounded bg-slate-900 px-2 py-1 font-mono text-[11px] text-slate-400 border border-slate-800">
                OpenStreetMap Tiles
              </span>
            </div>
          </div>
        </div>

        {/* Damage Distribution Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="rounded-lg bg-indigo-500/10 p-2 text-indigo-400 border border-indigo-500/20">
                <BarChart3 className="h-4 w-4" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-white">Damage Classification</h2>
                <p className="text-xs text-slate-400">RDD2022 taxonomy breakdown</p>
              </div>
            </div>
            <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-slate-400 border border-slate-700">
              Placeholder
            </span>
          </div>

          <div className="mt-6 space-y-4">
            {DAMAGE_DISTRIBUTION.map((item) => (
              <div key={item.code} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-300">
                    <span className="font-mono text-slate-400 mr-1.5">{item.code}</span>
                    {item.label}
                  </span>
                  <span className="font-semibold text-white">
                    {item.count} <span className="text-[10px] text-slate-400">({item.percentage}%)</span>
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={`h-full rounded-full ${item.color}`}
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 rounded-lg border border-slate-800/80 bg-slate-950/50 p-3 text-xs text-slate-400">
            <span className="font-medium text-white">Target Accuracy:</span> CNN transfer learning model baseline targets ≥ 85% macro F1 across these 4 categories.
          </div>
        </div>
      </div>

      {/* Secondary Row: Predictive Analytics & Recent Incidents */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Predictive Analytics Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="rounded-lg bg-emerald-500/10 p-2 text-emerald-400 border border-emerald-500/20">
                <TrendingUp className="h-4 w-4" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-white">Predictive Analytics & Forecasting</h2>
                <p className="text-xs text-slate-400">LSTM time-series volume projections</p>
              </div>
            </div>
            <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-slate-400 border border-slate-700">
              Forecasting Model
            </span>
          </div>

          <div className="mt-6 flex h-60 flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 bg-slate-950/50 p-6 text-center">
            <div className="rounded-full bg-slate-900 p-3 border border-slate-800 text-slate-500 mb-2">
              <Sparkles className="h-6 w-6 text-emerald-400" />
            </div>
            <h3 className="text-sm font-semibold text-white">Recharts Time-Series Graph</h3>
            <p className="mt-1 text-xs text-slate-400 max-w-sm">
              Displays 30-day forward incident volume predictions with 95% confidence intervals generated by the FastAPI ML forecasting engine.
            </p>
          </div>
        </div>

        {/* Recent Incidents Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="rounded-lg bg-amber-500/10 p-2 text-amber-400 border border-amber-500/20">
                <Clock className="h-4 w-4" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-white">Recent Triage Feed</h2>
                <p className="text-xs text-slate-400">Newly analyzed road damage incidents</p>
              </div>
            </div>
            <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-slate-400 border border-slate-700">
              Benchmark Feed
            </span>
          </div>

          <div className="mt-4 divide-y divide-slate-800/60">
            {RECENT_INCIDENTS.map((incident) => {
              const severityConfig = SEVERITY_LEVELS[incident.severity as keyof typeof SEVERITY_LEVELS] || SEVERITY_LEVELS.low;
              return (
                <div key={incident.id} className="py-3 flex items-center justify-between text-xs">
                  <div className="min-w-0 pr-4">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-semibold text-slate-200">{incident.id}</span>
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium border ${severityConfig.bg}`}>
                        {severityConfig.label}
                      </span>
                    </div>
                    <p className="mt-1 truncate text-slate-400">{incident.road}</p>
                    <p className="text-[11px] text-slate-500">{incident.damageType}</p>
                  </div>
                  <div className="text-right shrink-0">
                    <span className="inline-flex items-center gap-1 font-mono text-emerald-400 text-[11px]">
                      <Sparkles className="h-3 w-3" />
                      {incident.aiConfidence}
                    </span>
                    <p className="mt-1 text-[10px] text-slate-500">{incident.reported}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

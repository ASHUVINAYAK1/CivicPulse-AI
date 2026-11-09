'use client';

import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  AlertTriangle,
  ShieldAlert,
  Clock,
  Layers,
  MapPin,
  RefreshCw,
  Lock,
  Calendar,
  Database,
  Info,
  CheckCircle2,
  Cpu,
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

const API_BASE = 'http://localhost:5000/api/v1';

interface SummaryData {
  datasetOverview: {
    totalRealIncidents: number;
    longitudinalSpanHours: number;
    longitudinalSpanDays: number;
    earliestTimestamp: string;
    latestTimestamp: string;
    bengaluruCount: number;
    delhiCount: number;
    unassignedCount: number;
  };
  syntheticSeedAudit: {
    totalSyntheticForecasts: number;
    status: string;
    note: string;
  };
  governanceStatus: {
    overallReadiness: string;
    predictiveMaintenanceGated: boolean;
    neuralForecastingLocked: boolean;
    gateReason: string;
  };
}

interface Bucket {
  bucketIndex: number;
  timestamp: string;
  periodLabel: string;
  incidentCount: number;
  verifiedCount: number;
  criticalCount: number;
  damageBreakdown: Record<string, number>;
  observationStatus?: 'OBSERVED_NONZERO' | 'OBSERVED_ZERO' | 'UNOBSERVED' | 'COLLECTION_OUTAGE';
  isZero: boolean;
  isObserved: boolean;
  outageReason?: string;
  displayTimestampLocal?: string;
  projectedCount?: number;
  lowerBound?: number;
  upperBound?: number;
}

interface ModelDescriptor {
  id: string;
  name: string;
  modelType: string;
  description: string;
  framework: string;
  status: 'ACTIVE' | 'LOCKED' | 'EXPERIMENTAL';
  statusReason: string;
  minimumBucketsRequired: Record<string, number>;
  seasonalPeriodsRequired: number;
  formulaDescription: string;
  pros: string[];
  cons: string[];
}

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [buckets, setBuckets] = useState<Bucket[]>([]);
  const [models, setModels] = useState<ModelDescriptor[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedRegion, setSelectedRegion] = useState<'ALL' | 'BENGALURU_METRO' | 'DELHI_NCR'>('ALL');
  const [selectedFrequency, setSelectedFrequency] = useState<'hourly' | 'daily' | 'weekly' | 'monthly'>('hourly');
  const [simulationActive, setSimulationActive] = useState(false);
  const [simModel, setSimModel] = useState<string>('moving_average');
  const [simHorizon, setSimHorizon] = useState<number>(4);
  const [simLoading, setSimLoading] = useState(false);

  // Fetch summary and models
  useEffect(() => {
    async function loadInitial() {
      try {
        const [sumRes, modRes] = await Promise.all([
          fetch(`${API_BASE}/forecasts/summary`).then((r) => r.json()),
          fetch(`${API_BASE}/forecasts/models`).then((r) => r.json()),
        ]);
        if (sumRes.success) setSummary(sumRes.data);
        if (modRes.success) setModels(modRes.data);
      } catch (err) {
        console.error('Failed to load forecasting summary', err);
      }
    }
    loadInitial();
  }, []);

  // Fetch history time series
  useEffect(() => {
    async function loadHistory() {
      setLoading(true);
      try {
        const res = await fetch(
          `${API_BASE}/forecasts/history?region=${selectedRegion}&frequency=${selectedFrequency}`
        );
        const json = await res.json();
        if (json.success && json.data?.buckets) {
          setBuckets(json.data.buckets);
          setSimulationActive(false);
        }
      } catch (err) {
        console.error('Failed to load history', err);
      } finally {
        setLoading(false);
      }
    }
    loadHistory();
  }, [selectedRegion, selectedFrequency]);

  // Handle simulation run
  const runSimulation = async () => {
    setSimLoading(true);
    try {
      const res = await fetch(`${API_BASE}/forecasts/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          region: selectedRegion,
          frequency: selectedFrequency,
          modelType: simModel,
          horizon: simHorizon,
          forceSimulate: true,
        }),
      });
      const json = await res.json();
      if (json.success && json.data?.predictions) {
        // Append forecast points to chart
        const simulatedBuckets: Bucket[] = json.data.predictions.map((p: any, idx: number) => ({
          bucketIndex: buckets.length + idx,
          timestamp: p.period,
          periodLabel: `${p.periodLabel} (Sim)`,
          incidentCount: 0,
          verifiedCount: 0,
          criticalCount: 0,
          damageBreakdown: {},
          isZero: false,
          isObserved: false,
          projectedCount: p.predictedIncidents,
          lowerBound: p.lowerBound,
          upperBound: p.upperBound,
        }));
        setBuckets((prev) => [...prev.filter((b) => b.isObserved), ...simulatedBuckets]);
        setSimulationActive(true);
      }
    } catch (err) {
      console.error('Simulation failed', err);
    } finally {
      setSimLoading(false);
    }
  };

  const clearSimulation = () => {
    setBuckets((prev) => prev.filter((b) => b.isObserved));
    setSimulationActive(false);
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <BarChart3 className="h-7 w-7 text-indigo-400" />
            Analytics & Incident Volume Forecasting
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Temporal aggregation, regional time series, and statistical forecasting governance gates.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
            <ShieldAlert className="h-3.5 w-3.5" />
            Forecasting Gated: DATA_INSUFFICIENT
          </span>
        </div>
      </div>

      {/* Top Governance Alert Banner */}
      <div className="rounded-xl border border-amber-500/40 bg-amber-950/20 p-5 backdrop-blur-sm text-slate-200">
        <div className="flex items-start gap-4">
          <div className="p-2.5 rounded-lg bg-amber-500/20 text-amber-400 shrink-0 mt-0.5">
            <AlertTriangle className="h-6 w-6" />
          </div>
          <div className="space-y-2">
            <h3 className="text-base font-semibold text-amber-200">
              MLOps Gate Enforced: Longitudinal Time-Series Depth Inadequate
            </h3>
            <p className="text-xs text-amber-300/90 leading-relaxed">
              CivicPulse AI data governance mandates that predictive forecasting models must never extrapolate from
              unsupported sample windows. All <strong>66 real incidents</strong> in the database were submitted within a
              single <strong>5.8-hour observation window on November 7, 2026</strong>.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1 text-xs">
              <div className="rounded-lg bg-slate-900/60 p-2.5 border border-slate-800">
                <span className="text-slate-400 font-medium">Statistical Baselines (SMA, Trend):</span>
                <p className="text-slate-200 font-semibold mt-0.5">Requires ≥ 30 Continuous Days</p>
                <p className="text-amber-400/80 text-[11px] mt-0.5">Currently available: 0.24 days</p>
              </div>
              <div className="rounded-lg bg-slate-900/60 p-2.5 border border-slate-800">
                <span className="text-slate-400 font-medium">Seasonal Cycles (Seasonal Naive, ARIMA):</span>
                <p className="text-slate-200 font-semibold mt-0.5">Requires ≥ 2 Full Cycles (14d/24m)</p>
                <p className="text-amber-400/80 text-[11px] mt-0.5">Currently available: 0 cycles</p>
              </div>
              <div className="rounded-lg bg-slate-900/60 p-2.5 border border-slate-800">
                <span className="text-slate-400 font-medium">Deep Neural Models (BiLSTM):</span>
                <p className="text-slate-200 font-semibold mt-0.5">Strictly Locked (Requires ≥ 1-2 Years)</p>
                <p className="text-rose-400 text-[11px] mt-0.5">Prevent hallucination & circularity</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* KPI Ribbon */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Total Real Incidents</span>
            <Database className="h-4 w-4 text-indigo-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-white">
            {summary?.datasetOverview.totalRealIncidents ?? 66}
          </p>
          <span className="text-[11px] text-emerald-400">100% genuine citizen/field reports</span>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Temporal Observation Span</span>
            <Clock className="h-4 w-4 text-amber-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-amber-400">
            {summary?.datasetOverview.longitudinalSpanHours ?? 5.8} Hours
          </p>
          <span className="text-[11px] text-slate-400">Single afternoon (Nov 7, 2026)</span>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Bengaluru Metro</span>
            <MapPin className="h-4 w-4 text-cyan-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-white">
            {summary?.datasetOverview.bengaluruCount ?? 58}
          </p>
          <span className="text-[11px] text-cyan-400/80">87.9% of real reports</span>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Delhi NCR</span>
            <MapPin className="h-4 w-4 text-orange-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-white">
            {summary?.datasetOverview.delhiCount ?? 8}
          </p>
          <span className="text-[11px] text-orange-400/80">12.1% of real reports</span>
        </div>
      </div>

      {/* Main Chart Section */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-6">
        {/* Controls Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
          {/* Region Tabs */}
          <div className="flex items-center gap-1 rounded-lg bg-slate-800/80 p-1 border border-slate-700/60">
            <button
              onClick={() => setSelectedRegion('ALL')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                selectedRegion === 'ALL'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              All Regions (66)
            </button>
            <button
              onClick={() => setSelectedRegion('BENGALURU_METRO')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                selectedRegion === 'BENGALURU_METRO'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Bengaluru Metro (58)
            </button>
            <button
              onClick={() => setSelectedRegion('DELHI_NCR')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                selectedRegion === 'DELHI_NCR'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Delhi NCR (8)
            </button>
          </div>

          {/* Frequency Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Resolution:</span>
            <div className="flex items-center gap-1 rounded-lg bg-slate-800/80 p-1 border border-slate-700/60">
              <button
                onClick={() => setSelectedFrequency('hourly')}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  selectedFrequency === 'hourly'
                    ? 'bg-slate-700 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Hourly
              </button>
              <button
                onClick={() => setSelectedFrequency('daily')}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  selectedFrequency === 'daily'
                    ? 'bg-slate-700 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Daily
              </button>
              <button
                onClick={() => setSelectedFrequency('weekly')}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  selectedFrequency === 'weekly'
                    ? 'bg-slate-700 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Weekly
              </button>
              <button
                onClick={() => setSelectedFrequency('monthly')}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  selectedFrequency === 'monthly'
                    ? 'bg-slate-700 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Monthly
              </button>
            </div>
          </div>
        </div>

        {/* Chart View */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Historical Incident Volume — {selectedRegion} ({selectedFrequency.toUpperCase()})
              </h3>
              <p className="text-xs text-slate-400">
                {selectedFrequency === 'hourly'
                  ? 'Continuous uniform hourly buckets over the 5.8-hour observation timeline on 2026-11-07.'
                  : `Single observed ${selectedFrequency} period available. Insufficient longitudinal span for multi-period plotting.`}
              </p>
            </div>

            {simulationActive && (
              <button
                onClick={clearSimulation}
                className="text-xs text-amber-400 hover:text-amber-300 flex items-center gap-1"
              >
                <RefreshCw className="h-3 w-3" /> Clear Simulation Overlay
              </button>
            )}
          </div>

          <div className="h-80 w-full pt-4">
            {loading ? (
              <div className="h-full flex items-center justify-center text-slate-400 text-xs">
                <RefreshCw className="h-5 w-5 animate-spin mr-2" />
                Aggregating incident time-series...
              </div>
            ) : buckets.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-500 text-xs">
                No incidents found for the selected geographic region.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={buckets} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="incidentGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="verifiedGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="simGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.5} />
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="periodLabel" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.5rem',
                      fontSize: '12px',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Area
                    type="monotone"
                    dataKey="incidentCount"
                    name="Reported Incidents (Real)"
                    stroke="#6366f1"
                    fillOpacity={1}
                    fill="url(#incidentGrad)"
                    strokeWidth={2}
                  />
                  <Area
                    type="monotone"
                    dataKey="verifiedCount"
                    name="Inspector Verified"
                    stroke="#10b981"
                    fillOpacity={1}
                    fill="url(#verifiedGrad)"
                    strokeWidth={2}
                  />
                  {simulationActive && (
                    <Area
                      type="monotone"
                      dataKey="projectedCount"
                      name="Simulated Projection (Non-Operational)"
                      stroke="#f59e0b"
                      strokeDasharray="5 5"
                      fillOpacity={1}
                      fill="url(#simGrad)"
                      strokeWidth={2}
                    />
                  )}
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Simulation Sandbox Drawer */}
        <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-indigo-400" />
              <span className="text-xs font-semibold text-slate-200">
                Parametric Simulation Sandbox (Audited Dry-Run)
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-400 border border-slate-700">
                Non-Operational
              </span>
            </div>
            <span className="text-[11px] text-slate-500">
              Evaluates baseline forecast formulas on current timeline with explicit simulated flags
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3 pt-1">
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Baseline Algorithm:</span>
              <select
                value={simModel}
                onChange={(e) => setSimModel(e.target.value)}
                className="rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="moving_average">Simple Moving Average (SMA-3)</option>
                <option value="naive">Naive Persistence (y_t)</option>
                <option value="linear_trend">Linear Trend (OLS)</option>
                <option value="seasonal_naive" disabled>Seasonal Naive (Locked - 0 Cycles)</option>
                <option value="lstm" disabled>BiLSTM Neural (Locked - Insufficient Samples)</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Horizon:</span>
              <select
                value={simHorizon}
                onChange={(e) => setSimHorizon(Number(e.target.value))}
                className="rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value={2}>2 periods ahead</option>
                <option value={4}>4 periods ahead</option>
                <option value={6}>6 periods ahead</option>
              </select>
            </div>

            <button
              onClick={runSimulation}
              disabled={simLoading || buckets.length === 0}
              className="px-3 py-1 rounded-md text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition-colors flex items-center gap-1.5 disabled:opacity-50"
            >
              {simLoading ? <RefreshCw className="h-3 w-3 animate-spin" /> : <TrendingUp className="h-3 w-3" />}
              Generate Simulation Curve
            </button>
          </div>
        </div>
      </div>

      {/* Model Catalog & Governance Matrix */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-4">
        <div>
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <Layers className="h-5 w-5 text-indigo-400" />
            Forecasting Model Registry & Minimum Data Requirements
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Architectural catalog of time-series baselines, empirical data thresholds, and operational statuses.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800/60 text-slate-300 font-semibold border-b border-slate-700/60">
              <tr>
                <th className="p-3">Model Name</th>
                <th className="p-3">Architecture</th>
                <th className="p-3">Minimum Data Required</th>
                <th className="p-3">Current Data Available</th>
                <th className="p-3">Mathematical Formulation</th>
                <th className="p-3">Governance Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80 text-slate-300">
              {models.map((m) => {
                const isLocked = m.status === 'LOCKED';
                return (
                  <tr key={m.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="p-3 font-semibold text-white">
                      {m.name}
                      <p className="text-[11px] text-slate-400 font-normal">{m.description}</p>
                    </td>
                    <td className="p-3 text-slate-400">{m.framework}</td>
                    <td className="p-3">
                      <span className="text-slate-200">
                        {m.minimumBucketsRequired?.daily ?? 30} days / {m.minimumBucketsRequired?.monthly ?? 12} months
                      </span>
                    </td>
                    <td className="p-3 text-amber-400 font-medium">0.24 days (5.8 hours)</td>
                    <td className="p-3 font-mono text-[11px] text-slate-400">{m.formulaDescription}</td>
                    <td className="p-3">
                      {isLocked ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/30">
                          <Lock className="h-3 w-3" /> LOCKED: Insufficient Data
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                          <AlertTriangle className="h-3 w-3" /> GATED: Needs Multi-Day
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Synthetic Seed Audit Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Info className="h-5 w-5 text-indigo-400" />
            <h3 className="text-sm font-semibold text-white">
              Database Seed Forecast Audit (Phase 2 Demo Records)
            </h3>
          </div>
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-700">
            Audit Status: FLAGGED_SYNTHETIC
          </span>
        </div>

        <p className="text-xs text-slate-400 leading-relaxed">
          The table <code className="text-indigo-300">incident_forecasts</code> contains{' '}
          <strong>{summary?.syntheticSeedAudit.totalSyntheticForecasts ?? 6} records</strong> generated by{' '}
          <code className="text-slate-300">prisma/seed.ts</code> for <code className="text-slate-300">delhi-central-zone</code>.
          In Stage 6, these records have been permanently flagged with <code className="text-amber-400">is_synthetic = true</code>,
          preventing any circular training or contamination with real incident observations.
        </p>

        <div className="rounded-lg bg-slate-950/70 p-3 text-xs text-slate-400 border border-slate-800 flex items-center justify-between">
          <span>Synthetic demo forecasts excluded from training & evaluation pipelines:</span>
          <span className="font-semibold text-emerald-400 flex items-center gap-1">
            <CheckCircle2 className="h-3.5 w-3.5" /> 6 Records Excluded & Isolated
          </span>
        </div>
      </div>
    </div>
  );
}

'use client';

import React, { useState, useEffect } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  MapPin,
  Search,
  SlidersHorizontal,
  X,
  ShieldCheck,
  RefreshCw,
  Info,
  Check,
  TrendingUp,
  Layers,
  Wrench,
  Calculator,
  Ruler,
  IndianRupee,
} from 'lucide-react';

interface FactorDetail {
  score: number;
  weight: number;
  weightedScore: number;
  provenance?: 'inspector_verified' | 'provisional_heuristic' | 'unverified_missing';
  rationale: string;
  metadata?: Record<string, any>;
}

interface RankedPriorityItem {
  id?: string;
  incidentId: string;
  incidentNumber: string;
  title: string;
  address: string | null;
  region: 'BENGALURU_METRO' | 'DELHI_NCR' | 'OUT_OF_BOUNDS_AMBIGUOUS';
  coordinates: {
    latitude: number;
    longitude: number;
  };
  recommendedPriority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'NEEDS_REVIEW';
  priorityScore: number;
  factorScores: {
    severity: FactorDetail;
    damageExtent: FactorDetail;
    incidentAge: FactorDetail;
    localDensity: FactorDetail;
    safetyHazard: FactorDetail;
    evidenceQuality: FactorDetail;
  };
  contributingEvidence: {
    damageCategory: string;
    verifiedClass?: string | null;
    verifiedSeverity?: string | null;
    provisionalSeverity?: string | null;
    defectCount: number;
    normalizedArea: number;
    incidentAgeHours: number;
    nearbyIncidentsIn500m: number;
    isMajorCorridor: boolean;
    cvConfidence: number;
  };
  missingInformation: {
    hasMissingSeverity: boolean;
    hasMissingImage: boolean;
    hasMissingCoordinates: boolean;
    hasUnverifiedStatus: boolean;
    hasLowConfidenceCV: boolean;
    hasConflictingEvidence: boolean;
    warnings: string[];
  };
  explanation: string;
  scoringRuleVersion: string;
  computedAt: string;
  reviewStatus: 'pending' | 'accepted' | 'overridden' | 'rejected' | 'requires_inspection';
  reviewedPriority?: string | null;
  reviewedBy?: string | null;
  reviewedAt?: string | null;
  overrideReason?: string | null;
}

interface CostItemDetail {
  item: string;
  quantity: number;
  unit: string;
  unitRateINR: number;
  subtotalINR: number;
  rationale: string;
}

interface RepairEstimateData {
  incidentId: string;
  incidentNumber: string;
  region: string;
  estimationStatus: 'ESTIMATED' | 'MEASUREMENT_PENDING' | 'RATE_SOURCE_UNAVAILABLE';
  rateScheduleVersion: string;
  currency: string;
  measurement: {
    measuredAreaSqMeters: number | null;
    depthMeters: number | null;
    measurementSource: string;
    opticalBoundingBoxAreaRatio: number;
    isPhysicalAreaMeasured: boolean;
    measurementCaveat: string;
  };
  costBreakdown: {
    materials: CostItemDetail[];
    labor: CostItemDetail[];
    equipment: CostItemDetail[];
    mobilizationSubtotalINR: number;
    subtotalMaterialsINR: number;
    subtotalLaborINR: number;
    subtotalEquipmentINR: number;
    totalEstimatedCostINR: number;
  } | null;
  totalEstimatedCostINR: number | null;
  durationBreakdown: {
    operationalHours: number;
    setupAndTrafficControlHours: number;
    curingWaitHours: number;
    totalEstimatedHours: number;
    totalEstimatedDays: number;
    crewSize: number;
    productivityRateUsed: string;
    operationalExclusions: string[];
  } | null;
  totalEstimatedDurationHours: number | null;
  totalEstimatedDurationDays: number | null;
  assumptions: string[];
  exclusions: string[];
  governanceNotice: string;
}

interface QueueSummary {
  totalIncidents: number;
  priorityBreakdown: {
    CRITICAL: number;
    HIGH: number;
    MEDIUM: number;
    LOW: number;
    NEEDS_REVIEW: number;
  };
  reviewBreakdown: {
    pending: number;
    accepted: number;
    overridden: number;
    rejected: number;
    requiresInspection: number;
  };
  regionalBreakdown: {
    bengaluru: number;
    delhi: number;
    ambiguous: number;
  };
  supervisedMlReadiness: {
    status: string;
    currentCompletedRepairs: number;
    requiredCompletedRepairs: number;
    rationale: string;
  };
}

export default function MaintenancePriorityPage() {
  const [items, setItems] = useState<RankedPriorityItem[]>([]);
  const [summary, setSummary] = useState<QueueSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [recalculating, setRecalculating] = useState(false);
  const [search, setSearch] = useState('');
  const [priorityFilter, setPriorityFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [regionFilter, setRegionFilter] = useState('all');

  // Selected Incident Modal State
  const [selectedIncident, setSelectedIncident] = useState<RankedPriorityItem | null>(null);
  const [modalTab, setModalTab] = useState<'priority' | 'estimation'>('priority');
  const [reviewAction, setReviewAction] = useState<'accept' | 'override' | 'reject' | 'requires_inspection'>('accept');
  const [overridePriorityChoice, setOverridePriorityChoice] = useState('CRITICAL');
  const [overrideReasonInput, setOverrideReasonInput] = useState('');
  const [submittingReview, setSubmittingReview] = useState(false);
  const [reviewSuccessMessage, setReviewSuccessMessage] = useState<string | null>(null);

  // Repair Estimation State
  const [repairEstimate, setRepairEstimate] = useState<RepairEstimateData | null>(null);
  const [estimateLoading, setEstimateLoading] = useState(false);
  const [inputArea, setInputArea] = useState('2.5');
  const [inputDepth, setInputDepth] = useState('0.05');
  const [inputSource, setInputSource] = useState('field_measurement');
  const [calculatingEstimate, setCalculatingEstimate] = useState(false);

  const fetchQueueData = async () => {
    try {
      setLoading(true);
      const [sumRes, queueRes] = await Promise.all([
        fetch('http://localhost:5000/api/v1/maintenance-priority/summary'),
        fetch('http://localhost:5000/api/v1/maintenance-priority/ranked'),
      ]);

      if (sumRes.ok) {
        const sumJson = await sumRes.json();
        setSummary(sumJson.data);
      }
      if (queueRes.ok) {
        const queueJson = await queueRes.json();
        setItems(queueJson.data.items || []);
      }
    } catch (err) {
      console.error('Error fetching maintenance priority queue:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQueueData();
  }, []);

  const handleRecalculate = async () => {
    setRecalculating(true);
    try {
      const res = await fetch('http://localhost:5000/api/v1/maintenance-priority/recalculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (res.ok) {
        await fetchQueueData();
      }
    } catch (err) {
      console.error('Recalculation error:', err);
    } finally {
      setRecalculating(false);
    }
  };

  const openIncidentModal = async (item: RankedPriorityItem) => {
    setSelectedIncident(item);
    setReviewSuccessMessage(null);
    setOverrideReasonInput(item.overrideReason || '');
    if (item.reviewedPriority) {
      setOverridePriorityChoice(item.reviewedPriority);
    }
    setModalTab('priority');

    // Fetch Repair Estimate
    setEstimateLoading(true);
    try {
      const res = await fetch(`http://localhost:5000/api/v1/repair-estimates/incidents/${item.incidentId}`);
      if (res.ok) {
        const json = await res.json();
        setRepairEstimate(json.data);
        if (json.data?.measurement?.measuredAreaSqMeters) {
          setInputArea(String(json.data.measurement.measuredAreaSqMeters));
        }
        if (json.data?.measurement?.depthMeters) {
          setInputDepth(String(json.data.measurement.depthMeters));
        }
      } else {
        setRepairEstimate(null);
      }
    } catch (err) {
      console.error('Error loading repair estimate:', err);
      setRepairEstimate(null);
    } finally {
      setEstimateLoading(false);
    }
  };

  const handleCalculateEstimate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedIncident) return;
    setCalculatingEstimate(true);
    try {
      const res = await fetch('http://localhost:5000/api/v1/repair-estimates/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          incidentId: selectedIncident.incidentId,
          measuredAreaSqMeters: parseFloat(inputArea) || 0,
          depthMeters: parseFloat(inputDepth) || 0.05,
          measurementSource: inputSource,
        }),
      });
      if (res.ok) {
        const json = await res.json();
        setRepairEstimate(json.data);
      } else {
        alert('Failed to calculate estimate.');
      }
    } catch (err) {
      console.error(err);
      alert('Network error calculating estimate.');
    } finally {
      setCalculatingEstimate(false);
    }
  };

  const handleReviewSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedIncident) return;

    if (reviewAction === 'override' && (!overrideReasonInput || overrideReasonInput.trim().length < 5)) {
      alert('Please provide an override reason (at least 5 characters).');
      return;
    }

    setSubmittingReview(true);
    try {
      const res = await fetch('http://localhost:5000/api/v1/maintenance-priority/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          incidentId: selectedIncident.incidentId,
          action: reviewAction,
          reviewedPriority: reviewAction === 'override' ? overridePriorityChoice : undefined,
          overrideReason: overrideReasonInput || undefined,
        }),
      });

      if (res.ok) {
        const json = await res.json();
        setReviewSuccessMessage(json.data.message || 'Review committed.');
        await fetchQueueData();

        // Refresh selected incident detail
        const detRes = await fetch(`http://localhost:5000/api/v1/maintenance-priority/incidents/${selectedIncident.incidentId}`);
        if (detRes.ok) {
          const detJson = await detRes.json();
          setSelectedIncident(detJson.data);
        }
      } else {
        const errJson = await res.json();
        alert(errJson.error || 'Failed to submit review.');
      }
    } catch (err) {
      console.error(err);
      alert('Network error submitting review.');
    } finally {
      setSubmittingReview(false);
    }
  };

  // Filtered Items
  const filteredItems = items.filter((item) => {
    if (priorityFilter !== 'all' && item.recommendedPriority.toLowerCase() !== priorityFilter.toLowerCase()) {
      return false;
    }
    if (statusFilter !== 'all' && item.reviewStatus.toLowerCase() !== statusFilter.toLowerCase()) {
      return false;
    }
    if (regionFilter !== 'all' && item.region.toLowerCase() !== regionFilter.toLowerCase()) {
      return false;
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      return (
        item.incidentNumber.toLowerCase().includes(q) ||
        item.title.toLowerCase().includes(q) ||
        (item.address && item.address.toLowerCase().includes(q))
      );
    }
    return true;
  });

  const getPriorityBadge = (p: string) => {
    switch (p.toUpperCase()) {
      case 'CRITICAL':
        return 'bg-red-500/15 text-red-400 border-red-500/30';
      case 'HIGH':
        return 'bg-amber-500/15 text-amber-400 border-amber-500/30';
      case 'MEDIUM':
        return 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30';
      case 'LOW':
        return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30';
      case 'NEEDS_REVIEW':
        return 'bg-purple-500/15 text-purple-400 border-purple-500/30';
      default:
        return 'bg-slate-800 text-slate-400 border-slate-700';
    }
  };

  const getReviewBadge = (st: string) => {
    switch (st) {
      case 'accepted':
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
      case 'overridden':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/30';
      case 'rejected':
        return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
      case 'requires_inspection':
        return 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30';
      default:
        return 'bg-slate-800/60 text-slate-400 border-slate-700';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header & Policy Notice */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <SlidersHorizontal className="h-6 w-6 text-emerald-400" />
            Maintenance Intelligence & Estimation
          </h1>
          <p className="mt-1 text-xs text-slate-400">
            Multi-factor priority triage and parametric repair cost & duration estimation.
          </p>
        </div>

        <button
          type="button"
          onClick={handleRecalculate}
          disabled={recalculating}
          className="inline-flex items-center gap-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3.5 py-2 text-xs font-semibold text-slate-200 transition-colors disabled:opacity-50 self-start md:self-auto"
        >
          <RefreshCw className={`h-4 w-4 text-emerald-400 ${recalculating ? 'animate-spin' : ''}`} />
          {recalculating ? 'Recalculating Scores...' : 'Recalculate Priorities'}
        </button>
      </div>

      {/* Mandatory Governance Notice */}
      <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200 flex items-start gap-2.5">
        <Info className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-amber-300">Administrative Governance Notice:</span>{' '}
          Maintenance priority scores and repair estimates are explainable administrative recommendations, not validated predictions or automatic dispatch decisions.
        </div>
      </div>

      {/* Queue Metrics Summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5 space-y-1">
          <p className="text-[11px] font-medium text-slate-400">Total Incidents</p>
          <p className="text-2xl font-bold text-white font-mono">{summary?.totalIncidents ?? items.length}</p>
          <p className="text-[10px] text-slate-500">Bangalore: {summary?.regionalBreakdown.bengaluru ?? 58} | Delhi: {summary?.regionalBreakdown.delhi ?? 8}</p>
        </div>

        <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-3.5 space-y-1">
          <p className="text-[11px] font-medium text-red-300">Critical Urgency</p>
          <p className="text-2xl font-bold text-red-400 font-mono">{summary?.priorityBreakdown.CRITICAL ?? 0}</p>
          <p className="text-[10px] text-red-400/70">Score &ge; 80 or arterial hazard</p>
        </div>

        <div className="rounded-xl border border-amber-500/20 bg-amber-950/20 p-3.5 space-y-1">
          <p className="text-[11px] font-medium text-amber-300">High Priority</p>
          <p className="text-2xl font-bold text-amber-400 font-mono">{summary?.priorityBreakdown.HIGH ?? 0}</p>
          <p className="text-[10px] text-amber-400/70">65 &le; Score &lt; 80</p>
        </div>

        <div className="rounded-xl border border-purple-500/20 bg-purple-950/20 p-3.5 space-y-1">
          <p className="text-[11px] font-medium text-purple-300">Needs Field Review</p>
          <p className="text-2xl font-bold text-purple-400 font-mono">{summary?.priorityBreakdown.NEEDS_REVIEW ?? 0}</p>
          <p className="text-[10px] text-purple-400/70">Incomplete or conflicting data</p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5 space-y-1 col-span-2 md:col-span-1">
          <p className="text-[11px] font-medium text-slate-400">Regression Gate</p>
          <p className="text-xs font-mono font-bold text-red-400">DATA_INSUFFICIENT</p>
          <p className="text-[10px] text-slate-500">Completed Repairs: 0 / 50 req</p>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3 bg-slate-900/40 border border-slate-800/80 p-3 rounded-xl">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search by incident number, title, or road corridor..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 rounded-lg border border-slate-800 bg-slate-950 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          {/* Priority Filter */}
          <select
            value={priorityFilter}
            onChange={(e) => setPriorityFilter(e.target.value)}
            className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-slate-300 focus:outline-none focus:border-emerald-500"
          >
            <option value="all">All Priorities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="needs_review">Needs Review</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>

          {/* Region Filter */}
          <select
            value={regionFilter}
            onChange={(e) => setRegionFilter(e.target.value)}
            className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-slate-300 focus:outline-none focus:border-emerald-500"
          >
            <option value="all">All Regions</option>
            <option value="bengaluru_metro">Bengaluru Metro</option>
            <option value="delhi_ncr">Delhi NCR</option>
          </select>

          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-slate-300 focus:outline-none focus:border-emerald-500"
          >
            <option value="all">All Review States</option>
            <option value="pending">Pending</option>
            <option value="accepted">Accepted</option>
            <option value="overridden">Overridden</option>
            <option value="requires_inspection">Inspection Needed</option>
          </select>
        </div>
      </div>

      {/* Ranked Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 border-b border-slate-800 text-[11px] font-semibold text-slate-400">
              <tr>
                <th className="px-4 py-3">Rank</th>
                <th className="px-4 py-3">Incident & Corridor</th>
                <th className="px-4 py-3">Region</th>
                <th className="px-4 py-3">Recommended Priority</th>
                <th className="px-4 py-3">Priority Score</th>
                <th className="px-4 py-3">Factor Highlights</th>
                <th className="px-4 py-3">Review Status</th>
                <th className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {loading ? (
                <tr>
                  <td colSpan={8} className="px-4 py-12 text-center text-slate-500">
                    <RefreshCw className="h-5 w-5 animate-spin mx-auto mb-2 text-emerald-400" />
                    Loading ranked maintenance queue...
                  </td>
                </tr>
              ) : filteredItems.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                    No incidents match the active filters.
                  </td>
                </tr>
              ) : (
                filteredItems.map((item, idx) => (
                  <tr key={item.incidentId} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-4 py-3 font-mono font-bold text-slate-500">#{idx + 1}</td>
                    <td className="px-4 py-3 max-w-[240px]">
                      <p className="font-mono font-semibold text-white">{item.incidentNumber}</p>
                      <p className="text-[11px] text-slate-400 truncate">{item.title}</p>
                      <p className="text-[10px] text-slate-500 truncate flex items-center gap-1 mt-0.5">
                        <MapPin className="h-2.5 w-2.5 text-slate-500 shrink-0" />
                        {item.address || 'GPS Geocoded Point'}
                      </p>
                    </td>
                    <td className="px-4 py-3">
                      <span className="inline-flex items-center gap-1 text-[10px] font-medium text-slate-300">
                        {item.region === 'BENGALURU_METRO' ? (
                          <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-300 border border-blue-500/20">
                            Bengaluru Metro
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                            Delhi NCR
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex px-2 py-0.5 rounded text-[10px] font-bold border uppercase ${getPriorityBadge(
                          item.recommendedPriority
                        )}`}
                      >
                        {item.recommendedPriority}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="space-y-1 w-24">
                        <div className="flex items-center justify-between text-[11px] font-mono">
                          <span className="font-bold text-white">{item.priorityScore.toFixed(1)}</span>
                          <span className="text-slate-500">/100</span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              item.priorityScore >= 80
                                ? 'bg-red-500'
                                : item.priorityScore >= 65
                                ? 'bg-amber-500'
                                : item.priorityScore >= 45
                                ? 'bg-yellow-500'
                                : 'bg-emerald-500'
                            }`}
                            style={{ width: `${Math.min(100, item.priorityScore)}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-[11px] space-y-0.5 text-slate-400">
                      <div>
                        Sev:{' '}
                        <span
                          className={`font-semibold ${
                            item.factorScores.severity.provenance === 'inspector_verified'
                              ? 'text-emerald-400'
                              : 'text-amber-400'
                          }`}
                        >
                          {item.contributingEvidence.verifiedSeverity || item.contributingEvidence.provisionalSeverity || 'Unverified'}
                        </span>{' '}
                        <span className="text-[9px] text-slate-500">
                          ({item.factorScores.severity.provenance === 'inspector_verified' ? 'Ground Truth' : 'Heuristic'})
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-500">
                        Density: {item.contributingEvidence.nearbyIncidentsIn500m} nearby | Age: {item.factorScores.incidentAge.metadata?.ageDays?.toFixed(0) ?? 0}d
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex px-2 py-0.5 rounded text-[10px] font-semibold border capitalize ${getReviewBadge(
                          item.reviewStatus
                        )}`}
                      >
                        {item.reviewStatus}
                      </span>
                      {item.reviewedPriority && (
                        <span className="block text-[10px] text-blue-400 font-mono mt-0.5">
                          &rarr; {item.reviewedPriority}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        onClick={() => openIncidentModal(item)}
                        className="inline-flex items-center gap-1 rounded bg-slate-800 hover:bg-slate-700 px-2.5 py-1 text-[11px] font-semibold text-slate-200 transition-colors"
                      >
                        <ShieldCheck className="h-3 w-3 text-emerald-400" />
                        Inspect & Review
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail & Multi-Module Modal */}
      {selectedIncident && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 overflow-y-auto">
          <div className="relative w-full max-w-4xl max-h-[92vh] overflow-y-auto rounded-xl border border-slate-800 bg-slate-900 p-6 space-y-5 shadow-2xl my-auto">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-bold text-white">{selectedIncident.title}</h3>
                  <span
                    className={`inline-flex px-2 py-0.5 rounded text-[10px] font-bold border uppercase ${getPriorityBadge(
                      selectedIncident.recommendedPriority
                    )}`}
                  >
                    Recommended: {selectedIncident.recommendedPriority}
                  </span>
                  <span
                    className={`inline-flex px-2 py-0.5 rounded text-[10px] font-semibold border capitalize ${getReviewBadge(
                      selectedIncident.reviewStatus
                    )}`}
                  >
                    Status: {selectedIncident.reviewStatus}
                  </span>
                </div>
                <p className="text-xs text-slate-400 font-mono mt-0.5">
                  {selectedIncident.incidentNumber} &bull; {selectedIncident.address} &bull; Region: {selectedIncident.region}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSelectedIncident(null)}
                className="rounded-lg p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Modal Tabs */}
            <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
              <button
                type="button"
                onClick={() => setModalTab('priority')}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                  modalTab === 'priority'
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'bg-slate-800/60 text-slate-400 hover:text-white'
                }`}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" />
                Stage 3: Priority Decision Support
              </button>
              <button
                type="button"
                onClick={() => setModalTab('estimation')}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                  modalTab === 'estimation'
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'bg-slate-800/60 text-slate-400 hover:text-white'
                }`}
              >
                <Calculator className="h-3.5 w-3.5" />
                Stage 4: Repair Cost & Duration Estimation
              </button>
            </div>

            {/* Success Message Banner */}
            {reviewSuccessMessage && (
              <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs flex items-center gap-2">
                <Check className="h-4 w-4 shrink-0" />
                <span>{reviewSuccessMessage}</span>
              </div>
            )}

            {/* TAB 1: PRIORITY DECISION SUPPORT */}
            {modalTab === 'priority' && (
              <div className="space-y-4">
                {/* Plain-Language Explanation */}
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800 text-xs space-y-1">
                  <p className="font-semibold text-emerald-400 flex items-center gap-1.5">
                    <TrendingUp className="h-3.5 w-3.5" /> Explainable Decision Support Assessment
                  </p>
                  <p className="text-slate-300 leading-relaxed">{selectedIncident.explanation}</p>
                </div>

                {/* Warnings & Missing Info */}
                {selectedIncident.missingInformation.warnings.length > 0 && (
                  <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/25 text-xs text-amber-300 space-y-1">
                    <p className="font-semibold flex items-center gap-1 text-amber-400">
                      <AlertTriangle className="h-3.5 w-3.5" /> Evidence & Data Governance Cautions
                    </p>
                    <ul className="list-disc list-inside space-y-0.5 text-slate-300 text-[11px]">
                      {selectedIncident.missingInformation.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* 6-Factor Breakdown Grid */}
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                    Six-Factor Normalization & Score Breakdown
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                    {/* 1. Severity */}
                    <div className="p-3 rounded-lg border border-slate-800 bg-slate-800/30 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-slate-300">1. Severity (Weight: 30%)</span>
                        <span className="font-mono text-emerald-400 font-bold">
                          {selectedIncident.factorScores.severity.score}/100 ({selectedIncident.factorScores.severity.weightedScore} pts)
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">{selectedIncident.factorScores.severity.rationale}</p>
                      <p className="text-[10px] text-slate-500 italic">
                        Provenance: {selectedIncident.factorScores.severity.provenance === 'inspector_verified' ? 'Field Ground Truth' : 'Provisional Rule Heuristic'}
                      </p>
                    </div>

                    {/* 2. Safety Hazard */}
                    <div className="p-3 rounded-lg border border-slate-800 bg-slate-800/30 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-slate-300">2. Safety Hazard (Weight: 15%)</span>
                        <span className="font-mono text-emerald-400 font-bold">
                          {selectedIncident.factorScores.safetyHazard.score}/100 ({selectedIncident.factorScores.safetyHazard.weightedScore} pts)
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">{selectedIncident.factorScores.safetyHazard.rationale}</p>
                    </div>

                    {/* 3. Damage Extent */}
                    <div className="p-3 rounded-lg border border-slate-800 bg-slate-800/30 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-slate-300">3. Damage Extent (Weight: 20%)</span>
                        <span className="font-mono text-emerald-400 font-bold">
                          {selectedIncident.factorScores.damageExtent.score}/100 ({selectedIncident.factorScores.damageExtent.weightedScore} pts)
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">{selectedIncident.factorScores.damageExtent.rationale}</p>
                    </div>

                    {/* 4. Local Density */}
                    <div className="p-3 rounded-lg border border-slate-800 bg-slate-800/30 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-slate-300">4. Local Density in 500m (Weight: 15%)</span>
                        <span className="font-mono text-emerald-400 font-bold">
                          {selectedIncident.factorScores.localDensity.score}/100 ({selectedIncident.factorScores.localDensity.weightedScore} pts)
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">{selectedIncident.factorScores.localDensity.rationale}</p>
                    </div>

                    {/* 5. Incident Age */}
                    <div className="p-3 rounded-lg border border-slate-800 bg-slate-800/30 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-slate-300">5. Incident Age (Weight: 15%)</span>
                        <span className="font-mono text-emerald-400 font-bold">
                          {selectedIncident.factorScores.incidentAge.score}/100 ({selectedIncident.factorScores.incidentAge.weightedScore} pts)
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">{selectedIncident.factorScores.incidentAge.rationale}</p>
                    </div>

                    {/* 6. Evidence Quality */}
                    <div className="p-3 rounded-lg border border-slate-800 bg-slate-800/30 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-slate-300">6. Evidence Quality (Weight: 5%)</span>
                        <span className="font-mono text-emerald-400 font-bold">
                          {selectedIncident.factorScores.evidenceQuality.score}/100 ({selectedIncident.factorScores.evidenceQuality.weightedScore} pts)
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">{selectedIncident.factorScores.evidenceQuality.rationale}</p>
                    </div>
                  </div>
                </div>

                {/* Admin Review & Override Panel */}
                <form onSubmit={handleReviewSubmit} className="pt-3 border-t border-slate-800 space-y-3 text-xs">
                  <h4 className="font-bold text-white flex items-center gap-1.5">
                    <ShieldCheck className="h-4 w-4 text-emerald-400" /> Administrative Review & Override Action
                  </h4>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <label className="text-slate-300 font-medium">Review Action:</label>
                      <select
                        value={reviewAction}
                        onChange={(e) => setReviewAction(e.target.value as any)}
                        className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-white focus:outline-none focus:border-emerald-500"
                      >
                        <option value="accept">Accept Recommendation (Approve Priority)</option>
                        <option value="override">Override Priority (Administrative Adjustment)</option>
                        <option value="requires_inspection">Require Additional Inspection (Field Triage)</option>
                        <option value="reject">Reject Recommendation (Deprioritize / Dismiss)</option>
                      </select>
                    </div>

                    {reviewAction === 'override' && (
                      <div className="space-y-1">
                        <label className="text-slate-300 font-medium">Adjusted Approved Priority:</label>
                        <select
                          value={overridePriorityChoice}
                          onChange={(e) => setOverridePriorityChoice(e.target.value)}
                          className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-white focus:outline-none focus:border-emerald-500"
                        >
                          <option value="CRITICAL">CRITICAL — Severe hazard / expedited maintenance</option>
                          <option value="HIGH">HIGH — High traffic impact requiring near-term repair</option>
                          <option value="MEDIUM">MEDIUM — Standard scheduled road maintenance</option>
                          <option value="LOW">LOW — Superficial cosmetic / preventive maintenance</option>
                        </select>
                      </div>
                    )}
                  </div>

                  {reviewAction === 'override' && (
                    <div className="space-y-1">
                      <label className="text-slate-300 font-medium">
                        Mandatory Override Reason (Logged to AuditLog):
                      </label>
                      <input
                        type="text"
                        value={overrideReasonInput}
                        onChange={(e) => setOverrideReasonInput(e.target.value)}
                        placeholder="e.g. VIP movement route, bus transitway disruption, or pending utility excavation"
                        className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                        required
                      />
                    </div>
                  )}

                  <p className="text-[11px] text-slate-500 italic">
                    Preservation Guarantee: Overriding a recommendation records an immutable audit log entry with reviewer ID and timestamp. Automatic dispatch of repair crews is strictly prohibited.
                  </p>

                  <div className="flex items-center justify-end gap-3 pt-2">
                    <button
                      type="button"
                      onClick={() => setSelectedIncident(null)}
                      className="rounded-lg bg-slate-800 hover:bg-slate-700 px-4 py-2 font-semibold text-slate-300"
                    >
                      Close
                    </button>
                    <button
                      type="submit"
                      disabled={submittingReview}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 px-4 py-2 font-semibold text-white transition-colors disabled:opacity-50 shadow-md"
                    >
                      <Check className="h-4 w-4" />
                      {submittingReview ? 'Recording Action...' : 'Commit Administrative Decision'}
                    </button>
                  </div>
                </form>
              </div>
            )}

            {/* TAB 2: PARAMETRIC REPAIR ESTIMATION */}
            {modalTab === 'estimation' && (
              <div className="space-y-4 text-xs">
                {estimateLoading ? (
                  <div className="py-12 text-center text-slate-500">
                    <RefreshCw className="h-5 w-5 animate-spin mx-auto mb-2 text-emerald-400" />
                    Loading parametric cost schedule...
                  </div>
                ) : (
                  <>
                    {/* Schedule Header & Governance Notice */}
                    <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800 space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-white flex items-center gap-1.5">
                          <Wrench className="h-4 w-4 text-cyan-400" />
                          Regional Rate Schedule: {repairEstimate?.rateScheduleVersion || 'KPWD-BBMP-PROVISIONAL-2026.1'}
                        </span>
                        <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-300 border border-blue-500/20 font-mono text-[10px]">
                          {repairEstimate?.region || selectedIncident.region}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">
                        {repairEstimate?.governanceNotice ||
                          'Parametric Decision Support Estimate: This is an engineering projection, not an actual expenditure or contractor tender. Actual cost will be recorded upon physical work order completion.'}
                      </p>
                    </div>

                    {/* Status & Measurement Alert */}
                    {repairEstimate?.estimationStatus === 'MEASUREMENT_PENDING' ? (
                      <div className="p-3.5 rounded-lg border border-amber-500/30 bg-amber-500/10 space-y-2">
                        <div className="flex items-center gap-2 text-amber-300 font-semibold">
                          <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
                          <span>Status: MEASUREMENT_PENDING (Physical Dimensions Unmeasured)</span>
                        </div>
                        <p className="text-slate-300 text-[11px] leading-relaxed">
                          2D bounding-box pixel area ratio on camera frame cannot be converted into physical ground area without field measurement or photogrammetric ground calibration. Enter physical measurements below to generate a transparent estimate.
                        </p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {/* Cost Box */}
                        <div className="p-3.5 rounded-lg border border-emerald-500/30 bg-emerald-950/20 space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="font-bold text-slate-300 flex items-center gap-1">
                              <IndianRupee className="h-4 w-4 text-emerald-400" />
                              Estimated Repair Cost
                            </span>
                            <span className="text-lg font-mono font-bold text-emerald-400">
                              ₹ {repairEstimate?.totalEstimatedCostINR?.toLocaleString('en-IN') ?? '0.00'}
                            </span>
                          </div>
                          <div className="space-y-1 text-[11px] text-slate-400 pt-1 border-t border-slate-800">
                            <div className="flex justify-between">
                              <span>Materials:</span>
                              <span className="font-mono text-white">₹ {repairEstimate?.costBreakdown?.subtotalMaterialsINR.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Labor (4-person crew):</span>
                              <span className="font-mono text-white">₹ {repairEstimate?.costBreakdown?.subtotalLaborINR.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Equipment & Compaction:</span>
                              <span className="font-mono text-white">₹ {repairEstimate?.costBreakdown?.subtotalEquipmentINR.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Mobilization & Site Mgt:</span>
                              <span className="font-mono text-white">₹ {repairEstimate?.costBreakdown?.mobilizationSubtotalINR.toFixed(2)}</span>
                            </div>
                          </div>
                        </div>

                        {/* Duration Box */}
                        <div className="p-3.5 rounded-lg border border-cyan-500/30 bg-cyan-950/20 space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="font-bold text-slate-300 flex items-center gap-1">
                              <Clock className="h-4 w-4 text-cyan-400" />
                              Estimated Duration
                            </span>
                            <span className="text-lg font-mono font-bold text-cyan-400">
                              {repairEstimate?.totalEstimatedDurationHours?.toFixed(1) ?? '0.0'} hrs ({repairEstimate?.totalEstimatedDurationDays?.toFixed(1) ?? '0.0'} days)
                            </span>
                          </div>
                          <div className="space-y-1 text-[11px] text-slate-400 pt-1 border-t border-slate-800">
                            <div className="flex justify-between">
                              <span>Active Operations:</span>
                              <span className="font-mono text-white">{repairEstimate?.durationBreakdown?.operationalHours.toFixed(1)} hrs</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Setup & Traffic Control:</span>
                              <span className="font-mono text-white">{repairEstimate?.durationBreakdown?.setupAndTrafficControlHours.toFixed(1)} hrs</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Asphalt Curing Wait:</span>
                              <span className="font-mono text-white">{repairEstimate?.durationBreakdown?.curingWaitHours.toFixed(1)} hrs</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Assumed Crew Size:</span>
                              <span className="font-mono text-white">{repairEstimate?.durationBreakdown?.crewSize} personnel</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Interactive Measurement Input Box */}
                    <form onSubmit={handleCalculateEstimate} className="p-3.5 rounded-lg border border-slate-800 bg-slate-950 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-white flex items-center gap-1.5">
                          <Ruler className="h-4 w-4 text-emerald-400" />
                          Field Measurement Input & Recalculation
                        </span>
                        <span className="text-[10px] text-slate-500">
                          Current Source: {repairEstimate?.measurement.measurementSource || 'unmeasured_missing'}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        <div className="space-y-1">
                          <label className="text-slate-400 text-[11px]">Physical Area (sq. meters):</label>
                          <input
                            type="number"
                            step="0.1"
                            min="0.1"
                            max="500.0"
                            value={inputArea}
                            onChange={(e) => setInputArea(e.target.value)}
                            className="w-full rounded border border-slate-800 bg-slate-900 px-2.5 py-1.5 text-white focus:outline-none focus:border-emerald-500"
                            required
                          />
                        </div>

                        <div className="space-y-1">
                          <label className="text-slate-400 text-[11px]">Cavity Depth (meters):</label>
                          <input
                            type="number"
                            step="0.01"
                            min="0.01"
                            max="0.5"
                            value={inputDepth}
                            onChange={(e) => setInputDepth(e.target.value)}
                            className="w-full rounded border border-slate-800 bg-slate-900 px-2.5 py-1.5 text-white focus:outline-none focus:border-emerald-500"
                            required
                          />
                        </div>

                        <div className="space-y-1">
                          <label className="text-slate-400 text-[11px]">Measurement Source:</label>
                          <select
                            value={inputSource}
                            onChange={(e) => setInputSource(e.target.value)}
                            className="w-full rounded border border-slate-800 bg-slate-900 px-2.5 py-1.5 text-white focus:outline-none focus:border-emerald-500"
                          >
                            <option value="field_measurement">Inspector Field Measurement</option>
                            <option value="calibrated_optical">Calibrated Ground Reference</option>
                          </select>
                        </div>
                      </div>

                      <div className="flex items-center justify-between pt-1">
                        <p className="text-[10px] text-slate-500 italic">
                          Entering measured area immediately computes material volume, labor shifts, and machine hours.
                        </p>
                        <button
                          type="submit"
                          disabled={calculatingEstimate}
                          className="inline-flex items-center gap-1.5 rounded bg-cyan-600 hover:bg-cyan-500 px-3 py-1.5 text-xs font-semibold text-white transition-colors disabled:opacity-50"
                        >
                          <Calculator className="h-3.5 w-3.5" />
                          {calculatingEstimate ? 'Computing...' : 'Calculate Parametric Estimate'}
                        </button>
                      </div>
                    </form>

                    {/* Actuals vs Estimates Comparison Box */}
                    <div className="p-3 rounded-lg border border-slate-800 bg-slate-800/20 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-slate-300">Expenditure Accounting Governance</span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                          ACTUALS SEPARATED
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-[11px]">
                        <div className="p-2 rounded bg-slate-900 border border-slate-800 space-y-0.5">
                          <span className="text-slate-500 block">Parametric Estimate:</span>
                          <span className="text-white font-mono font-bold">
                            {repairEstimate?.totalEstimatedCostINR ? `₹ ${repairEstimate.totalEstimatedCostINR.toLocaleString('en-IN')}` : 'Measurement Pending'}
                          </span>
                        </div>
                        <div className="p-2 rounded bg-slate-900 border border-slate-800 space-y-0.5">
                          <span className="text-slate-500 block">Actual Recorded Expenditure:</span>
                          <span className="text-amber-400 font-mono font-medium">Pending Work Order Completion</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-end pt-2 border-t border-slate-800">
                      <button
                        type="button"
                        onClick={() => setSelectedIncident(null)}
                        className="rounded-lg bg-slate-800 hover:bg-slate-700 px-4 py-2 font-semibold text-slate-300"
                      >
                        Close
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

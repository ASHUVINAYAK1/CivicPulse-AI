'use client';

import React, { useState, useEffect } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  MapPin,
  Plus,
  Search,
  Upload,
  Eye,
  Layers,
  Sparkles,
  ShieldCheck,
  X,
  Copy,
  GitMerge,
  Ban,
  Check,
  ArrowRightLeft,
  SlidersHorizontal,
} from 'lucide-react';

interface IncidentItem {
  id: string;
  incidentNumber: string;
  title: string;
  description: string;
  damageType: string;
  severity: string;
  status: string;
  address: string;
  latitude: number | string;
  longitude: number | string;
  reportedAt: string;
  verifiedAt: string | null;
  canonicalIncidentId?: string | null;
  duplicateStatus?: string;
  duplicateConfidence?: number | null;
  canonicalIncident?: {
    id: string;
    incidentNumber: string;
    title: string;
    status: string;
  } | null;
  duplicateReports?: Array<{
    id: string;
    incidentNumber: string;
    title: string;
    duplicateStatus: string;
  }>;
  images: Array<{
    id: string;
    storageKey: string;
    originalFilename: string;
    url?: string;
    dHash?: string | null;
    perceptualHash?: string | null;
  }>;
  predictions: Array<{
    id: string;
    predictedClass: string;
    confidence: number;
    processingTimeMs: number;
    predictionPayload: any;
    modelRegistry?: {
      modelName: string;
      version: string;
    };
  }>;
  verifications: Array<{
    id: string;
    verifiedClass: string;
    verifiedSeverity: string;
    remarks: string;
  }>;
}

interface CandidateResult {
  candidateIncident: {
    id: string;
    incidentNumber: string;
    title: string;
    description: string;
    damageType: string;
    severity: string | null;
    status: string;
    latitude: number;
    longitude: number;
    address: string | null;
    reportedAt: string;
    distanceMeters: number;
    images: {
      id: string;
      url: string;
      originalFilename: string;
      dHash: string | null;
      perceptualHash: string | null;
    }[];
  };
  scoring: {
    confidenceScore: number;
    verdict: 'high_probability_duplicate' | 'potential_duplicate' | 'unlikely_duplicate';
    isSuspectedDuplicate: boolean;
    components: {
      imageScore: number;
      geoScore: number;
      timeScore: number;
      typeScore: number;
      metadataBonus: number;
    };
    reasons: string[];
  };
}

export default function IncidentsPage() {
  const [incidents, setIncidents] = useState<IncidentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filterTab, setFilterTab] = useState<'all' | 'suspected' | 'confirmed'>('all');
  const [selectedIncident, setSelectedIncident] = useState<IncidentItem | null>(null);
  const [severityProfile, setSeverityProfile] = useState<any>(null);
  const [severityLoading, setSeverityLoading] = useState(false);
  const [verifySeverityChoice, setVerifySeverityChoice] = useState('critical');
  const [verifyRemarksInput, setVerifyRemarksInput] = useState('');

  // Duplicate Review Modal State
  const [duplicateReviewIncident, setDuplicateReviewIncident] = useState<IncidentItem | null>(null);
  const [candidates, setCandidates] = useState<CandidateResult[]>([]);
  const [loadingCandidates, setLoadingCandidates] = useState(false);
  const [selectedCandidate, setSelectedCandidate] = useState<CandidateResult | null>(null);
  const [inspectorRemarks, setInspectorRemarks] = useState('');
  const [actionLoading, setActionLoading] = useState(false);

  // Upload modal state
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [roadName, setRoadName] = useState('Hosur Road Silk Board Junction');
  const [address, setAddress] = useState('Bengaluru South District');
  const [verifying, setVerifying] = useState(false);

  // Fetch incidents
  const fetchIncidents = async () => {
    try {
      setLoading(true);
      const res = await fetch('http://localhost:5000/api/v1/incidents');
      if (res.ok) {
        const json = await res.json();
        setIncidents(json.data || []);
      }
    } catch {
      setIncidents([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchIncidents();
  }, []);

  // Open Duplicate Review for an incident
  const openDuplicateReview = async (incident: IncidentItem) => {
    setDuplicateReviewIncident(incident);
    setLoadingCandidates(true);
    setInspectorRemarks('');
    setSelectedCandidate(null);

    try {
      const res = await fetch(`http://localhost:5000/api/v1/incidents/${incident.id}/duplicate-candidates?radius=25&days=30`);
      if (res.ok) {
        const json = await res.json();
        const foundCandidates = json.data?.candidates || [];
        setCandidates(foundCandidates);
        if (foundCandidates.length > 0) {
          setSelectedCandidate(foundCandidates[0]);
        }
      } else {
        setCandidates([]);
      }
    } catch (err) {
      console.error('Error fetching duplicate candidates:', err);
      setCandidates([]);
    } finally {
      setLoadingCandidates(false);
    }
  };

  // Confirm Duplicate Action
  const handleConfirmDuplicate = async (incidentId: string, canonicalId: string) => {
    setActionLoading(true);
    try {
      const res = await fetch(`http://localhost:5000/api/v1/incidents/${incidentId}/duplicates/confirm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          canonicalIncidentId: canonicalId,
          remarks: inspectorRemarks || 'Inspector verified physical duplicate defect on field triage.',
        }),
      });

      if (res.ok) {
        setDuplicateReviewIncident(null);
        await fetchIncidents();
      } else {
        alert('Failed to confirm duplicate');
      }
    } catch (err) {
      console.error(err);
      alert('Network error confirming duplicate');
    } finally {
      setActionLoading(false);
    }
  };

  // Reject Duplicate Action
  const handleRejectDuplicate = async (incidentId: string, candidateId: string) => {
    setActionLoading(true);
    try {
      const res = await fetch(`http://localhost:5000/api/v1/incidents/${incidentId}/duplicates/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidateIncidentId: candidateId,
          remarks: inspectorRemarks || 'Inspector verified distinct physical defect upon survey.',
        }),
      });

      if (res.ok) {
        setDuplicateReviewIncident(null);
        await fetchIncidents();
      } else {
        alert('Failed to reject duplicate suggestion');
      }
    } catch (err) {
      console.error(err);
      alert('Network error rejecting duplicate');
    } finally {
      setActionLoading(false);
    }
  };

  // Handle Image Upload & AI Inference
  const handleReportSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('image', uploadFile);
      formData.append('roadName', roadName);
      formData.append('address', address);

      const res = await fetch('http://localhost:5000/api/v1/incidents/report', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        setShowUploadModal(false);
        setUploadFile(null);
        await fetchIncidents();
      } else {
        alert('Failed to report incident');
      }
    } catch (err) {
      console.error(err);
      alert('Network error connecting to API');
    } finally {
      setUploading(false);
    }
  };

  // Open Incident Details & Fetch 4-Way Severity Profile
  const openIncidentDetails = async (inc: IncidentItem) => {
    setSelectedIncident(inc);
    setSeverityLoading(true);
    setVerifySeverityChoice(inc.severity || 'critical');
    setVerifyRemarksInput('');
    try {
      const res = await fetch(`http://localhost:5000/api/v1/severity/incidents/${inc.id}`);
      if (res.ok) {
        const json = await res.json();
        setSeverityProfile(json.data);
      } else {
        setSeverityProfile(null);
      }
    } catch (err) {
      console.warn('Could not load severity profile:', err);
      setSeverityProfile(null);
    } finally {
      setSeverityLoading(false);
    }
  };

  // Handle Human Inspector Verification
  const handleVerify = async (incidentId: string) => {
    setVerifying(true);
    try {
      const res = await fetch(`http://localhost:5000/api/v1/incidents/${incidentId}/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          verifiedClass: selectedIncident?.damageType || 'pothole',
          verifiedSeverity: verifySeverityChoice,
          remarks: verifyRemarksInput || 'Verified by Field Inspector on dashboard.',
        }),
      });

      if (res.ok) {
        await fetchIncidents();
        // Refresh severity profile
        try {
          const profRes = await fetch(`http://localhost:5000/api/v1/severity/incidents/${incidentId}`);
          if (profRes.ok) {
            const profJson = await profRes.json();
            setSeverityProfile(profJson.data);
          }
        } catch (e) {}

        if (selectedIncident) {
          setSelectedIncident({ ...selectedIncident, status: 'verified', severity: verifySeverityChoice });
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setVerifying(false);
    }
  };

  const suspectedCount = incidents.filter((i) => i.duplicateStatus === 'suspected').length;
  const confirmedCount = incidents.filter((i) => i.duplicateStatus === 'confirmed_duplicate').length;

  const filteredIncidents = incidents
    .filter((inc) => {
      if (filterTab === 'suspected') return inc.duplicateStatus === 'suspected';
      if (filterTab === 'confirmed') return inc.duplicateStatus === 'confirmed_duplicate';
      return true;
    })
    .filter(
      (inc) =>
        inc.title.toLowerCase().includes(search.toLowerCase()) ||
        inc.damageType.toLowerCase().includes(search.toLowerCase()) ||
        inc.address.toLowerCase().includes(search.toLowerCase()) ||
        inc.incidentNumber.toLowerCase().includes(search.toLowerCase())
    );

  const getSeverityBadge = (sev: string) => {
    switch (sev?.toLowerCase()) {
      case 'critical':
        return 'bg-red-500/20 text-red-300 border-red-500/40';
      case 'high':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'medium':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/40';
      default:
        return 'bg-slate-700/50 text-slate-300 border-slate-600';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <AlertTriangle className="h-6 w-6 text-amber-400" />
            Road Distress & AI Inspection Queue
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Automated computer vision triage, perceptual deduplication, and municipal field verification.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShowUploadModal(true)}
          className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white hover:bg-emerald-500 transition-colors shadow-sm"
        >
          <Plus className="h-4 w-4" />
          Report Road Damage
        </button>
      </div>

      {/* Tabs & Search Filter */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-1.5 p-1 rounded-lg bg-slate-900 border border-slate-800 text-xs font-medium">
          <button
            type="button"
            onClick={() => setFilterTab('all')}
            className={`px-3 py-1.5 rounded-md transition-colors ${
              filterTab === 'all' ? 'bg-slate-800 text-white font-semibold' : 'text-slate-400 hover:text-white'
            }`}
          >
            All Reports ({incidents.length})
          </button>
          <button
            type="button"
            onClick={() => setFilterTab('suspected')}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors ${
              filterTab === 'suspected'
                ? 'bg-amber-500/20 text-amber-300 font-semibold border border-amber-500/40'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <Copy className="h-3 w-3" />
            Suspected Duplicates ({suspectedCount})
          </button>
          <button
            type="button"
            onClick={() => setFilterTab('confirmed')}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors ${
              filterTab === 'confirmed'
                ? 'bg-purple-500/20 text-purple-300 font-semibold border border-purple-500/40'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <GitMerge className="h-3 w-3" />
            Confirmed Duplicates ({confirmedCount})
          </button>
        </div>

        <div className="relative w-full sm:w-80">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search by ref, roadway corridor, or defect..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-lg border border-slate-800 bg-slate-900/60 pl-9 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
          />
        </div>
      </div>

      {/* Incidents Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="px-4 py-3 font-semibold">Incident Ref</th>
                <th className="px-4 py-3 font-semibold">Distress Category</th>
                <th className="px-4 py-3 font-semibold">Severity</th>
                <th className="px-4 py-3 font-semibold">AI Confidence</th>
                <th className="px-4 py-3 font-semibold">Deduplication Status</th>
                <th className="px-4 py-3 font-semibold">Location</th>
                <th className="px-4 py-3 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {loading ? (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-slate-500">
                    Loading incident records...
                  </td>
                </tr>
              ) : filteredIncidents.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-slate-500">
                    No matching incidents found. Click &quot;Report Road Damage&quot; to submit a new survey.
                  </td>
                </tr>
              ) : (
                filteredIncidents.map((inc) => {
                  const pred = inc.predictions?.[0];
                  return (
                    <tr key={inc.id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-4 py-3 font-mono font-medium text-white">
                        {inc.incidentNumber}
                      </td>
                      <td className="px-4 py-3">
                        <span className="font-semibold text-slate-200 capitalize">
                          {inc.damageType.replace('_', ' ')}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`inline-flex px-2 py-0.5 rounded text-[10px] font-bold border uppercase ${getSeverityBadge(
                            inc.severity
                          )}`}
                        >
                          {inc.severity || 'UNKNOWN'}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        {pred ? (
                          <div className="flex items-center gap-1.5 font-mono text-emerald-400">
                            <Sparkles className="h-3 w-3" />
                            {(pred.confidence * 100).toFixed(1)}%
                          </div>
                        ) : (
                          <span className="text-slate-500">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        {inc.duplicateStatus === 'suspected' ? (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                            <Copy className="h-3 w-3" />
                            Suspected ({Math.round((inc.duplicateConfidence || 0.8) * 100)}%)
                          </span>
                        ) : inc.duplicateStatus === 'confirmed_duplicate' ? (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/40">
                            <GitMerge className="h-3 w-3" />
                            Duplicate ({inc.canonicalIncident?.incidentNumber || 'Linked'})
                          </span>
                        ) : inc.duplicateStatus === 'rejected' ? (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium bg-slate-800 text-slate-400 border border-slate-700">
                            <Check className="h-3 w-3 text-slate-400" />
                            Distinct
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[11px] text-slate-400">
                            Unique Report
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-slate-400 truncate max-w-[180px]">
                        {inc.address}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => openIncidentDetails(inc)}
                            className="inline-flex items-center gap-1 rounded bg-slate-800 hover:bg-slate-700 px-2 py-1 text-[11px] font-medium text-slate-200 transition-colors"
                          >
                            <Eye className="h-3 w-3" /> Inspect
                          </button>
                          <button
                            type="button"
                            onClick={() => openDuplicateReview(inc)}
                            className={`inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
                              inc.duplicateStatus === 'suspected'
                                ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-sm'
                                : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                            }`}
                          >
                            <ArrowRightLeft className="h-3 w-3" />
                            {inc.duplicateStatus === 'suspected' ? 'Review Dup' : 'Check Dup'}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ============================================================== */}
      {/* INSPECTOR DUPLICATE REVIEW MODAL                                */}
      {/* ============================================================== */}
      {duplicateReviewIncident && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-xl border border-slate-800 bg-slate-900 p-6 space-y-6 shadow-2xl">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <ArrowRightLeft className="h-5 w-5 text-amber-400" />
                  Intelligent Incident Deduplication Review
                </h3>
                <p className="text-xs text-slate-400">
                  Target Incident: <span className="font-mono text-white font-semibold">{duplicateReviewIncident.incidentNumber}</span> —{' '}
                  {duplicateReviewIncident.title}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setDuplicateReviewIncident(null)}
                className="rounded-lg p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {loadingCandidates ? (
              <div className="py-12 text-center text-slate-400 space-y-2">
                <SlidersHorizontal className="h-8 w-8 mx-auto animate-pulse text-amber-400" />
                <p className="font-semibold text-white">Running PostGIS & Perceptual Hashing Analysis...</p>
                <p className="text-xs text-slate-500">Querying ST_DWithin 25m spatial radius and computing Hamming distances</p>
              </div>
            ) : candidates.length === 0 ? (
              <div className="py-12 text-center text-slate-400 space-y-2 rounded-lg border border-slate-800 bg-slate-950 p-6">
                <CheckCircle2 className="h-10 w-10 mx-auto text-emerald-400" />
                <p className="font-bold text-white text-sm">No Duplicate Candidates Detected</p>
                <p className="text-xs text-slate-500 max-w-md mx-auto">
                  PostGIS ST_DWithin found zero nearby reports within a 25-meter spatial radius and 30-day time window. This report appears to be a unique physical road distress.
                </p>
              </div>
            ) : (
              <div className="space-y-6 text-xs">
                {/* Candidate Selection Tabs */}
                {candidates.length > 1 && (
                  <div className="flex items-center gap-2 overflow-x-auto pb-1">
                    <span className="text-slate-400 font-semibold">Suspected Candidates:</span>
                    {candidates.map((cand, idx) => (
                      <button
                        key={cand.candidateIncident.id}
                        type="button"
                        onClick={() => setSelectedCandidate(cand)}
                        className={`px-3 py-1 rounded-md font-mono text-xs transition-colors ${
                          selectedCandidate?.candidateIncident.id === cand.candidateIncident.id
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 font-bold'
                            : 'bg-slate-800 text-slate-400 hover:text-white'
                        }`}
                      >
                        #{idx + 1} {cand.candidateIncident.incidentNumber} ({Math.round(cand.scoring.confidenceScore * 100)}%)
                      </button>
                    ))}
                  </div>
                )}

                {selectedCandidate && (
                  <>
                    {/* Confidence & Explainability Banner */}
                    <div
                      className={`p-4 rounded-xl border ${
                        selectedCandidate.scoring.confidenceScore >= 0.70
                          ? 'bg-amber-500/10 border-amber-500/30'
                          : 'bg-blue-500/10 border-blue-500/30'
                      }`}
                    >
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                        <div>
                          <span
                            className={`inline-flex px-2.5 py-0.5 rounded text-[11px] font-bold uppercase tracking-wider ${
                              selectedCandidate.scoring.confidenceScore >= 0.70
                                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                                : 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                            }`}
                          >
                            {selectedCandidate.scoring.verdict.replace(/_/g, ' ')}
                          </span>
                          <h4 className="text-sm font-bold text-white mt-1">
                            Duplicate Confidence Score: {(selectedCandidate.scoring.confidenceScore * 100).toFixed(1)}%
                          </h4>
                        </div>
                        <div className="flex items-center gap-4 text-slate-300 font-mono">
                          <div>
                            <span className="text-slate-500 block text-[10px]">GPS DISTANCE</span>
                            <span className="font-bold text-emerald-400">
                              {selectedCandidate.candidateIncident.distanceMeters}m
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500 block text-[10px]">VISUAL MATCH</span>
                            <span className="font-bold text-cyan-400">
                              {(selectedCandidate.scoring.components.imageScore * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Explainability Breakdown */}
                      <div className="mt-3 pt-3 border-t border-slate-800 space-y-1">
                        <p className="font-semibold text-slate-300">Algorithmic Evidence & Explainability:</p>
                        <ul className="list-disc list-inside space-y-0.5 text-slate-400">
                          {selectedCandidate.scoring.reasons.map((r, i) => (
                            <li key={i}>{r}</li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    {/* Side-by-Side Photographic Comparison */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Target Incident Box */}
                      <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-3">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                          <span className="font-semibold text-slate-300">Report Under Review (New)</span>
                          <span className="font-mono text-emerald-400 font-bold">{duplicateReviewIncident.incidentNumber}</span>
                        </div>
                        <div className="relative aspect-video rounded-lg border border-slate-800 bg-slate-900 overflow-hidden flex items-center justify-center">
                          {duplicateReviewIncident.images?.[0]?.url ? (
                            <img
                              src={duplicateReviewIncident.images[0].url}
                              alt="Target Incident"
                              className="h-full w-full object-cover"
                            />
                          ) : (
                            <div className="text-slate-500 text-center p-4">
                              <Layers className="h-6 w-6 mx-auto mb-1 text-slate-600" />
                              <p>Image: {duplicateReviewIncident.images?.[0]?.originalFilename || 'Not Loaded'}</p>
                            </div>
                          )}
                        </div>
                        <div className="space-y-1 text-slate-400">
                          <p><span className="text-slate-500">Distress Category:</span> <span className="font-semibold text-white capitalize">{duplicateReviewIncident.damageType}</span></p>
                          <p><span className="text-slate-500">Location:</span> {duplicateReviewIncident.address}</p>
                          <p><span className="text-slate-500">Reported:</span> {new Date(duplicateReviewIncident.reportedAt).toLocaleString()}</p>
                          <p className="font-mono text-[11px]"><span className="text-slate-500">dHash:</span> {duplicateReviewIncident.images?.[0]?.dHash || 'N/A'}</p>
                        </div>
                      </div>

                      {/* Candidate Incident Box */}
                      <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-3">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                          <span className="font-semibold text-amber-300">Candidate Canonical Report (Existing)</span>
                          <span className="font-mono text-amber-400 font-bold">{selectedCandidate.candidateIncident.incidentNumber}</span>
                        </div>
                        <div className="relative aspect-video rounded-lg border border-slate-800 bg-slate-900 overflow-hidden flex items-center justify-center">
                          {selectedCandidate.candidateIncident.images?.[0]?.url ? (
                            <img
                              src={selectedCandidate.candidateIncident.images[0].url}
                              alt="Candidate Duplicate"
                              className="h-full w-full object-cover"
                            />
                          ) : (
                            <div className="text-slate-500 text-center p-4">
                              <Layers className="h-6 w-6 mx-auto mb-1 text-slate-600" />
                              <p>Image: {selectedCandidate.candidateIncident.images?.[0]?.originalFilename || 'Not Loaded'}</p>
                            </div>
                          )}
                        </div>
                        <div className="space-y-1 text-slate-400">
                          <p><span className="text-slate-500">Distress Category:</span> <span className="font-semibold text-white capitalize">{selectedCandidate.candidateIncident.damageType}</span></p>
                          <p><span className="text-slate-500">Location:</span> {selectedCandidate.candidateIncident.address}</p>
                          <p><span className="text-slate-500">Reported:</span> {new Date(selectedCandidate.candidateIncident.reportedAt).toLocaleString()}</p>
                          <p className="font-mono text-[11px]"><span className="text-slate-500">dHash:</span> {selectedCandidate.candidateIncident.images?.[0]?.dHash || 'N/A'}</p>
                        </div>
                      </div>
                    </div>

                    {/* Inspector Remarks & Decision Actions */}
                    <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-3">
                      <label className="font-semibold text-slate-300 block">
                        Inspector Verification Remarks & Audit Note
                      </label>
                      <textarea
                        rows={2}
                        value={inspectorRemarks}
                        onChange={(e) => setInspectorRemarks(e.target.value)}
                        placeholder="e.g. Field inspection confirms identical pothole crater on northbound lane; linking citizen submission to canonical ticket."
                        className="w-full rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-amber-500"
                      />

                      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2 border-t border-slate-800/80">
                        <p className="text-[11px] text-slate-500">
                          Preservation Guarantee: Citizen submissions are never deleted. Confirming creates an immutable AuditLog and canonical linkage.
                        </p>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            disabled={actionLoading}
                            onClick={() =>
                              handleRejectDuplicate(duplicateReviewIncident.id, selectedCandidate.candidateIncident.id)
                            }
                            className="inline-flex items-center gap-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 px-3 py-2 text-xs font-semibold text-slate-200 transition-colors disabled:opacity-50"
                          >
                            <Ban className="h-4 w-4 text-red-400" />
                            Reject Suggestion (Distinct Defect)
                          </button>
                          <button
                            type="button"
                            disabled={actionLoading}
                            onClick={() =>
                              handleConfirmDuplicate(duplicateReviewIncident.id, selectedCandidate.candidateIncident.id)
                            }
                            className="inline-flex items-center gap-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 px-4 py-2 text-xs font-semibold text-white transition-colors disabled:opacity-50 shadow-sm"
                          >
                            <GitMerge className="h-4 w-4" />
                            {actionLoading ? 'Linking...' : 'Confirm as Duplicate & Link'}
                          </button>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Incident Detail Modal with 4-Tier Severity Assessment */}
      {selectedIncident && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 overflow-y-auto">
          <div className="relative w-full max-w-4xl max-h-[92vh] overflow-y-auto rounded-xl border border-slate-800 bg-slate-900 p-6 space-y-6 shadow-2xl my-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-bold text-white">{selectedIncident.title}</h3>
                  <span
                    className={`inline-flex px-2 py-0.5 rounded text-[10px] font-bold border uppercase ${getSeverityBadge(
                      selectedIncident.severity
                    )}`}
                  >
                    {selectedIncident.severity || 'UNKNOWN'}
                  </span>
                  <span
                    className={`inline-flex px-2 py-0.5 rounded text-[10px] font-bold border capitalize ${
                      selectedIncident.status === 'verified'
                        ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                        : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                    }`}
                  >
                    {selectedIncident.status}
                  </span>
                </div>
                <p className="text-xs text-slate-400 font-mono mt-0.5">{selectedIncident.incidentNumber}</p>
              </div>
              <button
                type="button"
                onClick={() => {
                  setSelectedIncident(null);
                  setSeverityProfile(null);
                }}
                className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Top Grid: Photo and Geographic Details */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              {/* Photo Box */}
              <div className="space-y-2">
                <p className="font-semibold text-slate-400">Road Distress Imagery</p>
                <div className="relative aspect-video rounded-lg border border-slate-800 bg-slate-950 overflow-hidden flex items-center justify-center">
                  {selectedIncident.images?.[0]?.url ? (
                    <img
                      src={selectedIncident.images[0].url}
                      alt="Road Damage"
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="text-center text-slate-500 p-4">
                      <Layers className="h-8 w-8 mx-auto mb-1 text-slate-600" />
                      <p>Image Key: {selectedIncident.images?.[0]?.storageKey || 'MinIO Reference'}</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Geographic and Metadata Box */}
              <div className="space-y-3">
                <p className="font-semibold text-slate-400">Spatial & Metadata Attributes</p>
                <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Distress Category:</span>
                    <span className="font-bold text-white uppercase">{selectedIncident.damageType}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">GPS Coordinates:</span>
                    <span className="font-mono text-cyan-400">
                      {typeof selectedIncident.latitude === 'number'
                        ? selectedIncident.latitude.toFixed(6)
                        : selectedIncident.latitude}
                      ,{' '}
                      {typeof selectedIncident.longitude === 'number'
                        ? selectedIncident.longitude.toFixed(6)
                        : selectedIncident.longitude}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Reported Timestamp:</span>
                    <span className="text-slate-300 font-mono">
                      {new Date(selectedIncident.reportedAt).toLocaleString()}
                    </span>
                  </div>
                  <div className="pt-1 border-t border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Corridor / Address:</span>
                    <span className="text-slate-300">{selectedIncident.address}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Phase 3 Stage 2: 4-Tier Explainable Severity Architecture */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-bold text-white flex items-center gap-2">
                  <SlidersHorizontal className="h-4 w-4 text-emerald-400" />
                  Phase 3 Stage 2: Four-Tier Severity Assessment
                </h4>
                {severityLoading && (
                  <span className="text-xs text-amber-400 animate-pulse font-mono">
                    Evaluating feature vectors...
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {/* TIER 1: CV Observation */}
                <div className="rounded-lg border border-slate-800 bg-slate-800/30 p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-300 flex items-center gap-1.5">
                      <Layers className="h-3.5 w-3.5 text-cyan-400" />
                      1. CV Observation
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-400 border border-cyan-800/50">
                      {severityProfile?.sourceObservation?.status || 'DERIVED PROXY'}
                    </span>
                  </div>
                  <div className="space-y-1.5 text-slate-400">
                    <div className="flex justify-between">
                      <span>Detected Defects:</span>
                      <span className="font-mono text-white">
                        {severityProfile?.sourceObservation?.detectionCount ??
                          selectedIncident.predictions?.length ??
                          0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Predominant Class:</span>
                      <span className="font-mono text-cyan-300 uppercase">
                        {severityProfile?.sourceObservation?.predominantClass ||
                          selectedIncident.predictions?.[0]?.predictedClass ||
                          'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Peak Bounding-Box Ratio:</span>
                      <span className="font-mono text-white">
                        {severityProfile?.sourceObservation?.maxNormalizedArea !== undefined
                          ? `${(severityProfile.sourceObservation.maxNormalizedArea * 100).toFixed(2)}%`
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Detector Confidence:</span>
                      <span className="font-mono text-emerald-400">
                        {severityProfile?.sourceObservation?.highestConfidence !== undefined
                          ? `${(severityProfile.sourceObservation.highestConfidence * 100).toFixed(1)}%`
                          : selectedIncident.predictions?.[0]
                          ? `${(selectedIncident.predictions[0].confidence * 100).toFixed(1)}%`
                          : 'N/A'}
                      </span>
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-500 pt-1 border-t border-slate-800/80 italic">
                    Bounding-box pixel area is an uncalibrated optical proxy and does not represent physical square meters.
                  </p>
                </div>

                {/* TIER 2: Provisional Engineering Assessment */}
                <div className="rounded-lg border border-slate-800 bg-slate-800/30 p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-300 flex items-center gap-1.5">
                      <Sparkles className="h-3.5 w-3.5 text-amber-400" />
                      2. Provisional Engineering Rule
                    </span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded border uppercase ${
                        severityProfile?.provisionalEngineeringAssessment?.provisionalSeverity === 'CRITICAL'
                          ? 'bg-red-500/20 text-red-300 border-red-500/40'
                          : severityProfile?.provisionalEngineeringAssessment?.provisionalSeverity === 'HIGH'
                          ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                          : severityProfile?.provisionalEngineeringAssessment?.provisionalSeverity === 'MODERATE'
                          ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40'
                          : severityProfile?.provisionalEngineeringAssessment?.provisionalSeverity === 'LOW'
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : 'bg-slate-800 text-slate-400 border-slate-700'
                      }`}
                    >
                      {severityProfile?.provisionalEngineeringAssessment?.provisionalSeverity || 'EVALUATING'}
                    </span>
                  </div>
                  <div className="space-y-1.5 text-slate-400">
                    <div className="flex justify-between">
                      <span>Primary Rule:</span>
                      <span className="font-mono text-amber-300">
                        {severityProfile?.provisionalEngineeringAssessment?.ruleId || 'N/A'}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-300 bg-slate-900/60 p-2 rounded border border-slate-800/60">
                      {severityProfile?.provisionalEngineeringAssessment?.rationale ||
                        'Awaiting evaluation of geometric defects and defect density.'}
                    </p>
                    {severityProfile?.provisionalEngineeringAssessment?.contributingRules?.length > 1 && (
                      <div className="text-[10px] text-slate-400">
                        Contributing: {severityProfile.provisionalEngineeringAssessment.contributingRules.join(', ')}
                      </div>
                    )}
                  </div>
                  <p className="text-[10px] text-slate-500 pt-1 border-t border-slate-800/80 italic">
                    Engineering heuristic only. Not official IRC compliance or verified ground truth.
                  </p>
                </div>

                {/* TIER 3: Inspector Verified Ground Truth */}
                <div className="rounded-lg border border-slate-800 bg-slate-800/30 p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-300 flex items-center gap-1.5">
                      <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                      3. Inspector Ground Truth
                    </span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded border ${
                        severityProfile?.inspectorGroundTruth?.isVerified || selectedIncident.status === 'verified'
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : 'bg-slate-800 text-slate-400 border-slate-700'
                      }`}
                    >
                      {severityProfile?.inspectorGroundTruth?.isVerified || selectedIncident.status === 'verified'
                        ? 'VERIFIED GROUND TRUTH'
                        : 'PENDING VERIFICATION'}
                    </span>
                  </div>

                  {severityProfile?.inspectorGroundTruth?.isVerified || selectedIncident.status === 'verified' ? (
                    <div className="space-y-1.5 text-slate-400">
                      <div className="flex justify-between">
                        <span>Verified Severity:</span>
                        <span className="font-bold text-emerald-400 uppercase">
                          {severityProfile?.inspectorGroundTruth?.verifiedSeverity || selectedIncident.severity}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Inspector Verification:</span>
                        <span className="text-slate-300">
                          {severityProfile?.inspectorGroundTruth?.verifiedAt
                            ? new Date(severityProfile.inspectorGroundTruth.verifiedAt).toLocaleString()
                            : 'Field Logged'}
                        </span>
                      </div>
                      <div className="pt-1">
                        <span className="text-slate-500 block mb-0.5">Inspector Remarks:</span>
                        <p className="text-[11px] text-slate-300 bg-slate-900/60 p-2 rounded border border-slate-800/60">
                          {severityProfile?.inspectorGroundTruth?.remarks ||
                            'Verified physical pavement defect and severity rating.'}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-2 pt-1">
                      <div className="space-y-1">
                        <label className="text-[11px] text-slate-300 font-medium">Assign Verified Severity:</label>
                        <select
                          value={verifySeverityChoice}
                          onChange={(e) => setVerifySeverityChoice(e.target.value)}
                          className="w-full rounded border border-slate-800 bg-slate-900 px-2.5 py-1.5 text-white focus:outline-none focus:border-emerald-500"
                        >
                          <option value="critical">CRITICAL — Severe structural failure / hazard</option>
                          <option value="high">HIGH — Deep distress requiring priority repair</option>
                          <option value="medium">MEDIUM — Moderate cracking / early deterioration</option>
                          <option value="low">LOW — Minor surface distress</option>
                        </select>
                      </div>
                      <div className="space-y-1">
                        <label className="text-[11px] text-slate-300 font-medium">Inspector Log Remarks:</label>
                        <input
                          type="text"
                          value={verifyRemarksInput}
                          onChange={(e) => setVerifyRemarksInput(e.target.value)}
                          placeholder="e.g., Confirmed pothole depth >50mm on lane 2"
                          className="w-full rounded border border-slate-800 bg-slate-900 px-2.5 py-1.5 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                        />
                      </div>
                      <button
                        type="button"
                        disabled={verifying}
                        onClick={() => handleVerify(selectedIncident.id)}
                        className="w-full inline-flex items-center justify-center gap-1.5 rounded bg-emerald-600 hover:bg-emerald-500 px-3 py-1.5 text-xs font-semibold text-white transition-colors disabled:opacity-50 mt-1"
                      >
                        <ShieldCheck className="h-3.5 w-3.5" />
                        {verifying ? 'Committing Ground Truth...' : 'Commit Inspector Ground Truth'}
                      </button>
                    </div>
                  )}
                  <p className="text-[10px] text-slate-500 pt-1 border-t border-slate-800/80 italic">
                    Human inspector verification is the sole authoritative ground-truth label in CivicPulse AI.
                  </p>
                </div>

                {/* TIER 4: Experimental Supervised ML */}
                <div className="rounded-lg border border-slate-800 bg-slate-800/30 p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-300 flex items-center gap-1.5">
                      <AlertTriangle className="h-3.5 w-3.5 text-red-400" />
                      4. Supervised Model
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-red-950/60 text-red-400 border border-red-800/50">
                      {severityProfile?.experimentalSupervisedPrediction?.status || 'DATA_INSUFFICIENT'}
                    </span>
                  </div>
                  <div className="space-y-1.5 text-slate-400">
                    <div className="flex justify-between">
                      <span>Ground Truth Census:</span>
                      <span className="font-mono text-amber-400">
                        N = {severityProfile?.experimentalSupervisedPrediction?.currentVerifiedSampleSize ?? 6} /{' '}
                        {severityProfile?.experimentalSupervisedPrediction?.requiredSampleSize ?? 50} required
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Model Gate Status:</span>
                      <span className="font-mono text-red-300">LOCKED (Data Insufficient)</span>
                    </div>
                    <p className="text-[11px] text-slate-300 bg-slate-900/60 p-2 rounded border border-slate-800/60">
                      Supervised training is strictly gated to prevent false precision. Real operational data collection is ongoing.
                    </p>
                  </div>
                  <p className="text-[10px] text-slate-500 pt-1 border-t border-slate-800/80 italic">
                    Synthetic labels are forbidden in production intelligence.
                  </p>
                </div>
              </div>
            </div>

            {/* Bottom Actions */}
            <div className="flex items-center justify-between pt-3 border-t border-slate-800">
              <button
                type="button"
                onClick={() => {
                  const target = selectedIncident;
                  setSelectedIncident(null);
                  setSeverityProfile(null);
                  openDuplicateReview(target);
                }}
                className="inline-flex items-center gap-1.5 rounded-lg bg-amber-600/20 border border-amber-500/40 hover:bg-amber-600/30 px-3 py-2 text-xs font-semibold text-amber-300 transition-colors"
              >
                <ArrowRightLeft className="h-4 w-4" />
                Check Duplicates
              </button>

              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => {
                    setSelectedIncident(null);
                    setSeverityProfile(null);
                  }}
                  className="rounded-lg bg-slate-800 hover:bg-slate-700 px-4 py-2 text-xs font-semibold text-slate-300"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-md rounded-xl border border-slate-800 bg-slate-900 p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Upload className="h-5 w-5 text-emerald-400" />
                Upload Road Distress Photo
              </h3>
              <button
                type="button"
                onClick={() => setShowUploadModal(false)}
                className="rounded-lg p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <form onSubmit={handleReportSubmit} className="space-y-4 text-xs">
              <div className="space-y-1">
                <label className="text-slate-300 font-medium">Road / Corridor Name</label>
                <input
                  type="text"
                  value={roadName}
                  onChange={(e) => setRoadName(e.target.value)}
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-white focus:outline-none focus:border-emerald-500"
                  required
                />
              </div>

              <div className="space-y-1">
                <label className="text-slate-300 font-medium">Address / Landmark</label>
                <input
                  type="text"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-white focus:outline-none focus:border-emerald-500"
                  required
                />
              </div>

              <div className="space-y-1">
                <label className="text-slate-300 font-medium">Inspection Photograph (JPEG, PNG, WebP)</label>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2 text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-emerald-600 file:text-white hover:file:bg-emerald-500"
                  required
                />
              </div>

              <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-slate-300 space-y-1">
                <p className="font-semibold text-emerald-400 flex items-center gap-1">
                  <Sparkles className="h-3.5 w-3.5" /> Automated GPU & Deduplication Pipeline
                </p>
                <p className="text-[11px] text-slate-400">
                  Image will be stored in MinIO, perceptual hashes (dHash/pHash) generated, and nearby incidents checked within 10 meters via PostGIS.
                </p>
              </div>

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowUploadModal(false)}
                  className="rounded-lg bg-slate-800 hover:bg-slate-700 px-4 py-2 font-semibold text-slate-300"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={uploading || !uploadFile}
                  className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 px-4 py-2 font-semibold text-white transition-colors disabled:opacity-50"
                >
                  {uploading ? 'Analyzing...' : 'Submit & Analyze'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

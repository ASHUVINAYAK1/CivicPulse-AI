'use client';

import React, { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import {
  MapPin,
  Layers,
  Sliders,
  AlertTriangle,
  Info,
  RefreshCw,
  Filter,
  CheckCircle2,
  AlertCircle,
  Hash,
  Compass,
} from 'lucide-react';
import { API_BASE_URL } from '@/lib/constants';
import { MapIncidentItem, MapClusterItem } from '@/components/map/hotspot-map-view';

// Dynamically import Leaflet view with SSR disabled
const HotspotMapView = dynamic(
  () => import('@/components/map/hotspot-map-view'),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-[620px] w-full items-center justify-center rounded-xl border border-slate-800 bg-slate-950 text-slate-500">
        <div className="flex flex-col items-center gap-2">
          <RefreshCw className="h-6 w-6 animate-spin text-indigo-400" />
          <span className="text-xs">Initializing Geospatial GIS Viewer...</span>
        </div>
      </div>
    ),
  }
);

export default function MapPage() {
  const [algorithm, setAlgorithm] = useState<'kmeans' | 'dbscan' | 'raw'>('kmeans');
  const [kValue, setKValue] = useState<number>(4);
  const [epsMeters, setEpsMeters] = useState<number>(500);
  const [minSamples, setMinSamples] = useState<number>(3);
  const [damageTypeFilter, setDamageTypeFilter] = useState<string>('all');
  const [verificationFilter, setVerificationFilter] = useState<string>('all');

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [summaryData, setSummaryData] = useState<any>(null);
  const [clusteringResult, setClusteringResult] = useState<any>(null);
  const [selectedClusterIndex, setSelectedClusterIndex] = useState<number | null>(null);
  const [selectedIncident, setSelectedIncident] = useState<MapIncidentItem | null>(null);

  // Fetch Summary on mount
  useEffect(() => {
    async function fetchSummary() {
      try {
        const res = await fetch(`${API_BASE_URL}/hotspots/summary`);
        if (res.ok) {
          const json = await res.json();
          setSummaryData(json.data);
        }
      } catch (err: any) {
        console.warn('Could not load summary from API:', err.message);
      }
    }
    fetchSummary();
  }, []);

  // Fetch Hotspot Data based on active algorithm and parameters
  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      setLoading(true);
      setError(null);
      setSelectedIncident(null);

      try {
        let endpoint = '';
        const params = new URLSearchParams();
        if (damageTypeFilter !== 'all') params.append('damageType', damageTypeFilter);
        if (verificationFilter !== 'all') params.append('verificationStatus', verificationFilter);

        if (algorithm === 'kmeans') {
          params.append('k', kValue.toString());
          params.append('seed', '42');
          params.append('evaluateRange', 'true');
          endpoint = `${API_BASE_URL}/hotspots/kmeans?${params.toString()}`;
        } else if (algorithm === 'dbscan') {
          params.append('eps', epsMeters.toString());
          params.append('minSamples', minSamples.toString());
          endpoint = `${API_BASE_URL}/hotspots/dbscan?${params.toString()}`;
        } else {
          // Raw incidents view: fetch via summary or kmeans k=2 to get point list
          params.append('k', '2');
          endpoint = `${API_BASE_URL}/hotspots/kmeans?${params.toString()}`;
        }

        const res = await fetch(endpoint);
        if (!res.ok) {
          throw new Error(`Failed to fetch hotspots (${res.status} ${res.statusText})`);
        }
        const json = await res.json();
        if (isMounted) {
          setClusteringResult(json.data);
          setSelectedClusterIndex(null);
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'Error communicating with Hotspot API service.');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    loadData();
    return () => {
      isMounted = false;
    };
  }, [algorithm, kValue, epsMeters, minSamples, damageTypeFilter, verificationFilter]);

  // Format Map Incidents
  const mapIncidents: MapIncidentItem[] = useMemo(() => {
    if (!clusteringResult || !clusteringResult.incidents) return [];

    if (algorithm === 'kmeans') {
      const assignmentsMap = new Map<string, { cluster: number; dist: number }>();
      if (clusteringResult.result && clusteringResult.result.assignments) {
        clusteringResult.result.assignments.forEach((a: any) => {
          assignmentsMap.set(a.incidentId, {
            cluster: a.clusterIndex,
            dist: a.distanceToCentroidMeters,
          });
        });
      }

      return clusteringResult.incidents.map((inc: any) => {
        const assign = assignmentsMap.get(inc.id);
        return {
          id: inc.id,
          incidentNumber: inc.incidentNumber,
          title: inc.title,
          damageType: inc.damageType,
          severity: inc.severity,
          status: inc.status,
          latitude: inc.latitude,
          longitude: inc.longitude,
          address: inc.address,
          clusterIndex: assign ? assign.cluster : 0,
          distanceToCentroidMeters: assign ? assign.dist : undefined,
        };
      });
    } else if (algorithm === 'dbscan') {
      const diagMap = new Map<string, any>();
      if (clusteringResult.result && clusteringResult.result.diagnostics) {
        clusteringResult.result.diagnostics.forEach((d: any) => {
          diagMap.set(d.incidentId, d);
        });
      }

      return clusteringResult.incidents.map((inc: any) => {
        const diag = diagMap.get(inc.id);
        return {
          id: inc.id,
          incidentNumber: inc.incidentNumber,
          title: inc.title,
          damageType: inc.damageType,
          severity: inc.severity,
          status: inc.status,
          latitude: inc.latitude,
          longitude: inc.longitude,
          address: inc.address,
          clusterIndex: diag ? diag.clusterLabel : -1,
          pointType: diag ? diag.pointType : undefined,
          diagnosticReason: diag ? diag.classificationReason : undefined,
        };
      });
    } else {
      // Raw
      return clusteringResult.incidents.map((inc: any) => ({
        id: inc.id,
        incidentNumber: inc.incidentNumber,
        title: inc.title,
        damageType: inc.damageType,
        severity: inc.severity,
        status: inc.status,
        latitude: inc.latitude,
        longitude: inc.longitude,
        address: inc.address,
        clusterIndex: 0,
      }));
    }
  }, [clusteringResult, algorithm]);

  // Format Map Clusters
  const mapClusters: MapClusterItem[] = useMemo(() => {
    if (!clusteringResult || !clusteringResult.result || !clusteringResult.result.clusters) {
      return [];
    }

    return clusteringResult.result.clusters.map((c: any) => ({
      clusterIndex: c.clusterIndex,
      centroidLatitude: c.centroidLatitude,
      centroidLongitude: c.centroidLongitude,
      memberCount: c.memberCount,
      averageDistanceMeters: c.averageDistanceMeters,
      maxDistanceMeters: c.maxDistanceMeters,
      damageTypeBreakdown: c.damageTypeBreakdown || {},
      severityBreakdown: c.severityBreakdown || {},
      corePoints: c.corePoints,
      borderPoints: c.borderPoints,
    }));
  }, [clusteringResult]);

  return (
    <div className="space-y-6">
      {/* 1. Header & Mandatory Exploratory Notice */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-md bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold uppercase text-indigo-400 border border-indigo-500/20">
                Phase 3 Intelligence
              </span>
              <span className="rounded-md bg-emerald-500/10 px-2 py-0.5 text-xs font-semibold text-emerald-400 border border-emerald-500/20">
                PostGIS 3.4 Live
              </span>
            </div>
            <h1 className="mt-1 text-2xl font-bold tracking-tight text-white">
              Exploratory Incident Hotspot Analysis
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              Unsupervised spatial clustering (Spherical K-Means & Haversine DBSCAN) of verified municipal damage reports.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setKValue(4);
                setEpsMeters(500);
                setMinSamples(3);
                setDamageTypeFilter('all');
                setVerificationFilter('all');
              }}
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Reset Parameters
            </button>
          </div>
        </div>

        {/* Mandatory Governance Notice */}
        <div className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-4 text-xs text-amber-200">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
            <div className="space-y-1">
              <div className="font-semibold text-amber-300">
                Data Governance & Exploratory Analytics Policy
              </div>
              <p className="leading-relaxed text-amber-200/90">
                <strong>Clusters describe the spatial distribution of incidents currently available in the system. They are not validated predictions of future road damage.</strong>
              </p>
              <p className="text-[11px] text-amber-300/80">
                Note: Cluster centroids depicted on the map represent mathematical density centers calculated using 3D unit-sphere vector averages. They do not indicate the physical presence of road damage at that coordinate.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Top-Level Spatial Metrics KPIs */}
      {summaryData && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-sm">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Total GPS Incidents</span>
              <MapPin className="h-4 w-4 text-indigo-400" />
            </div>
            <div className="mt-2 text-2xl font-bold text-white">
              {summaryData.validIncidents}
            </div>
            <div className="mt-1 text-[11px] text-slate-500">
              100% WGS84 GPS Validated
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-sm">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Active Clusters</span>
              <Layers className="h-4 w-4 text-emerald-400" />
            </div>
            <div className="mt-2 text-2xl font-bold text-white">
              {algorithm === 'kmeans' ? kValue : clusteringResult?.result?.clusterCount || 0}
            </div>
            <div className="mt-1 text-[11px] text-slate-500">
              {algorithm === 'kmeans' ? `K-Means (K=${kValue})` : `DBSCAN (${clusteringResult?.result?.clusterCount || 0} zones)`}
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-sm">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Isolated Noise Points</span>
              <AlertCircle className="h-4 w-4 text-rose-400" />
            </div>
            <div className="mt-2 text-2xl font-bold text-white">
              {algorithm === 'dbscan' ? (clusteringResult?.result?.noiseCount ?? 0) : 'N/A'}
            </div>
            <div className="mt-1 text-[11px] text-slate-500">
              {algorithm === 'dbscan'
                ? `${((clusteringResult?.result?.noiseRatio ?? 0) * 100).toFixed(1)}% outside dense clusters`
                : 'K-Means forces all points into clusters'}
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-sm">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Spatial Coherence</span>
              <Compass className="h-4 w-4 text-amber-400" />
            </div>
            <div className="mt-2 text-2xl font-bold text-white">
              {clusteringResult?.result?.silhouetteScore?.toFixed(2) ?? '0.00'}
            </div>
            <div className="mt-1 text-[11px] text-slate-500">
              Haversine Silhouette Score [-1, +1]
            </div>
          </div>
        </div>
      )}

      {/* 3. Algorithm Selector, Parameter Controls & Filter Bar */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 backdrop-blur-md">
        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* Algorithm Buttons */}
          <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-950 p-1">
            <button
              onClick={() => setAlgorithm('kmeans')}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                algorithm === 'kmeans'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Spherical K-Means
            </button>
            <button
              onClick={() => setAlgorithm('dbscan')}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                algorithm === 'dbscan'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Haversine DBSCAN
            </button>
            <button
              onClick={() => setAlgorithm('raw')}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                algorithm === 'raw'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Raw Incident Markers
            </button>
          </div>

          {/* Dynamic Algorithm Parameters */}
          <div className="flex flex-wrap items-center gap-4 text-xs">
            {algorithm === 'kmeans' && (
              <div className="flex items-center gap-2">
                <span className="text-slate-400">Clusters (K):</span>
                <input
                  type="range"
                  min="2"
                  max="8"
                  value={kValue}
                  onChange={(e) => setKValue(parseInt(e.target.value, 10))}
                  className="h-1.5 w-24 cursor-pointer accent-indigo-500"
                />
                <span className="rounded bg-slate-800 px-2 py-0.5 font-bold text-white">
                  K = {kValue}
                </span>
              </div>
            )}

            {algorithm === 'dbscan' && (
              <>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Radius (eps):</span>
                  <input
                    type="range"
                    min="200"
                    max="1500"
                    step="50"
                    value={epsMeters}
                    onChange={(e) => setEpsMeters(parseInt(e.target.value, 10))}
                    className="h-1.5 w-24 cursor-pointer accent-indigo-500"
                  />
                  <span className="rounded bg-slate-800 px-2 py-0.5 font-bold text-white">
                    {epsMeters}m
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Min Samples:</span>
                  <select
                    value={minSamples}
                    onChange={(e) => setMinSamples(parseInt(e.target.value, 10))}
                    className="rounded bg-slate-800 px-2 py-1 text-xs text-white border border-slate-700 focus:outline-none"
                  >
                    <option value={2}>2 incidents</option>
                    <option value={3}>3 incidents (default)</option>
                    <option value={5}>5 incidents</option>
                  </select>
                </div>
              </>
            )}

            {/* Filters */}
            <div className="flex items-center gap-2 border-l border-slate-800 pl-4">
              <Filter className="h-3.5 w-3.5 text-slate-400" />
              <select
                value={damageTypeFilter}
                onChange={(e) => setDamageTypeFilter(e.target.value)}
                className="rounded bg-slate-800 px-2 py-1 text-xs text-slate-300 border border-slate-700"
              >
                <option value="all">All Damage Types</option>
                <option value="pothole">Potholes Only</option>
                <option value="alligator_crack">Alligator Cracks</option>
                <option value="D00">D00 Longitudinal</option>
                <option value="D10">D10 Transverse</option>
                <option value="D20">D20 Alligator</option>
                <option value="D40">D40 Pothole</option>
              </select>

              <select
                value={verificationFilter}
                onChange={(e) => setVerificationFilter(e.target.value)}
                className="rounded bg-slate-800 px-2 py-1 text-xs text-slate-300 border border-slate-700"
              >
                <option value="all">All Statuses</option>
                <option value="verified">Verified Only</option>
                <option value="unverified">Unverified Only</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Main Spatial Canvas & Cluster Drill-Down Sidebar */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
        {/* Left Map Canvas (3 cols) */}
        <div className="lg:col-span-3">
          {error ? (
            <div className="flex h-[620px] flex-col items-center justify-center rounded-xl border border-rose-900/50 bg-rose-950/20 p-6 text-center text-rose-300">
              <AlertCircle className="h-10 w-10 text-rose-400 mb-2" />
              <div className="font-semibold">Hotspot Service Error</div>
              <p className="mt-1 text-xs text-rose-300/80 max-w-md">{error}</p>
            </div>
          ) : (
            <HotspotMapView
              algorithm={algorithm}
              incidents={mapIncidents}
              clusters={mapClusters}
              selectedCluster={selectedClusterIndex}
              onSelectCluster={(c) => {
                setSelectedClusterIndex(c);
                setSelectedIncident(null);
              }}
              onSelectIncident={(inc) => setSelectedIncident(inc)}
            />
          )}
        </div>

        {/* Right Sidebar: Cluster Details & Incident Drill-down */}
        <div className="space-y-4 lg:col-span-1">
          {/* Active Filter / Cluster Header */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">Cluster Breakdown</h3>
              {selectedClusterIndex !== null && (
                <button
                  onClick={() => setSelectedClusterIndex(null)}
                  className="text-[11px] text-indigo-400 hover:text-indigo-300"
                >
                  Show All
                </button>
              )}
            </div>

            <div className="mt-3 space-y-2 max-h-[220px] overflow-y-auto pr-1">
              {mapClusters.map((cluster) => {
                const isSelected = selectedClusterIndex === cluster.clusterIndex;
                return (
                  <button
                    key={cluster.clusterIndex}
                    onClick={() =>
                      setSelectedClusterIndex(
                        isSelected ? null : cluster.clusterIndex
                      )
                    }
                    className={`w-full rounded-lg border p-2.5 text-left text-xs transition-all ${
                      isSelected
                        ? 'border-indigo-500 bg-indigo-950/40'
                        : 'border-slate-800 bg-slate-950/50 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between font-semibold">
                      <span className="flex items-center gap-1.5 text-white">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{
                            backgroundColor:
                              cluster.clusterIndex === 0
                                ? '#3b82f6'
                                : cluster.clusterIndex === 1
                                ? '#10b981'
                                : cluster.clusterIndex === 2
                                ? '#f59e0b'
                                : cluster.clusterIndex === 3
                                ? '#8b5cf6'
                                : '#ec4899',
                          }}
                        />
                        Cluster #{cluster.clusterIndex + 1}
                      </span>
                      <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300">
                        {cluster.memberCount} reports
                      </span>
                    </div>

                    <div className="mt-1.5 grid grid-cols-2 gap-1 text-[11px] text-slate-400">
                      <div>Avg: {cluster.averageDistanceMeters}m</div>
                      <div>Radius: {cluster.maxDistanceMeters}m</div>
                    </div>
                  </button>
                );
              })}

              {algorithm === 'dbscan' && clusteringResult?.result?.noiseCount > 0 && (
                <div className="rounded-lg border border-red-900/50 bg-red-950/20 p-2.5 text-xs">
                  <div className="flex items-center justify-between font-semibold text-red-300">
                    <span className="flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full bg-red-500" />
                      Isolated Outliers
                    </span>
                    <span>{clusteringResult.result.noiseCount} noise</span>
                  </div>
                  <p className="mt-1 text-[10px] text-red-300/80">
                    Incidents lacking sufficient neighbors within {epsMeters}m radius.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Incident Drill-down Card */}
          {selectedIncident ? (
            <div className="rounded-xl border border-indigo-500/50 bg-slate-900/90 p-4 shadow-xl backdrop-blur-md">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-indigo-400">
                  {selectedIncident.incidentNumber}
                </span>
                <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] uppercase font-semibold text-slate-300">
                  {selectedIncident.damageType}
                </span>
              </div>

              <div className="mt-2 text-xs font-semibold text-white">
                {selectedIncident.title}
              </div>
              <div className="mt-0.5 text-[11px] text-slate-400">
                {selectedIncident.address || 'Address unrecorded'}
              </div>

              <div className="mt-3 space-y-1.5 rounded-lg bg-slate-950 p-2 text-[11px]">
                <div className="flex justify-between">
                  <span className="text-slate-500">Severity:</span>
                  <span className="capitalize font-medium text-slate-300">
                    {selectedIncident.severity || 'Unassigned'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Status:</span>
                  <span className="capitalize font-medium text-slate-300">
                    {selectedIncident.status}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">GPS Coordinates:</span>
                  <span className="font-mono text-slate-300">
                    {selectedIncident.latitude.toFixed(4)}, {selectedIncident.longitude.toFixed(4)}
                  </span>
                </div>
                {selectedIncident.distanceToCentroidMeters !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-slate-500">Centroid Offset:</span>
                    <span className="font-medium text-indigo-400">
                      {selectedIncident.distanceToCentroidMeters}m
                    </span>
                  </div>
                )}
              </div>

              {selectedIncident.diagnosticReason && (
                <div className="mt-2 rounded bg-slate-950 p-2 text-[10px] text-slate-300 border border-slate-800">
                  <strong>Spatial Diagnostic:</strong>
                  <p className="mt-0.5 text-slate-400">{selectedIncident.diagnosticReason}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-slate-800 p-6 text-center text-xs text-slate-500">
              <Info className="mx-auto h-5 w-5 text-slate-600 mb-1" />
              Click any incident marker on the map to inspect spatial diagnostics and defect metadata.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

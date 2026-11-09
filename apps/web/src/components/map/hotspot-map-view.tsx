'use client';

import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Circle, Popup, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

export interface MapIncidentItem {
  id: string;
  incidentNumber: string;
  title: string;
  damageType: string;
  severity: string | null;
  status: string;
  latitude: number;
  longitude: number;
  address: string | null;
  clusterIndex: number; // -1 for noise
  pointType?: 'core' | 'border' | 'noise';
  distanceToCentroidMeters?: number;
  diagnosticReason?: string;
}

export interface MapClusterItem {
  clusterIndex: number;
  centroidLatitude: number;
  centroidLongitude: number;
  memberCount: number;
  averageDistanceMeters: number;
  maxDistanceMeters: number;
  damageTypeBreakdown: Record<string, number>;
  severityBreakdown: Record<string, number>;
  corePoints?: number;
  borderPoints?: number;
}

interface HotspotMapViewProps {
  algorithm: 'kmeans' | 'dbscan' | 'raw';
  incidents: MapIncidentItem[];
  clusters: MapClusterItem[];
  selectedCluster: number | null;
  onSelectCluster: (clusterIndex: number | null) => void;
  onSelectIncident?: (incident: MapIncidentItem) => void;
}

const CLUSTER_COLORS = [
  '#3b82f6', // blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
  '#14b8a6', // teal
];

export function getClusterColor(clusterIndex: number): string {
  if (clusterIndex < 0) return '#ef4444'; // Red for noise/outlier
  return CLUSTER_COLORS[clusterIndex % CLUSTER_COLORS.length];
}

export default function HotspotMapView({
  algorithm,
  incidents,
  clusters,
  selectedCluster,
  onSelectCluster,
  onSelectIncident,
}: HotspotMapViewProps) {
  // Center on Delhi NCR
  const centerLat = 28.6139;
  const centerLng = 77.2090;

  return (
    <div className="relative h-[620px] w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
      <MapContainer
        center={[centerLat, centerLng]}
        zoom={11}
        scrollWheelZoom={true}
        className="h-full w-full"
        style={{ background: '#020617' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://carto.com/" target="_blank">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {/* 1. Cluster Boundaries & Centroids (when K-Means or DBSCAN selected) */}
        {algorithm !== 'raw' &&
          clusters.map((cluster) => {
            const isSelected = selectedCluster === null || selectedCluster === cluster.clusterIndex;
            const color = getClusterColor(cluster.clusterIndex);

            return (
              <React.Fragment key={`cluster-group-${cluster.clusterIndex}`}>
                {/* Cluster Footprint Radius Circle */}
                {isSelected && cluster.maxDistanceMeters > 0 && (
                  <Circle
                    center={[cluster.centroidLatitude, cluster.centroidLongitude]}
                    radius={cluster.maxDistanceMeters}
                    pathOptions={{
                      color,
                      fillColor: color,
                      fillOpacity: selectedCluster === cluster.clusterIndex ? 0.15 : 0.05,
                      weight: 1.5,
                      dashArray: '4, 4',
                    }}
                  />
                )}

                {/* Centroid Marker (Distinct Large Symbol with Center Dot) */}
                <CircleMarker
                  center={[cluster.centroidLatitude, cluster.centroidLongitude]}
                  radius={12}
                  eventHandlers={{
                    click: () => onSelectCluster(cluster.clusterIndex),
                  }}
                  pathOptions={{
                    color: '#ffffff',
                    fillColor: color,
                    fillOpacity: 0.9,
                    weight: 3,
                  }}
                >
                  <Tooltip direction="top" offset={[0, -10]} opacity={0.95}>
                    <div className="text-xs font-semibold text-slate-900">
                      <div>Centroid of Cluster #{cluster.clusterIndex + 1}</div>
                      <div className="text-[10px] font-normal text-amber-700">
                        ⚠ Mathematical density center — NOT an incident location
                      </div>
                    </div>
                  </Tooltip>
                  <Popup>
                    <div className="space-y-1.5 p-1 text-xs text-slate-800">
                      <div className="font-bold text-slate-900">
                        Cluster #{cluster.clusterIndex + 1} Centroid
                      </div>
                      <div className="rounded bg-amber-50 p-1 text-[11px] text-amber-800 border border-amber-200">
                        <strong>Governance Notice:</strong> This point is a calculated spatial center. No physical damage exists at this exact point unless coincidental.
                      </div>
                      <div className="grid grid-cols-2 gap-1 text-[11px]">
                        <div>Member Incidents: <strong>{cluster.memberCount}</strong></div>
                        <div>Avg Distance: <strong>{cluster.averageDistanceMeters}m</strong></div>
                        <div>Max Radius: <strong>{cluster.maxDistanceMeters}m</strong></div>
                        {cluster.corePoints !== undefined && (
                          <div>Core / Border: <strong>{cluster.corePoints} / {cluster.borderPoints}</strong></div>
                        )}
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              </React.Fragment>
            );
          })}

        {/* 2. Individual Incident Markers */}
        {incidents.map((inc) => {
          const isSelected = selectedCluster === null || selectedCluster === inc.clusterIndex;
          if (!isSelected) return null;

          const color = algorithm === 'raw'
            ? (inc.severity === 'critical' ? '#ef4444' : inc.severity === 'high' ? '#f97316' : '#3b82f6')
            : getClusterColor(inc.clusterIndex);

          const isNoise = inc.clusterIndex === -1;

          return (
            <CircleMarker
              key={inc.id}
              center={[inc.latitude, inc.longitude]}
              radius={isNoise ? 5 : 7}
              eventHandlers={{
                click: () => onSelectIncident && onSelectIncident(inc),
              }}
              pathOptions={{
                color: isNoise ? '#ef4444' : '#ffffff',
                fillColor: color,
                fillOpacity: 0.85,
                weight: isNoise ? 2 : 1.5,
                dashArray: isNoise ? '2, 3' : undefined,
              }}
            >
              <Tooltip direction="top" offset={[0, -6]} opacity={0.95}>
                <div className="text-xs">
                  <div className="font-bold text-slate-900">
                    {inc.incidentNumber}: {inc.damageType.toUpperCase()}
                  </div>
                  <div className="text-slate-600">
                    {algorithm === 'dbscan' && isNoise ? '⚠️ Classified as Isolated Noise' : `Cluster #${inc.clusterIndex + 1}`}
                  </div>
                </div>
              </Tooltip>

              <Popup>
                <div className="min-w-[200px] space-y-2 p-1 text-xs text-slate-900">
                  <div className="flex items-center justify-between border-b border-slate-200 pb-1">
                    <span className="font-bold text-slate-900">{inc.incidentNumber}</span>
                    <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-slate-700">
                      {inc.damageType}
                    </span>
                  </div>

                  <div>
                    <div className="font-semibold">{inc.title}</div>
                    <div className="text-[11px] text-slate-500">{inc.address || 'Address unrecorded'}</div>
                  </div>

                  <div className="grid grid-cols-2 gap-1 rounded bg-slate-50 p-1.5 text-[11px]">
                    <div>Severity: <strong className="capitalize">{inc.severity || 'unassigned'}</strong></div>
                    <div>Status: <strong className="capitalize">{inc.status}</strong></div>
                    <div>Lat: <strong>{inc.latitude.toFixed(4)}</strong></div>
                    <div>Lng: <strong>{inc.longitude.toFixed(4)}</strong></div>
                  </div>

                  {algorithm !== 'raw' && (
                    <div className="rounded border border-indigo-100 bg-indigo-50/70 p-1.5 text-[11px] text-indigo-900">
                      {isNoise ? (
                        <div>
                          <strong className="text-red-700">Isolated Noise Point:</strong>
                          <p className="mt-0.5 text-[10px] text-slate-600">
                            {inc.diagnosticReason || 'Does not meet the spatial density threshold.'}
                          </p>
                        </div>
                      ) : (
                        <div>
                          <strong>Assigned to Cluster #{inc.clusterIndex + 1}</strong>
                          {inc.distanceToCentroidMeters !== undefined && (
                            <div className="text-[10px] text-slate-600">
                              Distance to Centroid: {inc.distanceToCentroidMeters}m
                            </div>
                          )}
                          {inc.pointType && (
                            <div className="text-[10px] capitalize text-slate-600">
                              Classification: {inc.pointType} point
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {/* Floating Map Legend */}
      <div className="absolute bottom-4 right-4 z-[1000] rounded-lg border border-slate-800 bg-slate-950/90 p-3 text-xs text-slate-300 shadow-xl backdrop-blur-md max-w-xs">
        <div className="font-semibold text-white mb-1.5">Map Legend</div>
        {algorithm === 'raw' ? (
          <div className="space-y-1 text-[11px]">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-red-500" /> Critical Severity
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-orange-500" /> High Severity
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-blue-500" /> Medium / Low / Other
            </div>
          </div>
        ) : (
          <div className="space-y-1.5 text-[11px]">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full border-2 border-white bg-indigo-500" /> Cluster Centroid (Mathematical center)
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-blue-500" /> Clustered Incident
            </div>
            {algorithm === 'dbscan' && (
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full border border-dashed border-red-500 bg-red-500/20" /> Noise / Isolated Incident
              </div>
            )}
            <div className="text-[10px] text-slate-500 pt-1 border-t border-slate-800">
              Dashed circle: Max cluster radius
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


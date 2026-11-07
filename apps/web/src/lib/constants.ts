/**
 * CivicPulse AI - Application Constants
 */

export const APP_NAME = 'CivicPulse AI';
export const APP_DESCRIPTION = 'Road Infrastructure Monitoring & Predictive Maintenance Platform';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api/v1';
export const ML_API_BASE_URL = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8000/api/v1';

export const DAMAGE_TYPES = {
  D00: { label: 'Longitudinal Crack', code: 'D00', color: '#f59e0b' },
  D10: { label: 'Transverse Crack', code: 'D10', color: '#ef4444' },
  D20: { label: 'Alligator Crack', code: 'D20', color: '#8b5cf6' },
  D40: { label: 'Pothole', code: 'D40', color: '#dc2626' },
  other: { label: 'Other Damage', code: 'other', color: '#6b7280' },
} as const;

export const SEVERITY_LEVELS = {
  low: { label: 'Low', color: '#22c55e', bg: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
  medium: { label: 'Medium', color: '#f59e0b', bg: 'bg-amber-500/10 text-amber-400 border-amber-500/20' },
  high: { label: 'High', color: '#f97316', bg: 'bg-orange-500/10 text-orange-400 border-orange-500/20' },
  critical: { label: 'Critical', color: '#ef4444', bg: 'bg-red-500/10 text-red-400 border-red-500/20' },
} as const;

export const PRIORITY_LEVELS = {
  low: { label: 'Low Priority', color: '#22c55e' },
  medium: { label: 'Medium Priority', color: '#f59e0b' },
  high: { label: 'High Priority', color: '#f97316' },
  urgent: { label: 'Urgent Priority', color: '#ef4444' },
} as const;

export const INCIDENT_STATUSES = {
  submitted: { label: 'Submitted', color: '#64748b' },
  ai_processing: { label: 'AI Processing', color: '#38bdf8' },
  ai_analyzed: { label: 'AI Analyzed', color: '#818cf8' },
  under_review: { label: 'Under Review', color: '#fbbf24' },
  verified: { label: 'Verified', color: '#34d399' },
  maintenance_assigned: { label: 'Maintenance Assigned', color: '#fb923c' },
  repair_in_progress: { label: 'Repair in Progress', color: '#f43f5e' },
  repair_completed: { label: 'Repair Completed', color: '#10b981' },
  closed: { label: 'Closed', color: '#94a3b8' },
  rejected: { label: 'Rejected', color: '#ef4444' },
} as const;

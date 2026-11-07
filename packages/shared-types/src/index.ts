/**
 * CivicPulse AI - Shared Types
 *
 * Common type definitions shared across the frontend and backend.
 */

// ---- User Roles ----
export type UserRole = 'citizen' | 'inspector' | 'administrator' | 'maintenance_team' | 'ml_admin';

// ---- Incident Types ----
export type DamageType = 'D00' | 'D10' | 'D20' | 'D40' | 'other';

export type Severity = 'low' | 'medium' | 'high' | 'critical';

export type IncidentStatus =
  | 'submitted'
  | 'ai_processing'
  | 'ai_analyzed'
  | 'under_review'
  | 'verified'
  | 'maintenance_assigned'
  | 'repair_in_progress'
  | 'repair_completed'
  | 'closed'
  | 'rejected';

export type Priority = 'low' | 'medium' | 'high' | 'urgent';

// ---- Maintenance Types ----
export type MaintenanceTaskStatus =
  | 'pending'
  | 'assigned'
  | 'in_progress'
  | 'completed'
  | 'verified'
  | 'cancelled';

// ---- ML Model Types ----
export type ModelStatus = 'experimental' | 'staging' | 'production' | 'archived';

export type PredictionType = 'damage_classification' | 'severity' | 'priority' | 'cost' | 'duration' | 'duplicate';

// ---- API Response Types ----
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
  meta?: {
    page?: number;
    limit?: number;
    total?: number;
  };
}

// ---- Health Check ----
export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  timestamp: string;
  services?: Record<string, {
    status: 'up' | 'down';
    latency_ms?: number;
  }>;
}

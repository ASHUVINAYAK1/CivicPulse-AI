/**
 * CivicPulse AI - Phase 3, Stage 4: Repair Cost & Duration Estimation Types
 */

export type EstimationStatus =
  | 'ESTIMATED'
  | 'MEASUREMENT_PENDING'
  | 'RATE_SOURCE_UNAVAILABLE'
  | 'ESTIMATION_INSUFFICIENT_DATA';

export type MeasurementSource =
  | 'field_measurement'
  | 'calibrated_optical'
  | 'unmeasured_missing';

export interface MaterialRateItem {
  id: string;
  name: string;
  unit: string; // e.g. "m3", "ton", "m2", "linear_meter"
  unitRateINR: number;
  description: string;
}

export interface LaborRateItem {
  role: string;
  hourlyRateINR: number;
  standardShiftHours: number;
}

export interface EquipmentRateItem {
  name: string;
  hourlyRateINR: number;
}

export interface RegionalRateSchedule {
  regionId: 'BENGALURU_METRO' | 'DELHI_NCR';
  regionName: string;
  scheduleVersion: string;
  effectiveDate: string;
  currency: 'INR';
  isOfficialDocument: false;
  issuingAuthority: string;
  documentTitle: string;
  officialStatusNotice: string;
  materials: Record<string, MaterialRateItem>;
  labor: Record<string, LaborRateItem>;
  equipment: Record<string, EquipmentRateItem>;
  productivity: {
    potholePatchingM2PerCrewHour: number;
    alligatorResurfacingM2PerCrewHour: number;
    crackSealingMetersPerCrewHour: number;
    setupAndTrafficControlHours: number;
    curingWaitHours: number;
  };
  overheadAndMobilizationPercentage: number;
}

export interface CostItemDetail {
  item: string;
  quantity: number;
  unit: string;
  unitRateINR: number;
  subtotalINR: number;
  rationale: string;
}

export interface CostBreakdown {
  materials: CostItemDetail[];
  labor: CostItemDetail[];
  equipment: CostItemDetail[];
  mobilizationSubtotalINR: number;
  subtotalMaterialsINR: number;
  subtotalLaborINR: number;
  subtotalEquipmentINR: number;
  totalEstimatedCostINR: number;
}

export interface DurationBreakdown {
  operationalHours: number;
  setupAndTrafficControlHours: number;
  curingWaitHours: number;
  totalEstimatedHours: number;
  totalEstimatedDays: number;
  crewSize: number;
  productivityRateUsed: string;
  operationalExclusions: string[];
}

export interface RepairEstimateResult {
  id?: string;
  incidentId: string;
  incidentNumber: string;
  title: string;
  address: string | null;
  region: 'BENGALURU_METRO' | 'DELHI_NCR' | 'OUT_OF_BOUNDS_AMBIGUOUS';
  estimationStatus: EstimationStatus;
  rateScheduleVersion: string;
  currency: 'INR';
  measurement: {
    measuredAreaSqMeters: number | null;
    depthMeters: number | null;
    measurementSource: MeasurementSource;
    opticalBoundingBoxAreaRatio: number;
    isPhysicalAreaMeasured: boolean;
    measurementCaveat: string;
  };
  costBreakdown: CostBreakdown | null;
  totalEstimatedCostINR: number | null;
  durationBreakdown: DurationBreakdown | null;
  totalEstimatedDurationHours: number | null;
  totalEstimatedDurationDays: number | null;
  assumptions: string[];
  exclusions: string[];
  computedAt: string;
  isActualExpenditure: false;
  governanceNotice: string;
}


/**
 * CivicPulse AI - Regional Rate Schedules & Labor/Material Cost Sources
 *
 * GOVERNANCE NOTICE:
 * An exhaustive repository audit confirmed that zero official state or municipal
 * Schedule of Rates (SoR) documents exist in the codebase.
 *
 * These schedules represent documented, configurable municipal parametric baselines.
 * They are versioned, isolated by administrative region, and must never be cross-applied.
 */

import { RegionalRateSchedule } from './types';

export const BENGALURU_RATE_SCHEDULE: RegionalRateSchedule = {
  regionId: 'BENGALURU_METRO',
  regionName: 'Bengaluru Metropolitan Area (BBMP / KPWD Zone)',
  scheduleVersion: 'KPWD-BBMP-PROVISIONAL-2026.1',
  effectiveDate: '2026-04-01',
  currency: 'INR',
  isOfficialDocument: false,
  issuingAuthority: 'Configurable Municipal Benchmark (Provisional Parametric)',
  documentTitle: 'Karnataka PWD Road Maintenance Engineering Benchmark Rates (Provisional)',
  officialStatusNotice:
    'Provisional Parametric Model: Authoritative BBMP/KPWD Schedule of Rates documents are not bundled in repository. These values are transparent municipal engineering defaults.',
  materials: {
    cold_mix_asphalt: {
      id: 'MAT-BLR-01',
      name: 'Cold Mix Bituminous Macadam (Patching)',
      unit: 'm3',
      unitRateINR: 4200.0,
      description: 'Pre-mixed cold asphalt binder for localized pothole cavity patching.',
    },
    hot_mix_bituminous: {
      id: 'MAT-BLR-02',
      name: 'Hot Mix Asphalt Concrete (Wearing Course)',
      unit: 'm3',
      unitRateINR: 5100.0,
      description: 'Hot mix asphalt for resurfacing extensive alligator base fatigue.',
    },
    tack_coat_emulsion: {
      id: 'MAT-BLR-03',
      name: 'Bituminous Tack Coat Primer',
      unit: 'm2',
      unitRateINR: 45.0,
      description: 'Bitumen emulsion tack coat applied to pre-cut excavation surfaces.',
    },
    granular_subbase: {
      id: 'MAT-BLR-04',
      name: 'Wet Mix Macadam / Granular Sub-base',
      unit: 'm3',
      unitRateINR: 1850.0,
      description: 'Crushed stone aggregate for deep sub-base structural stabilization.',
    },
    crack_sealant_polymer: {
      id: 'MAT-BLR-05',
      name: 'Polymer Modified Bitumen Crack Sealant',
      unit: 'linear_meter',
      unitRateINR: 95.0,
      description: 'Rubberized hot-pour sealant for linear crack sealing.',
    },
  },
  labor: {
    skilled_mason: {
      role: 'Skilled Road Mason / Asphalt Lead',
      hourlyRateINR: 120.0,
      standardShiftHours: 8,
    },
    unskilled_laborer: {
      role: 'Road Maintenance Crew Laborer',
      hourlyRateINR: 85.0,
      standardShiftHours: 8,
    },
    traffic_marshall: {
      role: 'Traffic Marshall / Safety Signalman',
      hourlyRateINR: 80.0,
      standardShiftHours: 8,
    },
  },
  equipment: {
    roller_compactor: {
      name: 'Vibratory Plate / Tandem Roller Compactor',
      hourlyRateINR: 450.0,
    },
    asphalt_cutter: {
      name: 'Diamond Blade Asphalt Joint Saw / Jackhammer',
      hourlyRateINR: 250.0,
    },
    blower_compressor: {
      name: 'Air Compressor & High-Pressure Dust Blower',
      hourlyRateINR: 200.0,
    },
  },
  productivity: {
    potholePatchingM2PerCrewHour: 1.5, // 4-person crew completes 1.5 m2 patch per hour
    alligatorResurfacingM2PerCrewHour: 4.0, // Resurfacing prep & inlay
    crackSealingMetersPerCrewHour: 15.0, // Hot pour routing & sealing
    setupAndTrafficControlHours: 1.0,
    curingWaitHours: 1.5,
  },
  overheadAndMobilizationPercentage: 10.0, // 10% mobilization and site management
};

export const DELHI_RATE_SCHEDULE: RegionalRateSchedule = {
  regionId: 'DELHI_NCR',
  regionName: 'National Capital Territory of Delhi (Delhi PWD / CPWD Zone)',
  scheduleVersion: 'CPWD-DELHI-PROVISIONAL-2026.1',
  effectiveDate: '2026-04-01',
  currency: 'INR',
  isOfficialDocument: false,
  issuingAuthority: 'Configurable Municipal Benchmark (Provisional Parametric)',
  documentTitle: 'Delhi PWD / CPWD Arterial Road Maintenance Benchmark Rates (Provisional)',
  officialStatusNotice:
    'Provisional Parametric Model: Authoritative Delhi PWD/CPWD Schedule of Rates documents are not bundled in repository. These values are transparent municipal engineering defaults.',
  materials: {
    cold_mix_asphalt: {
      id: 'MAT-DEL-01',
      name: 'Cold Mix Bituminous Macadam (Patching)',
      unit: 'm3',
      unitRateINR: 4500.0,
      description: 'High-grade cold mix binder for arterial pothole cavity repair in Delhi NCR.',
    },
    hot_mix_bituminous: {
      id: 'MAT-DEL-02',
      name: 'Dense Bituminous Macadam (DBM)',
      unit: 'm3',
      unitRateINR: 5400.0,
      description: 'Heavy vehicle load bearing asphalt concrete for flyovers and transit corridors.',
    },
    tack_coat_emulsion: {
      id: 'MAT-DEL-03',
      name: 'Rapid Setting Bituminous Emulsion (RS-1)',
      unit: 'm2',
      unitRateINR: 50.0,
      description: 'Tack coat applied across trimmed pavement cavity.',
    },
    granular_subbase: {
      id: 'MAT-DEL-04',
      name: 'Graded Crushed Stone Sub-base (GSB)',
      unit: 'm3',
      unitRateINR: 1950.0,
      description: 'Sub-grade base material for structural pavement stabilization.',
    },
    crack_sealant_polymer: {
      id: 'MAT-DEL-05',
      name: 'Modified Bitumen Hot-Pour Sealant',
      unit: 'linear_meter',
      unitRateINR: 105.0,
      description: 'Polymer elastomeric sealant for transverse thermal cracks.',
    },
  },
  labor: {
    skilled_mason: {
      role: 'Skilled Road Mason / Finisher',
      hourlyRateINR: 135.0,
      standardShiftHours: 8,
    },
    unskilled_laborer: {
      role: 'Tarmac Maintenance Laborer',
      hourlyRateINR: 95.0,
      standardShiftHours: 8,
    },
    traffic_marshall: {
      role: 'Delhi Corridor Traffic Safety Marshall',
      hourlyRateINR: 90.0,
      standardShiftHours: 8,
    },
  },
  equipment: {
    roller_compactor: {
      name: 'Mini Tandem Vibratory Roller',
      hourlyRateINR: 500.0,
    },
    asphalt_cutter: {
      name: 'Heavy Duty Asphalt Cutter & Chisel',
      hourlyRateINR: 280.0,
    },
    blower_compressor: {
      name: 'Industrial Pneumatic Compressor',
      hourlyRateINR: 220.0,
    },
  },
  productivity: {
    potholePatchingM2PerCrewHour: 1.25, // Arterial corridor safety restrictions
    alligatorResurfacingM2PerCrewHour: 3.5,
    crackSealingMetersPerCrewHour: 12.0,
    setupAndTrafficControlHours: 1.5, // Strict arterial cordon setup
    curingWaitHours: 2.0,
  },
  overheadAndMobilizationPercentage: 12.0, // 12% mobilization for metropolitan arterial corridors
};

/**
 * Returns regional rate schedule by validated region identifier.
 * Returns null if region rate schedule is unconfigured (RATE_SOURCE_UNAVAILABLE).
 */
export function getRegionalRateSchedule(
  region: 'BENGALURU_METRO' | 'DELHI_NCR' | 'OUT_OF_BOUNDS_AMBIGUOUS' | string
): RegionalRateSchedule | null {
  if (region === 'BENGALURU_METRO') {
    return BENGALURU_RATE_SCHEDULE;
  }
  if (region === 'DELHI_NCR') {
    return DELHI_RATE_SCHEDULE;
  }
  return null;
}


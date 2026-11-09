/**
 * CivicPulse AI - Parametric Repair Cost & Duration Estimator
 *
 * Enforces strict scientific and engineering boundaries:
 * 1. Pixel/optical bounding box ratio is NOT physical square meters.
 * 2. If physical repair area is unmeasured, returns MEASUREMENT_PENDING.
 * 3. Rates are isolated by administrative region; never cross-applied.
 * 4. Outputs are parametric estimates, never presented as actual expenditures.
 */

import {
  CostBreakdown,
  CostItemDetail,
  DurationBreakdown,
  MeasurementSource,
  RegionalRateSchedule,
  RepairEstimateResult,
} from './types';
import { getRegionalRateSchedule } from './rate-schedules';
import { classifyIncidentCoordinates } from '../maintenance-priority/geo-classifier';

export interface ParametricEstimationParams {
  incident: {
    id: string;
    incidentNumber: string;
    title: string;
    address: string | null;
    damageType: string;
    latitude: number;
    longitude: number;
    normalizedArea?: number;
  };
  measuredAreaSqMeters?: number | null;
  depthMeters?: number | null;
  measurementSource?: MeasurementSource;
  materialTypeOverride?: string | null;
}

export function estimateRepairCostAndDuration(
  params: ParametricEstimationParams
): RepairEstimateResult {
  const { incident, measuredAreaSqMeters, depthMeters, measurementSource, materialTypeOverride } = params;

  // 1. Regional Classification
  const geo = classifyIncidentCoordinates(incident.latitude, incident.longitude, incident.incidentNumber);
  const schedule: RegionalRateSchedule | null = getRegionalRateSchedule(geo.region);

  const opticalRatio = incident.normalizedArea ?? 0.04;
  const isPhysicalMeasured = typeof measuredAreaSqMeters === 'number' && measuredAreaSqMeters > 0;
  const depth = depthMeters && depthMeters > 0 ? depthMeters : 0.05; // 50mm default asphalt lift depth

  // 2. Region / Rate Schedule Availability Check
  if (!schedule) {
    return {
      incidentId: incident.id,
      incidentNumber: incident.incidentNumber,
      title: incident.title,
      address: incident.address,
      region: geo.region,
      estimationStatus: 'RATE_SOURCE_UNAVAILABLE',
      rateScheduleVersion: 'NONE_AVAILABLE',
      currency: 'INR',
      measurement: {
        measuredAreaSqMeters: measuredAreaSqMeters || null,
        depthMeters: depth,
        measurementSource: measurementSource || 'unmeasured_missing',
        opticalBoundingBoxAreaRatio: opticalRatio,
        isPhysicalAreaMeasured: isPhysicalMeasured,
        measurementCaveat: 'Coordinates fall outside configured municipal Schedule of Rates jurisdictions.',
      },
      costBreakdown: null,
      totalEstimatedCostINR: null,
      durationBreakdown: null,
      totalEstimatedDurationHours: null,
      totalEstimatedDurationDays: null,
      assumptions: ['No rate schedule available for territory.'],
      exclusions: ['All financial estimation excluded.'],
      computedAt: new Date().toISOString(),
      isActualExpenditure: false,
      governanceNotice:
        'Rate schedule unavailable: Regional municipal rate configuration missing. Estimation aborted to avoid cross-regional rate leakage.',
    };
  }

  // 3. Physical Measurement Validation
  // If physical dimensions are unknown, REFUSE to guess physical area from pixel bounding boxes!
  if (!isPhysicalMeasured) {
    return {
      incidentId: incident.id,
      incidentNumber: incident.incidentNumber,
      title: incident.title,
      address: incident.address,
      region: geo.region,
      estimationStatus: 'MEASUREMENT_PENDING',
      rateScheduleVersion: schedule.scheduleVersion,
      currency: 'INR',
      measurement: {
        measuredAreaSqMeters: null,
        depthMeters: null,
        measurementSource: 'unmeasured_missing',
        opticalBoundingBoxAreaRatio: opticalRatio,
        isPhysicalAreaMeasured: false,
        measurementCaveat:
          'Physical repair dimensions are unmeasured. 2D bounding-box pixel area ratio on camera frame cannot be converted into physical ground area without field measurement or photogrammetric ground calibration.',
      },
      costBreakdown: null,
      totalEstimatedCostINR: null,
      durationBreakdown: null,
      totalEstimatedDurationHours: null,
      totalEstimatedDurationDays: null,
      assumptions: [
        `Configured rate schedule: ${schedule.documentTitle} (${schedule.scheduleVersion})`,
        'Estimation will execute automatically once field inspector logs cavity length, width, and depth.',
      ],
      exclusions: [
        'Cost calculation withheld pending empirical physical measurement.',
        'Duration calculation withheld pending surface area determination.',
      ],
      computedAt: new Date().toISOString(),
      isActualExpenditure: false,
      governanceNotice:
        'Measurement Pending: CivicPulse AI prohibits fabricating physical square meters from uncalibrated image pixels. Submit inspector ground measurement to generate financial estimate.',
    };
  }

  const area = measuredAreaSqMeters!;
  const damageCategory = (incident.damageType || '').toLowerCase();
  const isPothole = damageCategory.includes('pothole') || damageCategory.includes('d40');
  const isAlligator = damageCategory.includes('alligator') || damageCategory.includes('d20');
  const isLinearCrack = damageCategory.includes('crack') || damageCategory.includes('d00') || damageCategory.includes('d10');

  // 4. Parametric Cost Calculation
  const materialItems: CostItemDetail[] = [];
  const laborItems: CostItemDetail[] = [];
  const equipmentItems: CostItemDetail[] = [];

  // Determine material selection & volume
  if (isPothole) {
    const volumeM3 = Number((area * depth).toFixed(3));
    const matKey = materialTypeOverride || 'cold_mix_asphalt';
    const matConfig = schedule.materials[matKey] || schedule.materials.cold_mix_asphalt;

    materialItems.push({
      item: matConfig.name,
      quantity: volumeM3,
      unit: matConfig.unit,
      unitRateINR: matConfig.unitRateINR,
      subtotalINR: Number((volumeM3 * matConfig.unitRateINR).toFixed(2)),
      rationale: `Cavity patch volume: ${area.toFixed(2)} m2 x ${depth.toFixed(3)}m depth = ${volumeM3} m3 of ${matConfig.name}.`,
    });

    // Tack coat emulsion primer
    const tackCoat = schedule.materials.tack_coat_emulsion;
    materialItems.push({
      item: tackCoat.name,
      quantity: area,
      unit: tackCoat.unit,
      unitRateINR: tackCoat.unitRateINR,
      subtotalINR: Number((area * tackCoat.unitRateINR).toFixed(2)),
      rationale: `Sub-surface tack coat primer across excavated ${area.toFixed(2)} m2 cavity area.`,
    });
  } else if (isAlligator) {
    const volumeM3 = Number((area * depth).toFixed(3));
    const matConfig = schedule.materials.hot_mix_bituminous;

    materialItems.push({
      item: matConfig.name,
      quantity: volumeM3,
      unit: matConfig.unit,
      unitRateINR: matConfig.unitRateINR,
      subtotalINR: Number((volumeM3 * matConfig.unitRateINR).toFixed(2)),
      rationale: `Structural mill & inlay: ${area.toFixed(2)} m2 x ${depth.toFixed(3)}m depth = ${volumeM3} m3 dense bituminous concrete.`,
    });

    const tackCoat = schedule.materials.tack_coat_emulsion;
    materialItems.push({
      item: tackCoat.name,
      quantity: area,
      unit: tackCoat.unit,
      unitRateINR: tackCoat.unitRateINR,
      subtotalINR: Number((area * tackCoat.unitRateINR).toFixed(2)),
      rationale: `Bonding tack coat over milled base (${area.toFixed(2)} m2).`,
    });
  } else {
    // Crack sealing: assume 3 linear meters per m2 of distress
    const linearMeters = Math.max(2.0, area * 3.0);
    const sealant = schedule.materials.crack_sealant_polymer;

    materialItems.push({
      item: sealant.name,
      quantity: Number(linearMeters.toFixed(1)),
      unit: sealant.unit,
      unitRateINR: sealant.unitRateINR,
      subtotalINR: Number((linearMeters * sealant.unitRateINR).toFixed(2)),
      rationale: `Polymer modified hot-pour crack sealant along ${linearMeters.toFixed(1)} linear meters of routed fissure.`,
    });
  }

  // 5. Duration Calculation
  let productivityRate = schedule.productivity.potholePatchingM2PerCrewHour;
  let productivityDesc = `${productivityRate} m2/crew-hour for cavity trimming and compaction`;

  if (isAlligator) {
    productivityRate = schedule.productivity.alligatorResurfacingM2PerCrewHour;
    productivityDesc = `${productivityRate} m2/crew-hour for asphalt inlay resurfacing`;
  } else if (isLinearCrack) {
    productivityRate = schedule.productivity.crackSealingMetersPerCrewHour;
    productivityDesc = `${productivityRate} linear meters/crew-hour for crack cleaning and hot-pour sealing`;
  }

  const netOperationalHours = Number(Math.max(0.5, area / productivityRate).toFixed(2));
  const setupHours = schedule.productivity.setupAndTrafficControlHours;
  const curingHours = schedule.productivity.curingWaitHours;
  const totalHours = Number((netOperationalHours + setupHours + curingHours).toFixed(2));
  const totalDays = Number((totalHours / 8.0).toFixed(2));

  // 6. Labor Calculation (Standard 4-person maintenance crew: 1 skilled mason, 2 laborers, 1 traffic marshall)
  const activeLaborHours = netOperationalHours + setupHours;

  const mason = schedule.labor.skilled_mason;
  laborItems.push({
    item: mason.role,
    quantity: Number(activeLaborHours.toFixed(1)),
    unit: 'hours',
    unitRateINR: mason.hourlyRateINR,
    subtotalINR: Number((activeLaborHours * mason.hourlyRateINR).toFixed(2)),
    rationale: `1 Skilled Mason / Asphalt Lead for excavation cut, laying, and leveling (${activeLaborHours.toFixed(1)} hrs).`,
  });

  const laborer = schedule.labor.unskilled_laborer;
  const laborerTotalHours = activeLaborHours * 2; // 2 laborers
  laborItems.push({
    item: `${laborer.role} (2-person team)`,
    quantity: Number(laborerTotalHours.toFixed(1)),
    unit: 'man-hours',
    unitRateINR: laborer.hourlyRateINR,
    subtotalINR: Number((laborerTotalHours * laborer.hourlyRateINR).toFixed(2)),
    rationale: `2 Laborers for debris removal, material wheeling, and raking (${activeLaborHours.toFixed(1)} hrs x 2 = ${laborerTotalHours.toFixed(1)} man-hours).`,
  });

  const marshall = schedule.labor.traffic_marshall;
  laborItems.push({
    item: marshall.role,
    quantity: Number(activeLaborHours.toFixed(1)),
    unit: 'hours',
    unitRateINR: marshall.hourlyRateINR,
    subtotalINR: Number((activeLaborHours * marshall.hourlyRateINR).toFixed(2)),
    rationale: `1 Traffic Safety Marshall for corridor diversion and safety signage (${activeLaborHours.toFixed(1)} hrs).`,
  });

  // 7. Equipment Calculation
  const roller = schedule.equipment.roller_compactor;
  equipmentItems.push({
    item: roller.name,
    quantity: Number(activeLaborHours.toFixed(1)),
    unit: 'machine-hours',
    unitRateINR: roller.hourlyRateINR,
    subtotalINR: Number((activeLaborHours * roller.hourlyRateINR).toFixed(2)),
    rationale: `Vibratory compaction of asphalt layers (${activeLaborHours.toFixed(1)} hrs).`,
  });

  const cutter = schedule.equipment.asphalt_cutter;
  const cutterHours = Math.max(0.5, Number((activeLaborHours * 0.5).toFixed(1)));
  equipmentItems.push({
    item: cutter.name,
    quantity: cutterHours,
    unit: 'machine-hours',
    unitRateINR: cutter.hourlyRateINR,
    subtotalINR: Number((cutterHours * cutter.hourlyRateINR).toFixed(2)),
    rationale: `Straight vertical edge box cutting for patch keying (${cutterHours} hrs).`,
  });

  const blower = schedule.equipment.blower_compressor;
  equipmentItems.push({
    item: blower.name,
    quantity: 0.5,
    unit: 'machine-hours',
    unitRateINR: blower.hourlyRateINR,
    subtotalINR: Number((0.5 * blower.hourlyRateINR).toFixed(2)),
    rationale: 'High-pressure air jet dusting of pre-cut cavity surface (0.5 hrs).',
  });

  // 8. Financial Summation
  const subtotalMaterials = Number(materialItems.reduce((acc, m) => acc + m.subtotalINR, 0).toFixed(2));
  const subtotalLabor = Number(laborItems.reduce((acc, l) => acc + l.subtotalINR, 0).toFixed(2));
  const subtotalEquipment = Number(equipmentItems.reduce((acc, e) => acc + e.subtotalINR, 0).toFixed(2));

  const directCost = subtotalMaterials + subtotalLabor + subtotalEquipment;
  const mobilizationPct = schedule.overheadAndMobilizationPercentage;
  const mobilizationSubtotal = Number(((directCost * mobilizationPct) / 100).toFixed(2));
  const totalCost = Number((directCost + mobilizationSubtotal).toFixed(2));

  const costBreakdown: CostBreakdown = {
    materials: materialItems,
    labor: laborItems,
    equipment: equipmentItems,
    mobilizationSubtotalINR: mobilizationSubtotal,
    subtotalMaterialsINR: subtotalMaterials,
    subtotalLaborINR: subtotalLabor,
    subtotalEquipmentINR: subtotalEquipment,
    totalEstimatedCostINR: totalCost,
  };

  const durationBreakdown: DurationBreakdown = {
    operationalHours: netOperationalHours,
    setupAndTrafficControlHours: setupHours,
    curingWaitHours: curingHours,
    totalEstimatedHours: totalHours,
    totalEstimatedDays: Math.max(0.25, totalDays),
    crewSize: 4,
    productivityRateUsed: productivityDesc,
    operationalExclusions: [
      'Night-time work differential not applied.',
      'Monsoon weather stoppage delays excluded.',
      'Extreme arterial peak-hour lane rental fees excluded.',
    ],
  };

  return {
    incidentId: incident.id,
    incidentNumber: incident.incidentNumber,
    title: incident.title,
    address: incident.address,
    region: geo.region,
    estimationStatus: 'ESTIMATED',
    rateScheduleVersion: schedule.scheduleVersion,
    currency: 'INR',
    measurement: {
      measuredAreaSqMeters: area,
      depthMeters: depth,
      measurementSource: measurementSource || 'field_measurement',
      opticalBoundingBoxAreaRatio: opticalRatio,
      isPhysicalAreaMeasured: true,
      measurementCaveat: `Field measurement source: ${measurementSource || 'field_measurement'}. Physical dimensions validated.`,
    },
    costBreakdown,
    totalEstimatedCostINR: totalCost,
    durationBreakdown,
    totalEstimatedDurationHours: totalHours,
    totalEstimatedDurationDays: Math.max(0.25, totalDays),
    assumptions: [
      `Regional rate schedule: ${schedule.documentTitle} (${schedule.scheduleVersion})`,
      `Standard 4-person municipal maintenance squad with dedicated traffic marshall.`,
      `Standard asphalt compaction depth: ${(depth * 1000).toFixed(0)}mm lift.`,
      `Mobilization & site management overhead: ${mobilizationPct}%.`,
    ],
    exclusions: [
      'Sub-grade geotechnical reconstruction excluded.',
      'Utility duct relocation or drainage pipe repairs excluded.',
      'Municipal administrative permitting fees excluded.',
    ],
    computedAt: new Date().toISOString(),
    isActualExpenditure: false,
    governanceNotice:
      'Parametric Decision Support Estimate: This is an engineering projection, not an actual expenditure or contractor tender. Actual cost will be recorded upon physical work order completion.',
  };
}

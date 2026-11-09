/**
 * CivicPulse AI - Repair Cost & Duration Estimation Service
 * Orchestrates regional rate lookups, parametric cost arithmetic,
 * database persistence in repair_cost_estimates, and summary reporting.
 */

import { prisma } from '../../config/prisma';
import {
  BENGALURU_RATE_SCHEDULE,
  DELHI_RATE_SCHEDULE,
} from './rate-schedules';
import {
  estimateRepairCostAndDuration,
  ParametricEstimationParams,
} from './estimator';
import {
  EstimationStatus,
  MeasurementSource,
  RegionalRateSchedule,
  RepairEstimateResult,
} from './types';
import { classifyIncidentCoordinates } from '../maintenance-priority/geo-classifier';

export class RepairEstimationService {
  /**
   * Calculates and saves parametric repair estimate for an incident.
   */
  async calculateEstimate(params: {
    incidentId: string;
    measuredAreaSqMeters?: number | null;
    depthMeters?: number | null;
    measurementSource?: MeasurementSource;
    materialTypeOverride?: string | null;
    taskId?: string | null;
  }): Promise<RepairEstimateResult> {
    const { incidentId, measuredAreaSqMeters, depthMeters, measurementSource, materialTypeOverride, taskId } = params;

    const incident = await prisma.incident.findUnique({
      where: { id: incidentId },
      include: {
        predictions: { orderBy: { createdAt: 'desc' }, take: 1 },
      },
    });

    if (!incident) {
      throw new Error(`Incident with ID ${incidentId} not found.`);
    }

    const lat = Number(incident.latitude);
    const lng = Number(incident.longitude);

    let normalizedArea = 0.04;
    const topPred = incident.predictions[0];
    if (topPred && topPred.predictionPayload) {
      const payload: any = topPred.predictionPayload;
      if (Array.isArray(payload.detections) && payload.detections[0]?.normalizedArea) {
        normalizedArea = payload.detections[0].normalizedArea;
      }
    }

    const estimationParams: ParametricEstimationParams = {
      incident: {
        id: incident.id,
        incidentNumber: incident.incidentNumber,
        title: incident.title,
        address: incident.address,
        damageType: incident.damageType,
        latitude: lat,
        longitude: lng,
        normalizedArea,
      },
      measuredAreaSqMeters: measuredAreaSqMeters ?? null,
      depthMeters: depthMeters ?? null,
      measurementSource: measurementSource ?? 'unmeasured_missing',
      materialTypeOverride: materialTypeOverride ?? null,
    };

    const result = estimateRepairCostAndDuration(estimationParams);

    // Save or update in database
    await prisma.$executeRawUnsafe(
      `
      INSERT INTO repair_cost_estimates (
        id, incident_id, task_id, region, rate_schedule_version,
        estimation_status, measured_area_sq_meters, measurement_source,
        depth_meters, material_type, material_quantity, material_unit,
        material_cost, labor_hours, labor_cost, equipment_hours,
        equipment_cost, mobilization_cost, total_estimated_cost,
        currency, estimated_duration_hours, estimated_duration_days,
        crew_size, assumptions, breakdown, missing_information,
        computed_at, is_actual_cost
      ) VALUES (
        gen_random_uuid(), $1::uuid, $2::uuid, $3, $4,
        $5, $6, $7,
        $8, $9, $10, $11,
        $12, $13, $14, $15,
        $16, $17, $18,
        $19, $20, $21,
        $22, $23::jsonb, $24::jsonb, $25::jsonb,
        CURRENT_TIMESTAMP, FALSE
      )
    `,
      result.incidentId,
      taskId || null,
      result.region,
      result.rateScheduleVersion,
      result.estimationStatus,
      result.measurement.measuredAreaSqMeters,
      result.measurement.measurementSource,
      result.measurement.depthMeters,
      materialTypeOverride || null,
      result.costBreakdown?.materials[0]?.quantity || null,
      result.costBreakdown?.materials[0]?.unit || null,
      result.costBreakdown?.subtotalMaterialsINR || null,
      result.durationBreakdown?.operationalHours || null,
      result.costBreakdown?.subtotalLaborINR || null,
      result.durationBreakdown?.operationalHours || null,
      result.costBreakdown?.subtotalEquipmentINR || null,
      result.costBreakdown?.mobilizationSubtotalINR || null,
      result.totalEstimatedCostINR,
      result.currency,
      result.totalEstimatedDurationHours,
      result.totalEstimatedDurationDays,
      result.durationBreakdown?.crewSize || null,
      JSON.stringify(result.assumptions),
      JSON.stringify({ cost: result.costBreakdown, duration: result.durationBreakdown }),
      JSON.stringify({ caveat: result.measurement.measurementCaveat, exclusions: result.exclusions })
    );

    return result;
  }

  /**
   * Retrieves latest repair estimate for an incident.
   */
  async getIncidentEstimate(incidentId: string): Promise<RepairEstimateResult> {
    const existing = await prisma.repairCostEstimate.findFirst({
      where: { incidentId },
      orderBy: { computedAt: 'desc' },
    });

    if (existing) {
      const inc = await prisma.incident.findUnique({ where: { id: incidentId } });
      const lat = Number(inc?.latitude ?? 0);
      const lng = Number(inc?.longitude ?? 0);
      const geo = classifyIncidentCoordinates(lat, lng, inc?.incidentNumber);

      const bd: any = existing.breakdown || {};
      const costBD = bd.cost || null;
      const durBD = bd.duration || null;

      return {
        id: existing.id,
        incidentId: existing.incidentId,
        incidentNumber: inc?.incidentNumber || '',
        title: inc?.title || '',
        address: inc?.address || null,
        region: geo.region,
        estimationStatus: existing.estimationStatus as EstimationStatus,
        rateScheduleVersion: existing.rateScheduleVersion,
        currency: 'INR',
        measurement: {
          measuredAreaSqMeters: existing.measuredAreaSqMeters,
          depthMeters: existing.depthMeters,
          measurementSource: existing.measurementSource as MeasurementSource,
          opticalBoundingBoxAreaRatio: 0.04,
          isPhysicalAreaMeasured: Boolean(existing.measuredAreaSqMeters && existing.measuredAreaSqMeters > 0),
          measurementCaveat:
            existing.measuredAreaSqMeters && existing.measuredAreaSqMeters > 0
              ? `Field measurement logged (${existing.measurementSource}).`
              : 'Physical repair dimensions are unmeasured. Cannot convert pixel area to physical m2.',
        },
        costBreakdown: costBD,
        totalEstimatedCostINR: existing.totalEstimatedCost,
        durationBreakdown: durBD,
        totalEstimatedDurationHours: existing.estimatedDurationHours,
        totalEstimatedDurationDays: existing.estimatedDurationDays,
        assumptions: (existing.assumptions as any) || [],
        exclusions: ((existing.missingInformation as any)?.exclusions as any) || [],
        computedAt: existing.computedAt.toISOString(),
        isActualExpenditure: false,
        governanceNotice:
          'Parametric Decision Support Estimate: This is an engineering projection, not an actual expenditure or contractor tender. Actual cost will be recorded upon physical work order completion.',
      };
    }

    // Compute default estimate (which returns MEASUREMENT_PENDING unless measured)
    return this.calculateEstimate({ incidentId });
  }

  /**
   * Summary census of repair estimates, regional distribution, and regression readiness.
   */
  async getEstimatesSummary(): Promise<{
    totalIncidents: number;
    estimationStatusBreakdown: Record<EstimationStatus, number>;
    measuredIncidentsCount: number;
    measurementPendingCount: number;
    rateSourceUnavailableCount: number;
    averageEstimatedCostINR: number;
    regionalBreakdown: {
      bengaluru: number;
      delhi: number;
      ambiguous: number;
    };
    supervisedRegressionReadiness: {
      status: 'DATA_INSUFFICIENT';
      currentCompletedRepairs: number;
      minimumRequiredRepairs: number;
      canTrainRegression: false;
      prohibitionRationale: string;
    };
    governanceNotice: string;
  }> {
    const incidents = await prisma.incident.findMany({
      include: {
        repairEstimates: { orderBy: { computedAt: 'desc' }, take: 1 },
      },
    });

    const statusCounts: Record<EstimationStatus, number> = {
      ESTIMATED: 0,
      MEASUREMENT_PENDING: 0,
      RATE_SOURCE_UNAVAILABLE: 0,
      ESTIMATION_INSUFFICIENT_DATA: 0,
    };

    let totalCostSum = 0;
    let estimatedCount = 0;

    const regionalCounts = {
      bengaluru: 0,
      delhi: 0,
      ambiguous: 0,
    };

    for (const inc of incidents) {
      const lat = Number(inc.latitude);
      const lng = Number(inc.longitude);
      const geo = classifyIncidentCoordinates(lat, lng, inc.incidentNumber);

      if (geo.region === 'BENGALURU_METRO') regionalCounts.bengaluru++;
      else if (geo.region === 'DELHI_NCR') regionalCounts.delhi++;
      else regionalCounts.ambiguous++;

      const est = inc.repairEstimates[0];
      if (est) {
        statusCounts[est.estimationStatus as EstimationStatus] =
          (statusCounts[est.estimationStatus as EstimationStatus] || 0) + 1;
        if (est.estimationStatus === 'ESTIMATED' && est.totalEstimatedCost) {
          totalCostSum += est.totalEstimatedCost;
          estimatedCount++;
        }
      } else {
        // Uncomputed defaults to measurement pending
        statusCounts.MEASUREMENT_PENDING++;
      }
    }

    const completedTasksCount = await prisma.maintenanceTask.count({
      where: { taskStatus: 'completed' },
    });

    return {
      totalIncidents: incidents.length,
      estimationStatusBreakdown: statusCounts,
      measuredIncidentsCount: estimatedCount,
      measurementPendingCount: statusCounts.MEASUREMENT_PENDING,
      rateSourceUnavailableCount: statusCounts.RATE_SOURCE_UNAVAILABLE,
      averageEstimatedCostINR: estimatedCount > 0 ? Number((totalCostSum / estimatedCount).toFixed(2)) : 0.0,
      regionalBreakdown: regionalCounts,
      supervisedRegressionReadiness: {
        status: 'DATA_INSUFFICIENT',
        currentCompletedRepairs: completedTasksCount,
        minimumRequiredRepairs: 50,
        canTrainRegression: false,
        prohibitionRationale:
          'Supervised cost and duration regression training is strictly locked: The database contains zero completed repairs with actual cost and duration ground truth. Training Linear, Random Forest, or Gradient Boosting regression on parametric formulas would produce circular hallucination without empirical validation.',
      },
      governanceNotice:
        'Repair estimates are parametric engineering approximations for municipal budgeting, not actual financial expenditures. Estimates must never be used as surrogate training labels for regression models.',
    };
  }

  /**
   * Returns all active regional rate schedules.
   */
  getAllRateSchedules(): RegionalRateSchedule[] {
    return [BENGALURU_RATE_SCHEDULE, DELHI_RATE_SCHEDULE];
  }
}


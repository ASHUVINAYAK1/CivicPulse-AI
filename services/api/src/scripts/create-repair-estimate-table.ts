import { prisma } from '../config/prisma';

async function main() {
  console.log('Creating repair_cost_estimates table if not exists...');

  await prisma.$executeRawUnsafe(`
    CREATE TABLE IF NOT EXISTS repair_cost_estimates (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      incident_id UUID NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
      task_id UUID REFERENCES maintenance_tasks(id) ON DELETE SET NULL,
      region VARCHAR(50) NOT NULL,
      rate_schedule_version VARCHAR(100) NOT NULL,
      estimation_status VARCHAR(50) NOT NULL,
      measured_area_sq_meters DOUBLE PRECISION,
      measurement_source VARCHAR(50) NOT NULL DEFAULT 'unmeasured_missing',
      depth_meters DOUBLE PRECISION,
      material_type VARCHAR(100),
      material_quantity DOUBLE PRECISION,
      material_unit VARCHAR(30),
      material_cost DOUBLE PRECISION,
      labor_hours DOUBLE PRECISION,
      labor_cost DOUBLE PRECISION,
      equipment_hours DOUBLE PRECISION,
      equipment_cost DOUBLE PRECISION,
      mobilization_cost DOUBLE PRECISION,
      total_estimated_cost DOUBLE PRECISION,
      currency VARCHAR(10) NOT NULL DEFAULT 'INR',
      estimated_duration_hours DOUBLE PRECISION,
      estimated_duration_days DOUBLE PRECISION,
      crew_size INTEGER,
      assumptions JSONB,
      breakdown JSONB,
      missing_information JSONB,
      computed_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
      is_actual_cost BOOLEAN NOT NULL DEFAULT FALSE
    )
  `);

  await prisma.$executeRawUnsafe(`CREATE INDEX IF NOT EXISTS idx_repair_est_incident ON repair_cost_estimates(incident_id)`);
  await prisma.$executeRawUnsafe(`CREATE INDEX IF NOT EXISTS idx_repair_est_status ON repair_cost_estimates(estimation_status)`);
  await prisma.$executeRawUnsafe(`CREATE INDEX IF NOT EXISTS idx_repair_est_region ON repair_cost_estimates(region)`);
  await prisma.$executeRawUnsafe(`CREATE INDEX IF NOT EXISTS idx_repair_est_computed ON repair_cost_estimates(computed_at)`);

  console.log('repair_cost_estimates table and indexes created successfully.');

  const count = await prisma.incident.count();
  console.log(`Verified incidents intact: ${count}`);

  await prisma.$disconnect();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});


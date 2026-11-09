import { prisma } from '../config/prisma';

async function main() {
  console.log('--- Migrating incident_forecasts schema ---');

  // Check existing columns
  const cols: any = await prisma.$queryRawUnsafe(
    "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'incident_forecasts';"
  );
  console.log('Current columns:', cols.map((c: any) => c.column_name));

  // Add provenance columns if they do not exist
  const existingCols = new Set(cols.map((c: any) => c.column_name));

  if (!existingCols.has('aggregation_frequency')) {
    console.log('Adding aggregation_frequency...');
    await prisma.$executeRawUnsafe(
      "ALTER TABLE incident_forecasts ADD COLUMN aggregation_frequency VARCHAR(30) DEFAULT 'monthly';"
    );
  }

  if (!existingCols.has('model_type')) {
    console.log('Adding model_type...');
    await prisma.$executeRawUnsafe(
      "ALTER TABLE incident_forecasts ADD COLUMN model_type VARCHAR(50) DEFAULT 'synthetic_seed';"
    );
  }

  if (!existingCols.has('readiness_status')) {
    console.log('Adding readiness_status...');
    await prisma.$executeRawUnsafe(
      "ALTER TABLE incident_forecasts ADD COLUMN readiness_status VARCHAR(50) DEFAULT 'DATA_INSUFFICIENT';"
    );
  }

  if (!existingCols.has('is_synthetic')) {
    console.log('Adding is_synthetic...');
    await prisma.$executeRawUnsafe(
      "ALTER TABLE incident_forecasts ADD COLUMN is_synthetic BOOLEAN DEFAULT false;"
    );
  }

  if (!existingCols.has('training_window_start')) {
    console.log('Adding training_window_start...');
    await prisma.$executeRawUnsafe(
      "ALTER TABLE incident_forecasts ADD COLUMN training_window_start TIMESTAMP(6) NULL;"
    );
  }

  if (!existingCols.has('training_window_end')) {
    console.log('Adding training_window_end...');
    await prisma.$executeRawUnsafe(
      "ALTER TABLE incident_forecasts ADD COLUMN training_window_end TIMESTAMP(6) NULL;"
    );
  }

  if (!existingCols.has('metadata')) {
    console.log('Adding metadata...');
    await prisma.$executeRawUnsafe(
      "ALTER TABLE incident_forecasts ADD COLUMN metadata JSONB NULL;"
    );
  }

  // Mark all 6 existing seed rows as synthetic demo
  console.log('Flagging existing seed forecasts as synthetic demo...');
  const updateResult = await prisma.$executeRawUnsafe(`
    UPDATE incident_forecasts
    SET is_synthetic = true,
        model_type = 'synthetic_seed',
        readiness_status = 'DATA_INSUFFICIENT',
        aggregation_frequency = 'monthly',
        metadata = '{"note": "Phase 2 synthetic demo seed record - excluded from real model training and evaluation", "source": "prisma/seed.ts"}'::jsonb
    WHERE is_synthetic IS NULL OR is_synthetic = false;
  `);
  console.log(`Updated ${updateResult} rows as synthetic demo records.`);

  // Verify
  const rows: any = await prisma.$queryRawUnsafe(
    'SELECT id, region_id, forecast_period, predicted_incidents, is_synthetic, model_type, readiness_status FROM incident_forecasts;'
  );
  console.log('Verified incident_forecasts rows:');
  console.table(rows);

  console.log('--- Migration completed successfully ---');
}

main()
  .catch((e) => {
    console.error('Migration failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });


import { prisma } from '../config/prisma';

async function auditDatabase() {
  console.log('===============================================================');
  console.log('CivicPulse AI — Phase 3, Stage 7: Database Consistency Audit');
  console.log('===============================================================\n');

  // 1. Table Census & Row Counts
  const tables = [
    'users',
    'incidents',
    'incident_images',
    'ai_predictions',
    'incident_verifications',
    'maintenance_tasks',
    'maintenance_updates',
    'maintenance_priority_recommendations',
    'repair_cost_estimates',
    'incident_forecasts',
    'model_registry',
    'audit_logs',
  ];

  console.log('--- 1. Table Census & Record Counts ---');
  for (const table of tables) {
    try {
      const res: any = await prisma.$queryRawUnsafe(`SELECT COUNT(*) as count FROM "${table}";`);
      const count = Number(res[0].count);
      console.log(`  â€¢ ${table.padEnd(38)}: ${count} records`);
    } catch (e: any) {
      console.log(`  â€¢ ${table.padEnd(38)}: ERROR (${e.message})`);
    }
  }

  // 2. Foreign Key & Referential Integrity Audit
  console.log('\n--- 2. Foreign Key & ON DELETE Actions ---');
  const fkQuery = `
    SELECT
      tc.table_name,
      kcu.column_name,
      ccu.table_name AS foreign_table_name,
      ccu.column_name AS foreign_column_name,
      rc.delete_rule
    FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
    JOIN information_schema.referential_constraints AS rc
      ON tc.constraint_name = rc.constraint_name
    JOIN information_schema.constraint_column_usage AS ccu
      ON rc.unique_constraint_name = ccu.constraint_name
      AND rc.constraint_schema = ccu.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
    ORDER BY tc.table_name, kcu.column_name;
  `;
  const fks: any = await prisma.$queryRawUnsafe(fkQuery);
  console.table(fks.map((fk: any) => ({
    table: fk.table_name,
    column: fk.column_name,
    references: `${fk.foreign_table_name}.${fk.foreign_column_name}`,
    on_delete: fk.delete_rule,
  })));

  // 3. PostGIS Geography & Geometry Columns
  console.log('\n--- 3. PostGIS Spatial Column Audit ---');
  const spatialQuery = `
    SELECT
      f_table_name as table_name,
      f_geography_column as column_name,
      srid,
      type
    FROM geography_columns;
  `;
  try {
    const spatialCols: any = await prisma.$queryRawUnsafe(spatialQuery);
    console.table(spatialCols);
  } catch (err: any) {
    console.log('Spatial columns query:', err.message);
  }

  // 4. Timestamp & Timezone Audit
  console.log('\n--- 4. Timestamp Types & UTC Verification ---');
  const tsQuery = `
    SELECT
      table_name,
      column_name,
      data_type
    FROM information_schema.columns
    WHERE table_name IN (${tables.map((t) => `'${t}'`).join(',')})
      AND data_type LIKE '%timestamp%'
    ORDER BY table_name, column_name;
  `;
  const tsCols: any = await prisma.$queryRawUnsafe(tsQuery);
  console.table(tsCols);

  // 5. Cost Separation Audit (Estimated vs Actual)
  console.log('\n--- 5. Cost & Duration Field Separation Audit ---');
  const costColsQuery = `
    SELECT
      table_name,
      column_name,
      data_type
    FROM information_schema.columns
    WHERE table_name IN ('repair_cost_estimates', 'maintenance_tasks')
      AND (column_name LIKE '%cost%' OR column_name LIKE '%duration%' OR column_name LIKE '%estimate%')
    ORDER BY table_name, column_name;
  `;
  const costCols: any = await prisma.$queryRawUnsafe(costColsQuery);
  console.table(costCols);

  // 6. Forecast Provenance & Synthetic Flag Audit
  console.log('\n--- 6. Forecast Synthetic vs Real Separation Audit ---');
  const forecastAuditQuery = `
    SELECT
      is_synthetic,
      model_type,
      readiness_status,
      COUNT(*) as count
    FROM incident_forecasts
    GROUP BY is_synthetic, model_type, readiness_status;
  `;
  const forecastAudit: any = await prisma.$queryRawUnsafe(forecastAuditQuery);
  console.table(forecastAudit.map((r: any) => ({
    is_synthetic: r.is_synthetic,
    model_type: r.model_type,
    readiness_status: r.readiness_status,
    count: Number(r.count),
  })));

  // 7. Orphan Checks (Referential Integrity Check)
  console.log('\n--- 7. Orphan Records & Integrity Check ---');
  const orphanChecks = [
    { name: 'Images without valid Incident', query: 'SELECT COUNT(*) as c FROM incident_images ii LEFT JOIN incidents i ON ii.incident_id = i.id WHERE i.id IS NULL;' },
    { name: 'Predictions without valid Incident', query: 'SELECT COUNT(*) as c FROM ai_predictions ap LEFT JOIN incidents i ON ap.incident_id = i.id WHERE i.id IS NULL;' },
    { name: 'Verifications without valid Incident', query: 'SELECT COUNT(*) as c FROM incident_verifications iv LEFT JOIN incidents i ON iv.incident_id = i.id WHERE i.id IS NULL;' },
    { name: 'Tasks without valid Incident', query: 'SELECT COUNT(*) as c FROM maintenance_tasks mt LEFT JOIN incidents i ON mt.incident_id = i.id WHERE i.id IS NULL;' },
    { name: 'Priority Recs without valid Incident', query: 'SELECT COUNT(*) as c FROM maintenance_priority_recommendations mpr LEFT JOIN incidents i ON mpr.incident_id = i.id WHERE i.id IS NULL;' },
    { name: 'Repair Estimates without valid Incident', query: 'SELECT COUNT(*) as c FROM repair_cost_estimates rce LEFT JOIN incidents i ON rce.incident_id = i.id WHERE i.id IS NULL;' },
    { name: 'Duplicate Incidents without Canonical Parent', query: "SELECT COUNT(*) as c FROM incidents WHERE duplicate_status = 'duplicate' AND canonical_incident_id IS NULL;" },
  ];

  for (const check of orphanChecks) {
    const res: any = await prisma.$queryRawUnsafe(check.query);
    const count = Number(res[0].c);
    const status = count === 0 ? 'âœ“ ZERO ORPHANS (CLEAN)' : `âœ— ${count} ORPHANS FOUND!`;
    console.log(`  â€¢ ${check.name.padEnd(45)}: ${status}`);
  }

  console.log('\n===============================================================');
  console.log('Database Consistency Audit Complete');
  console.log('===============================================================\n');
}

auditDatabase()
  .catch(console.error)
  .finally(async () => {
    await prisma.$disconnect();
  });


import { prisma } from '../config/prisma';

async function main() {
  console.log('Checking database tables and columns...');

  // Inspect columns of incidents
  const cols: any = await prisma.$queryRaw`
    SELECT column_name, data_type, udt_name 
    FROM information_schema.columns 
    WHERE table_name = 'incidents'
  `;
  console.log('Incident columns:', cols.map((c: any) => `${c.column_name} (${c.udt_name})`));

  // Create maintenance_priority_recommendations table safely without dropping any PostGIS column
  console.log('Creating maintenance_priority_recommendations table if not exists...');
  await prisma.$executeRawUnsafe(`
    CREATE TABLE IF NOT EXISTS maintenance_priority_recommendations (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      incident_id UUID NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
      recommended_priority VARCHAR(30) NOT NULL,
      priority_score DOUBLE PRECISION NOT NULL,
      factor_scores JSONB NOT NULL,
      weights_used JSONB NOT NULL,
      contributing_evidence JSONB NOT NULL,
      missing_information JSONB,
      explanation TEXT NOT NULL,
      scoring_rule_version VARCHAR(50) NOT NULL,
      computed_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
      review_status VARCHAR(40) NOT NULL DEFAULT 'pending',
      reviewed_priority VARCHAR(30),
      reviewed_by UUID REFERENCES users(id) ON DELETE SET NULL,
      reviewed_at TIMESTAMP(6),
      override_reason TEXT
    )
  `);

  await prisma.$executeRawUnsafe(`CREATE INDEX IF NOT EXISTS idx_maint_prio_incident ON maintenance_priority_recommendations(incident_id)`);
  await prisma.$executeRawUnsafe(`CREATE INDEX IF NOT EXISTS idx_maint_prio_rec_prio ON maintenance_priority_recommendations(recommended_priority)`);
  await prisma.$executeRawUnsafe(`CREATE INDEX IF NOT EXISTS idx_maint_prio_rev_status ON maintenance_priority_recommendations(review_status)`);
  await prisma.$executeRawUnsafe(`CREATE INDEX IF NOT EXISTS idx_maint_prio_computed ON maintenance_priority_recommendations(computed_at)`);

  console.log('Table and indexes created successfully.');

  // Verify record count on incidents
  const count = await prisma.incident.count();
  console.log(`Verified total incidents intact: ${count}`);

  await prisma.$disconnect();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

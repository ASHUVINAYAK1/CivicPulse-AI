import { prisma } from '../config/prisma';
import * as fs from 'fs';
import * as path from 'path';

async function main() {
  const incidents = await prisma.incident.findMany({
    select: {
      id: true,
      incidentNumber: true,
      title: true,
      damageType: true,
      severity: true,
      status: true,
      priority: true,
      latitude: true,
      longitude: true,
      address: true,
      reportedAt: true,
      verifiedAt: true,
      duplicateStatus: true,
    },
    orderBy: { reportedAt: 'asc' },
  });

  const formatted = incidents.map((inc) => ({
    id: inc.id,
    incidentNumber: inc.incidentNumber,
    title: inc.title,
    damageType: inc.damageType,
    severity: inc.severity,
    status: inc.status,
    priority: inc.priority,
    latitude: Number(inc.latitude),
    longitude: Number(inc.longitude),
    address: inc.address,
    reportedAt: inc.reportedAt.toISOString(),
    verifiedAt: inc.verifiedAt ? inc.verifiedAt.toISOString() : null,
    duplicateStatus: inc.duplicateStatus,
  }));

  const outDir = path.resolve(__dirname, '../../../ml/model_artifacts');
  const outFile = path.join(outDir, 'incident_coordinates.json');
  fs.writeFileSync(outFile, JSON.stringify(formatted, null, 2));
  console.log(`Exported ${formatted.length} real incident coordinates to ${outFile}`);
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());


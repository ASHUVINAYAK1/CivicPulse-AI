import { prisma } from '../config/prisma';
import * as fs from 'fs';
import * as path from 'path';

async function main() {
  const models = await prisma.modelRegistry.findMany({
    orderBy: { createdAt: 'desc' },
  });

  const outPath = path.resolve(__dirname, '../../../ml/model_artifacts/db_models_audit.json');
  fs.writeFileSync(outPath, JSON.stringify(models, null, 2));
  console.log(`Exported ${models.length} database model_registry records to ${outPath}`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });


import { prisma } from '../config/prisma';

async function main() {
  const result: any[] = await prisma.$queryRaw`
    SELECT 
      count(*) as total,
      count(location) as with_location,
      count(latitude) as with_lat,
      count(longitude) as with_lng
    FROM incidents;
  `;
  console.log('Location status:', result);

  const sample: any[] = await prisma.$queryRaw`
    SELECT id, incident_number, latitude, longitude, ST_AsText(location) as loc_text
    FROM incidents
    LIMIT 5;
  `;
  console.log('Sample incidents:', sample);
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());

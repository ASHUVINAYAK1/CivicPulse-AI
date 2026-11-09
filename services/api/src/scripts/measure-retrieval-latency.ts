import { prisma } from '../config/prisma';
import { findDuplicateCandidates } from '../modules/deduplication/service';

async function measureRetrievalLatency() {
  console.log('Measuring PostGIS candidate retrieval latency over 100 queries...');

  // Pick an incident with location in DB
  const incident = await prisma.incident.findFirst();

  if (!incident) {
    console.error('No incident found in DB with coordinates.');
    process.exit(1);
  }

  console.log(`Using incident ID: ${incident.id} (${incident.incidentNumber}) at Lat: ${incident.latitude}, Lng: ${incident.longitude}`);

  // Warmup 5 queries
  for (let i = 0; i < 5; i++) {
    await findDuplicateCandidates(incident.id, { radiusMeters: 10.0, timeWindowDays: 30 });
  }

  // Measure 100 iterations
  const latenciesMs: number[] = [];
  const N = 100;

  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    await findDuplicateCandidates(incident.id, { radiusMeters: 10.0, timeWindowDays: 30 });
    const elapsed = performance.now() - t0;
    latenciesMs.push(elapsed);
  }

  latenciesMs.sort((a, b) => a - b);
  const min = latenciesMs[0];
  const max = latenciesMs[latenciesMs.length - 1];
  const mean = latenciesMs.reduce((a, b) => a + b, 0) / N;
  const p50 = latenciesMs[Math.floor(N * 0.50)];
  const p95 = latenciesMs[Math.floor(N * 0.95)];
  const p99 = latenciesMs[Math.floor(N * 0.99)];

  console.log('\n--- PostGIS Candidate Retrieval Latency Results (N=100) ---');
  console.log(`  Min:  ${min.toFixed(2)} ms`);
  console.log(`  Mean: ${mean.toFixed(2)} ms`);
  console.log(`  p50:  ${p50.toFixed(2)} ms`);
  console.log(`  p95:  ${p95.toFixed(2)} ms`);
  console.log(`  p99:  ${p99.toFixed(2)} ms`);
  console.log(`  Max:  ${max.toFixed(2)} ms`);

  await prisma.$disconnect();
}

measureRetrievalLatency().catch((err) => {
  console.error('Latency measurement failed:', err);
  process.exit(1);
});

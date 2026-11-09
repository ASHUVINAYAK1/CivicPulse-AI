/**
 * CivicPulse AI — Stage 5: Geographical Hotspot Detection Test & Verification Suite
 * Tests mathematical functions, spherical projections, K-Means, DBSCAN,
 * PostGIS spatial queries, and Express API endpoints.
 */

import http from 'http';
import { prisma } from '../config/prisma';
import { createApp } from '../app';
import {
  haversineDistance,
  latLonToCartesian,
  cartesianToLatLon,
  validateCoordinates,
} from '../modules/hotspots/geo-utils';
import {
  preprocessSpatialIncidents,
  RawSpatialIncident,
} from '../modules/hotspots/preprocessor';
import {
  SphericalKMeansEngine,
} from '../modules/hotspots/kmeans';
import {
  GeographicDBSCANEngine,
} from '../modules/hotspots/dbscan';
import {
  getIncidentsWithinRadius,
  calculateSpatialDensity,
  getNearbyIncidentsForIncident,
  runPostGISNativeDBSCAN,
} from '../modules/hotspots/postgis';

async function runHotspotTestSuite() {
  console.log('\n===============================================================');
  console.log('  CivicPulse AI — Phase 3, Stage 5: Hotspot Detection Suite');
  console.log('===============================================================\n');

  let passedTests = 0;
  let failedTests = 0;

  function assert(condition: boolean, testName: string, detail: string = '') {
    if (condition) {
      console.log(`  [PASS] ${testName} ${detail ? `(${detail})` : ''}`);
      passedTests++;
    } else {
      console.error(`  [FAIL] ${testName} - FAILED! ${detail}`);
      failedTests++;
    }
  }

  // -----------------------------------------------------------------
  // 1. Math & Haversine Distance Tests against known geographic baselines
  // -----------------------------------------------------------------
  console.log('--- TEST GROUP 1: Spherical Mathematics & Haversine ---');

  // Ground truth: India Gate (28.6129° N, 77.2295° E) to Qutub Minar (28.5244° N, 77.1855° E)
  // Great-circle distance is ~10.74 km
  const distDelhi = haversineDistance(28.6129, 77.2295, 28.5244, 77.1855);
  assert(
    distDelhi >= 10500 && distDelhi <= 11000,
    'India Gate to Qutub Minar Distance',
    `Expected ~10.7 km, got ${(distDelhi / 1000).toFixed(2)} km`
  );

  // Ground truth: Delhi (28.6139, 77.2090) to Mumbai (19.0760, 72.8777) ~ 1148 km
  const distDelhiMumbai = haversineDistance(28.6139, 77.2090, 19.0760, 72.8777);
  assert(
    distDelhiMumbai >= 1140000 && distDelhiMumbai <= 1160000,
    'Delhi to Mumbai Distance',
    `Expected ~1,148 km, got ${(distDelhiMumbai / 1000).toFixed(1)} km`
  );

  // Distance to self is 0
  const zeroDist = haversineDistance(28.6139, 77.2090, 28.6139, 77.2090);
  assert(zeroDist === 0, 'Zero distance to self', `Got ${zeroDist}m`);

  // Equator 1-degree longitude distance ~ 111.19 km
  const distEquator = haversineDistance(0, 0, 0, 1);
  assert(
    distEquator >= 111000 && distEquator <= 112000,
    'Equatorial 1-degree Longitude Distance',
    `Expected ~111.2 km, got ${(distEquator / 1000).toFixed(2)} km`
  );

  // -----------------------------------------------------------------
  // 2. Unit Sphere Projections & Inverses
  // -----------------------------------------------------------------
  console.log('\n--- TEST GROUP 2: 3D Unit Sphere Transformations ---');

  const testCoords = [
    { latitude: 28.6139, longitude: 77.2090 }, // Delhi
    { latitude: 19.0760, longitude: 72.8777 }, // Mumbai
    { latitude: 0.0, longitude: 0.0 },         // Prime Meridian / Equator
    { latitude: 89.9, longitude: 45.0 },       // Near North Pole
    { latitude: -45.0, longitude: -120.0 },    // Southern hemisphere
  ];

  for (const tc of testCoords) {
    const cart = latLonToCartesian(tc.latitude, tc.longitude);
    const norm = Math.sqrt(cart.x * cart.x + cart.y * cart.y + cart.z * cart.z);
    assert(
      Math.abs(norm - 1.0) < 1e-6,
      `Unit sphere vector norm for (${tc.latitude}, ${tc.longitude})`,
      `Norm=${norm.toFixed(6)}`
    );

    const inverted = cartesianToLatLon(cart);
    const latDiff = Math.abs(inverted.latitude - tc.latitude);
    const lonDiff = Math.abs(inverted.longitude - tc.longitude);
    assert(
      latDiff < 1e-4 && lonDiff < 1e-4,
      `Roundtrip inversion for (${tc.latitude}, ${tc.longitude})`,
      `Inverted to (${inverted.latitude.toFixed(4)}, ${inverted.longitude.toFixed(4)})`
    );
  }

  // -----------------------------------------------------------------
  // 3. Coordinate Validation & Preprocessing
  // -----------------------------------------------------------------
  console.log('\n--- TEST GROUP 3: Coordinate Validation & Preprocessing ---');

  assert(validateCoordinates(28.6, 77.2).valid, 'Valid Delhi coordinate accepted');
  assert(!validateCoordinates(95.0, 77.2).valid, 'Invalid Latitude (>90) rejected');
  assert(!validateCoordinates(-95.0, 77.2).valid, 'Invalid Latitude (<-90) rejected');
  assert(!validateCoordinates(28.6, 195.0).valid, 'Invalid Longitude (>180) rejected');
  assert(!validateCoordinates(NaN, 77.2).valid, 'NaN latitude rejected');
  assert(!validateCoordinates(28.6, Infinity).valid, 'Infinite longitude rejected');
  assert(!validateCoordinates(null as any, 77.2).valid, 'Null latitude rejected');

  // Test duplicate coordinate detection and filtering
  const testIncidentsRaw: RawSpatialIncident[] = [
    {
      id: 'test-1',
      incidentNumber: 'T-001',
      title: 'Pothole A',
      damageType: 'pothole',
      severity: 'critical',
      status: 'submitted',
      priority: 'high',
      latitude: 28.6139,
      longitude: 77.2090,
      address: 'Road 1',
      reportedAt: '2026-11-07T12:00:00Z',
      verifiedAt: null,
    },
    {
      id: 'test-2',
      incidentNumber: 'T-002',
      title: 'Pothole Duplicate Coord',
      damageType: 'pothole',
      severity: 'critical',
      status: 'submitted',
      priority: 'high',
      latitude: 28.6139,
      longitude: 77.2090, // Exactly identical coords
      address: 'Road 1',
      reportedAt: '2026-11-07T12:10:00Z',
      verifiedAt: null,
    },
    {
      id: 'test-3',
      incidentNumber: 'T-003',
      title: 'Crack B',
      damageType: 'alligator_crack',
      severity: 'high',
      status: 'verified',
      priority: 'medium',
      latitude: 28.6250,
      longitude: 77.2150,
      address: 'Road 2',
      reportedAt: '2026-11-07T13:00:00Z',
      verifiedAt: '2026-11-07T13:30:00Z',
    },
    {
      id: 'test-4-invalid',
      incidentNumber: 'T-004',
      title: 'Malformed Point',
      damageType: 'pothole',
      severity: null,
      status: 'submitted',
      priority: null,
      latitude: 120.0, // Out of range
      longitude: 77.2,
      address: 'Nowhere',
      reportedAt: '2026-11-07T14:00:00Z',
      verifiedAt: null,
    },
  ];

  const prep = preprocessSpatialIncidents(testIncidentsRaw);
  assert(prep.totalValid === 3, 'Valid points count', `Expected 3, got ${prep.totalValid}`);
  assert(prep.totalRejected === 1, 'Rejected count', `Expected 1, got ${prep.totalRejected}`);
  assert(prep.duplicateCoordinatePairs === 1, 'Duplicate coordinate pair identified');
  assert(prep.incidents[0].isCoordinateDuplicate, 'First duplicate flagged');
  assert(prep.incidents[1].isCoordinateDuplicate, 'Second duplicate flagged');
  assert(!prep.incidents[2].isCoordinateDuplicate, 'Third non-duplicate clean');

  // Filter criteria test
  const filteredPrep = preprocessSpatialIncidents(testIncidentsRaw, {
    verificationStatus: 'verified',
  });
  assert(
    filteredPrep.totalValid === 1 && filteredPrep.incidents[0].damageType === 'alligator_crack',
    'Verification status filter',
    `Got ${filteredPrep.totalValid} verified incidents`
  );

  // -----------------------------------------------------------------
  // 4. Spherical K-Means from Scratch
  // -----------------------------------------------------------------
  console.log('\n--- TEST GROUP 4: Spherical K-Means from Scratch ---');

  // Fetch real incidents from database
  const realDbIncidents = await prisma.incident.findMany({
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
  });

  const realPrep = preprocessSpatialIncidents(
    realDbIncidents.map((r) => ({
      ...r,
      latitude: Number(r.latitude),
      longitude: Number(r.longitude),
    }))
  );
  assert(realPrep.totalValid === 66, 'Real DB incident load', `Loaded ${realPrep.totalValid} incidents`);

  // Run K-Means K=4
  const startKMeans = Date.now();
  const kmEngine = new SphericalKMeansEngine(4, 100, 1.0, 42);
  const kmResult = kmEngine.fit(realPrep.incidents);
  const kmDuration = Date.now() - startKMeans;

  assert(kmResult.converged, 'K-Means K=4 converged', `Iterations=${kmResult.iterations}`);
  assert(kmResult.clusters.length === 4, 'K-Means created 4 clusters');
  assert(kmResult.inertiaSqMeters > 0, 'K-Means positive inertia', `Inertia=${kmResult.inertiaSqMeters.toLocaleString()} m^2`);
  assert(kmResult.silhouetteScore > 0.80, 'K-Means high silhouette score', `Score=${kmResult.silhouetteScore}`);
  assert(kmDuration < 100, 'K-Means performance latency', `Completed in ${kmDuration}ms (<100ms)`);

  // Test Seed Reproducibility
  const kmEngine2 = new SphericalKMeansEngine(4, 100, 1.0, 42);
  const kmResult2 = kmEngine2.fit(realPrep.incidents);
  assert(
    kmResult.inertiaSqMeters === kmResult2.inertiaSqMeters,
    'Deterministic Seed Reproducibility',
    `Both runs yielded identical inertia: ${kmResult.inertiaSqMeters}`
  );

  // Multi-K range evaluation
  const rangeEval = SphericalKMeansEngine.evaluateRange(realPrep.incidents, 2, 6, 42);
  assert(rangeEval.length === 5, 'Multi-K Range Evaluation count (K=2..6)');
  assert(rangeEval[0].inertiaSqMeters > rangeEval[rangeEval.length - 1].inertiaSqMeters, 'Inertia decreases monotonically with increasing K');

  // -----------------------------------------------------------------
  // 5. Geographic DBSCAN with Haversine
  // -----------------------------------------------------------------
  console.log('\n--- TEST GROUP 5: Geographic DBSCAN with Haversine ---');

  const startDBSCAN = Date.now();
  const dbEngine = new GeographicDBSCANEngine(500.0, 3);
  const dbResult = dbEngine.fit(realPrep.incidents);
  const dbDuration = Date.now() - startDBSCAN;

  assert(dbResult.clusterCount >= 1, 'DBSCAN formed clusters', `Clusters=${dbResult.clusterCount}`);
  assert(dbResult.noiseCount > 0, 'DBSCAN identified isolated noise points', `Noise=${dbResult.noiseCount}`);
  assert(dbResult.diagnostics.length === 66, 'DBSCAN diagnostics for every point');
  assert(dbDuration < 100, 'DBSCAN performance latency', `Completed in ${dbDuration}ms (<100ms)`);

  // Check noise diagnostic reason format
  const sampleNoise = dbResult.diagnostics.find((d) => d.pointType === 'noise');
  assert(
    !!sampleNoise && sampleNoise.classificationReason.includes('Noise / isolated incident'),
    'DBSCAN explainable noise diagnostic rationale',
    sampleNoise ? sampleNoise.classificationReason : ''
  );

  const sampleCore = dbResult.diagnostics.find((d) => d.pointType === 'core');
  assert(
    !!sampleCore && sampleCore.classificationReason.includes('Core incident'),
    'DBSCAN explainable core point diagnostic rationale',
    sampleCore ? sampleCore.classificationReason : ''
  );

  // -----------------------------------------------------------------
  // 6. PostGIS Spatial Radius, Density & Native DBSCAN
  // -----------------------------------------------------------------
  console.log('\n--- TEST GROUP 6: PostGIS Spatial Database Integration ---');

  // Query incidents around Delhi center (28.6139, 77.2090) within 15 km
  const postgisRadius = await getIncidentsWithinRadius(28.6139, 77.2090, 15000, 20);
  assert(postgisRadius.length > 0, 'PostGIS ST_DWithin radius query returned incidents', `Found ${postgisRadius.length} incidents`);
  assert(postgisRadius[0].distanceMeters <= postgisRadius[postgisRadius.length - 1].distanceMeters, 'PostGIS ordered by distance ASC');

  // Spatial density
  const density = await calculateSpatialDensity(28.6139, 77.2090, 5000);
  assert(density.incidentCount > 0, 'Spatial density calculation', `${density.incidentCount} incidents, ${density.densityIncidentsPerKm2} /km^2`);

  // Nearby for first incident
  const firstId = realDbIncidents[0].id;
  const nearbyData = await getNearbyIncidentsForIncident(firstId, 5000, 5);
  assert(nearbyData.targetIncident !== null, 'Target incident retrieved by ID');
  assert(nearbyData.nearby.length > 0, 'Nearby incidents retrieved using PostGIS');

  // PostGIS Native ST_ClusterDBSCAN
  const postgisNative = await runPostGISNativeDBSCAN(1000, 3);
  assert(postgisNative.totalPoints === 66, 'PostGIS Native ST_ClusterDBSCAN executed on 66 rows');
  assert(postgisNative.clustersCount >= 1, 'PostGIS Native identified clusters', `Clusters=${postgisNative.clustersCount}, Noise=${postgisNative.noiseCount}`);

  // -----------------------------------------------------------------
  // 7. Express API Endpoints Integration Test
  // -----------------------------------------------------------------
  console.log('\n--- TEST GROUP 7: Express API Endpoints Integration ---');

  const app = createApp();
  const PORT = 5057;
  const server = app.listen(PORT, '127.0.0.1');

  function httpGet(path: string): Promise<{ status: number; body: any }> {
    return new Promise((resolve, reject) => {
      http
        .get(`http://127.0.0.1:${PORT}${path}`, (res) => {
          let rawData = '';
          res.on('data', (chunk) => (rawData += chunk));
          res.on('end', () => {
            try {
              const body = JSON.parse(rawData);
              resolve({ status: res.statusCode || 500, body });
            } catch (e) {
              resolve({ status: res.statusCode || 500, body: rawData });
            }
          });
        })
        .on('error', reject);
    });
  }

  try {
    // 7.1 GET /api/v1/hotspots/summary
    const resSummary = await httpGet('/api/v1/hotspots/summary');
    assert(resSummary.status === 200, 'GET /api/v1/hotspots/summary returned 200');
    assert(resSummary.body.data.isExploratory === true, 'Summary marked isExploratory=true');
    assert(resSummary.body.data.disclaimer.includes('Clusters describe the spatial distribution'), 'Mandatory disclaimer present');
    assert(resSummary.body.data.validIncidents === 66, 'Summary reported 66 valid incidents');

    // 7.2 GET /api/v1/hotspots/kmeans?k=4
    const resKMeans = await httpGet('/api/v1/hotspots/kmeans?k=4&evaluateRange=true');
    assert(resKMeans.status === 200, 'GET /api/v1/hotspots/kmeans returned 200');
    assert(resKMeans.body.data.result.k === 4, 'K-Means returned 4 clusters');
    assert(Array.isArray(resKMeans.body.data.rangeEvaluation), 'K-Means range evaluation included');

    // 7.3 GET /api/v1/hotspots/dbscan?eps=500&minSamples=3
    const resDBSCAN = await httpGet('/api/v1/hotspots/dbscan?eps=500&minSamples=3');
    assert(resDBSCAN.status === 200, 'GET /api/v1/hotspots/dbscan returned 200');
    assert(resDBSCAN.body.data.result.clusterCount >= 1, 'DBSCAN API returned clusters');
    assert(resDBSCAN.body.data.result.diagnostics.length === 66, 'DBSCAN API returned diagnostics');

    // 7.4 GET /api/v1/hotspots/parameters
    const resParams = await httpGet('/api/v1/hotspots/parameters');
    assert(resParams.status === 200, 'GET /api/v1/hotspots/parameters returned 200');
    assert(resParams.body.data.kmeans.defaultK === 4, 'Parameters API default K=4');
    assert(resParams.body.data.dbscan.defaultEpsMeters === 500, 'Parameters API default eps=500m');

    // 7.5 GET /api/v1/hotspots/incidents/:id/nearby
    const resNearby = await httpGet(`/api/v1/hotspots/incidents/${firstId}/nearby?radius=5000`);
    assert(resNearby.status === 200, 'GET /api/v1/hotspots/incidents/:id/nearby returned 200');
    assert(resNearby.body.data.targetIncident.id === firstId, 'Nearby target incident matched');

    // 7.6 Error handling: invalid K
    const resBadK = await httpGet('/api/v1/hotspots/kmeans?k=100');
    assert(resBadK.status === 400, 'Invalid K (>20) rejected with 400 Bad Request');

    // 7.7 Error handling: negative eps
    const resBadEps = await httpGet('/api/v1/hotspots/dbscan?eps=-50');
    assert(resBadEps.status === 400, 'Negative eps rejected with 400 Bad Request');
  } finally {
    server.close();
  }

  // -----------------------------------------------------------------
  // SUMMARY
  // -----------------------------------------------------------------
  console.log('\n===============================================================');
  console.log(`  HOTSPOT DETECTION TEST SUMMARY: ${passedTests} PASSED, ${failedTests} FAILED`);
  console.log('===============================================================\n');

  if (failedTests > 0) {
    process.exit(1);
  }
}

runHotspotTestSuite()
  .catch((e) => {
    console.error('Test suite failed with unexpected error:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

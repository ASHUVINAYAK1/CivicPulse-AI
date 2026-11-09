import { prisma } from '../../config/prisma';
import { computeHammingDistance, computeHashSimilarity } from './hasher';
import { scoreDuplicateCandidate, DuplicateScoreResult } from './scorer';
import { getImageUrl } from '../../config/storage';

export interface CandidateResult {
  candidateIncident: {
    id: string;
    incidentNumber: string;
    title: string;
    description: string;
    damageType: string;
    severity: string | null;
    status: string;
    latitude: number;
    longitude: number;
    address: string | null;
    reportedAt: Date;
    distanceMeters: number;
    images: {
      id: string;
      url: string;
      originalFilename: string;
      dHash: string | null;
      perceptualHash: string | null;
    }[];
  };
  scoring: DuplicateScoreResult;
}

export interface DuplicateDetectionOptions {
  radiusMeters?: number;
  timeWindowDays?: number;
}

/**
 * Searches for duplicate incident candidates within a geographic radius and time window.
 * Computes PostGIS spatial distance and perceptual image hash similarity.
 */
export async function findDuplicateCandidates(
  incidentId: string,
  options: DuplicateDetectionOptions = {}
): Promise<{ targetIncident: any; candidates: CandidateResult[] }> {
  const radiusMeters = options.radiusMeters || 10.0;
  const timeWindowDays = options.timeWindowDays || 30;

  // 1. Fetch Target Incident with images
  const targetIncident = await prisma.incident.findUnique({
    where: { id: incidentId },
    include: {
      images: true,
      predictions: { take: 1, orderBy: { createdAt: 'desc' } },
    },
  });

  if (!targetIncident) {
    throw new Error(`Incident not found: ${incidentId}`);
  }

  const targetLat = Number(targetIncident.latitude);
  const targetLng = Number(targetIncident.longitude);
  const targetReportedAt = new Date(targetIncident.reportedAt);

  const windowStartDate = new Date(targetReportedAt.getTime() - timeWindowDays * 24 * 60 * 60 * 1000);
  const windowEndDate = new Date(targetReportedAt.getTime() + timeWindowDays * 24 * 60 * 60 * 1000);

  // 2. Query PostGIS for nearby incidents within radius & time window
  // Uses ST_DWithin and ST_Distance on geography(Point, 4326) column
  const nearbyIncidentsRaw = await prisma.$queryRaw<
    Array<{
      id: string;
      incident_number: string;
      title: string;
      description: string;
      damage_type: string;
      severity: string | null;
      status: string;
      latitude: any;
      longitude: any;
      address: string | null;
      reported_at: Date;
      distance_meters: number;
    }>
  >`
    SELECT id, incident_number, title, description, damage_type, severity, status,
           latitude, longitude, address, reported_at,
           ST_Distance(location, ST_SetSRID(ST_MakePoint(${targetLng}, ${targetLat}), 4326)::geography) AS distance_meters
    FROM incidents
    WHERE id != ${incidentId}::uuid
      AND ST_DWithin(location, ST_SetSRID(ST_MakePoint(${targetLng}, ${targetLat}), 4326)::geography, ${radiusMeters})
      AND reported_at >= ${windowStartDate}
      AND reported_at <= ${windowEndDate}
      AND duplicate_status != 'confirmed_duplicate'
    ORDER BY distance_meters ASC
    LIMIT 25;
  `;

  if (!nearbyIncidentsRaw || nearbyIncidentsRaw.length === 0) {
    return { targetIncident, candidates: [] };
  }

  // 3. Score each nearby candidate
  const candidates: CandidateResult[] = [];

  for (const nearby of nearbyIncidentsRaw) {
    // Fetch candidate images
    const candidateImages = await prisma.incidentImage.findMany({
      where: { incidentId: nearby.id },
    });

    // Compare perceptual image hashes
    let bestImageSimilarity = 0.5;
    let bestHamming = 64;
    let hashTypeUsed: 'dHash' | 'pHash' | 'none' = 'none';

    for (const tImg of targetIncident.images) {
      for (const cImg of candidateImages) {
        // Compare dHash first
        if (tImg.dHash && cImg.dHash) {
          const dist = computeHammingDistance(tImg.dHash, cImg.dHash);
          const sim = computeHashSimilarity(tImg.dHash, cImg.dHash);
          if (sim > bestImageSimilarity || hashTypeUsed === 'none') {
            bestImageSimilarity = sim;
            bestHamming = dist;
            hashTypeUsed = 'dHash';
          }
        }
        // Also compare pHash
        if (tImg.perceptualHash && cImg.perceptualHash) {
          const dist = computeHammingDistance(tImg.perceptualHash, cImg.perceptualHash);
          const sim = computeHashSimilarity(tImg.perceptualHash, cImg.perceptualHash);
          if (sim > bestImageSimilarity) {
            bestImageSimilarity = sim;
            bestHamming = dist;
            hashTypeUsed = 'pHash';
          }
        }
      }
    }

    const timeDeltaHours = Math.abs(targetReportedAt.getTime() - new Date(nearby.reported_at).getTime()) / (1000 * 3600);

    const scoring = scoreDuplicateCandidate({
      distanceMeters: nearby.distance_meters,
      maxRadiusMeters: radiusMeters,
      timeDeltaHours,
      maxTimeWindowDays: timeWindowDays,
      targetDamageType: targetIncident.damageType,
      candidateDamageType: nearby.damage_type,
      bestImageSimilarity,
      bestHammingDistance: bestHamming,
      hashTypeUsed,
    });

    // Attach image URLs for inspector review
    const candidateImagesWithUrls = await Promise.all(
      candidateImages.map(async (img) => {
        let url = `/api/v1/images/${img.id}/raw`;
        try {
          url = await getImageUrl(img.storageKey);
        } catch {
          // fallback
        }
        return {
          id: img.id,
          url,
          originalFilename: img.originalFilename,
          dHash: img.dHash,
          perceptualHash: img.perceptualHash,
        };
      })
    );

    candidates.push({
      candidateIncident: {
        id: nearby.id,
        incidentNumber: nearby.incident_number,
        title: nearby.title,
        description: nearby.description,
        damageType: nearby.damage_type,
        severity: nearby.severity,
        status: nearby.status,
        latitude: Number(nearby.latitude),
        longitude: Number(nearby.longitude),
        address: nearby.address,
        reportedAt: nearby.reported_at,
        distanceMeters: Math.round(nearby.distance_meters * 10) / 10,
        images: candidateImagesWithUrls,
      },
      scoring,
    });
  }

  // Sort candidates by composite confidence score descending
  candidates.sort((a, b) => b.scoring.confidenceScore - a.scoring.confidenceScore);

  return {
    targetIncident: {
      ...targetIncident,
      images: targetIncident.images.map((img) => ({
        ...img,
        fileSize: img.fileSize.toString(),
      })),
    },
    candidates,
  };
}

/**
 * Inspector workflow: confirms a report as a duplicate and links it to a canonical incident.
 * NEVER deletes or alters citizen submission data.
 */
export async function confirmDuplicate(
  incidentId: string,
  canonicalIncidentId: string,
  inspectorId: string,
  remarks?: string
) {
  if (incidentId === canonicalIncidentId) {
    throw new Error('An incident cannot be marked as a duplicate of itself.');
  }

  // 1. Fetch both incidents
  const [incident, targetCanonical] = await Promise.all([
    prisma.incident.findUnique({ where: { id: incidentId } }),
    prisma.incident.findUnique({ where: { id: canonicalIncidentId } }),
  ]);

  if (!incident) throw new Error(`Incident not found: ${incidentId}`);
  if (!targetCanonical) throw new Error(`Canonical incident not found: ${canonicalIncidentId}`);

  // 2. Prevent direct or indirect circular relationships
  // Check if targetCanonical points back to incidentId directly
  if (targetCanonical.canonicalIncidentId === incidentId) {
    throw new Error('Circular duplicate relationship detected: candidate incident is already linked to this incident.');
  }

  // 3. Resolve canonical root to prevent multi-hop duplicate chains
  let rootCanonicalId = canonicalIncidentId;
  let currentCanonical = targetCanonical;
  const visited = new Set<string>([incidentId, canonicalIncidentId]);

  while (currentCanonical.canonicalIncidentId) {
    if (currentCanonical.canonicalIncidentId === incidentId) {
      throw new Error('Circular duplicate chain detected: traversal leads back to target incident.');
    }
    if (visited.has(currentCanonical.canonicalIncidentId)) {
      throw new Error('Cyclic relationship detected in canonical hierarchy.');
    }
    visited.add(currentCanonical.canonicalIncidentId);

    const parent = await prisma.incident.findUnique({
      where: { id: currentCanonical.canonicalIncidentId },
    });
    if (!parent) break;
    rootCanonicalId = parent.id;
    currentCanonical = parent;
  }

  // 4. Atomic transaction: update incident status, reroute any existing children, and record audit log
  return await prisma.$transaction(async (tx) => {
    // Update the duplicate incident
    const updatedIncident = await tx.incident.update({
      where: { id: incidentId },
      data: {
        canonicalIncidentId: rootCanonicalId,
        duplicateStatus: 'confirmed_duplicate',
        status: 'duplicate',
        updatedAt: new Date(),
      },
    });

    // If this incident previously had other incidents pointing to it, reroute them to rootCanonicalId
    await tx.incident.updateMany({
      where: { canonicalIncidentId: incidentId },
      data: {
        canonicalIncidentId: rootCanonicalId,
        updatedAt: new Date(),
      },
    });

    // Record immutable AuditLog entry within transaction
    await tx.auditLog.create({
      data: {
        userId: inspectorId,
        entityType: 'incident',
        entityId: incidentId,
        action: 'incident_duplicate_confirmed',
        metadata: {
          canonicalIncidentId: rootCanonicalId,
          requestedCanonicalId: canonicalIncidentId,
          canonicalIncidentNumber: currentCanonical.incidentNumber,
          originalIncidentNumber: incident.incidentNumber,
          inspectorRemarks: remarks || 'Confirmed physical duplicate by municipal inspector.',
          confirmedAt: new Date().toISOString(),
        },
      },
    });

    return updatedIncident;
  });
}

/**
 * Inspector workflow: rejects duplicate suggestion and confirms the incident is a distinct physical defect.
 */
export async function rejectDuplicate(
  incidentId: string,
  candidateIncidentId: string,
  inspectorId: string,
  remarks?: string
) {
  const incident = await prisma.incident.findUnique({ where: { id: incidentId } });
  if (!incident) throw new Error(`Incident not found: ${incidentId}`);

  return await prisma.$transaction(async (tx) => {
    const updatedIncident = await tx.incident.update({
      where: { id: incidentId },
      data: {
        duplicateStatus: 'rejected',
        updatedAt: new Date(),
      },
    });

    // Record immutable AuditLog entry within transaction
    await tx.auditLog.create({
      data: {
        userId: inspectorId,
        entityType: 'incident',
        entityId: incidentId,
        action: 'incident_duplicate_rejected',
        metadata: {
          rejectedCandidateIncidentId: candidateIncidentId,
          originalIncidentNumber: incident.incidentNumber,
          inspectorRemarks: remarks || 'Duplicate suggestion rejected: verified as distinct physical defect.',
          rejectedAt: new Date().toISOString(),
        },
      },
    });

    return updatedIncident;
  });
}

/**
 * Fetches all incidents marked as suspected duplicates requiring inspector review.
 */
export async function getPendingDuplicateReviews(limit: number = 20) {
  const pending = await prisma.incident.findMany({
    where: {
      duplicateStatus: 'suspected',
    },
    take: limit,
    orderBy: { reportedAt: 'desc' },
    include: {
      reporter: { select: { id: true, fullName: true, email: true } },
      images: true,
      predictions: { take: 1, orderBy: { createdAt: 'desc' } },
    },
  });

  return pending.map((inc) => ({
    ...inc,
    images: inc.images.map((img) => ({
      ...img,
      fileSize: img.fileSize.toString(),
    })),
  }));
}

# CivicPulse AI — Database Schema & Entity-Relationship (ER) Documentation

This document describes the complete relational and geospatial database schema for CivicPulse AI, implemented using PostgreSQL 16, PostGIS 3.4, and Prisma ORM.

---

## 1. Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    USERS ||--o{ INCIDENTS : "reports"
    USERS ||--o{ INCIDENT_VERIFICATIONS : "verifies"
    USERS ||--o{ MAINTENANCE_TASKS : "assigned_to / assigned_by"
    USERS ||--o{ AUDIT_LOGS : "acts"

    INCIDENTS ||--o{ INCIDENT_IMAGES : "contains"
    INCIDENTS ||--o{ AI_PREDICTIONS : "generates"
    INCIDENTS ||--o{ INCIDENT_VERIFICATIONS : "receives"
    INCIDENTS ||--o{ MAINTENANCE_TASKS : "initiates"

    MODEL_REGISTRY ||--o{ AI_PREDICTIONS : "predicts_via"
    MODEL_REGISTRY ||--o{ INCIDENT_FORECASTS : "forecasts_via"

    MAINTENANCE_TASKS ||--o{ MAINTENANCE_UPDATES : "logs"

    USERS {
        uuid id PK
        varchar full_name
        varchar email UK
        text password_hash
        user_role role
        boolean is_active
        timestamp created_at
        timestamp updated_at
    }

    INCIDENTS {
        uuid id PK
        varchar incident_number UK
        uuid reporter_id FK
        varchar title
        text description
        varchar damage_type
        varchar severity
        varchar status
        varchar priority
        decimal latitude
        decimal longitude
        geography location "PostGIS Point"
        text address
        timestamp reported_at
        timestamp verified_at
        timestamp created_at
        timestamp updated_at
    }

    INCIDENT_IMAGES {
        uuid id PK
        uuid incident_id FK
        text storage_key
        text original_filename
        varchar content_type
        bigint file_size
        varchar image_type
        timestamp uploaded_at
    }

    AI_PREDICTIONS {
        uuid id PK
        uuid incident_id FK
        uuid model_version_id FK
        varchar prediction_type
        varchar predicted_class
        float confidence
        jsonb prediction_payload
        int processing_time_ms
        timestamp created_at
    }

    INCIDENT_VERIFICATIONS {
        uuid id PK
        uuid incident_id FK
        uuid inspector_id FK
        varchar predicted_class
        varchar verified_class
        varchar predicted_severity
        varchar verified_severity
        text remarks
        timestamp verified_at
    }

    MAINTENANCE_TASKS {
        uuid id PK
        uuid incident_id FK
        uuid assigned_to FK
        uuid assigned_by FK
        varchar task_status
        varchar priority
        decimal estimated_cost
        decimal actual_cost
        int estimated_duration
        int actual_duration
        timestamp started_at
        timestamp completed_at
        timestamp created_at
    }

    MAINTENANCE_UPDATES {
        uuid id PK
        uuid task_id FK
        uuid updated_by FK
        varchar previous_status
        varchar new_status
        text remarks
        uuid image_id
        timestamp created_at
    }

    MODEL_REGISTRY {
        uuid id PK
        varchar model_name
        varchar model_type
        varchar version
        varchar framework
        text artifact_uri
        jsonb metrics
        varchar training_dataset
        varchar status
        timestamp created_at
    }

    AUDIT_LOGS {
        uuid id PK
        uuid user_id FK
        varchar entity_type
        uuid entity_id
        varchar action
        jsonb metadata
        timestamp created_at
    }

    INCIDENT_FORECASTS {
        uuid id PK
        varchar region_id
        uuid model_version_id FK
        date forecast_period
        float predicted_incidents
        float lower_bound
        float upper_bound
        timestamp generated_at
    }
```

---

## 2. Spatial Indexing & Geospatial Capabilities

### PostGIS Point Geometry
The `incidents` table contains a spatial column `location` of type `geography(Point, 4326)`:
```sql
ALTER TABLE incidents 
ADD COLUMN IF NOT EXISTS location geography(Point, 4326) 
GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude::float8, latitude::float8), 4326)::geography) STORED;
```

### Spatial Indexing (GiST)
```sql
CREATE INDEX IF NOT EXISTS idx_incidents_location_gist 
ON incidents USING GIST (location);
```

### Nearby Incident Queries
Find all incidents within a 500-meter radius of a coordinate:
```sql
SELECT id, incident_number, title, ST_Distance(location, ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326)::geography) AS distance_meters
FROM incidents
WHERE ST_DWithin(location, ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326)::geography, 500)
ORDER BY distance_meters ASC;
```

---

## 3. Database Security & Concurrency
- Primary keys are generated as UUIDv4 using `gen_random_uuid()`.
- Password hashes are stored as Argon2id/bcrypt strings in `users.password_hash`.
- Cascade deletion is enforced for child artifacts (`incident_images`, `ai_predictions`) when an incident is deleted.
- Verification and audit records preserve history without destructive updates.

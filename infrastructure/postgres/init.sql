-- ===========================================
-- CivicPulse AI - PostgreSQL Initialization
-- ===========================================
-- This script runs automatically when the PostgreSQL container
-- is first created. It sets up the required extensions and schemas.

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Verify PostGIS installation
DO $$
BEGIN
  RAISE NOTICE 'PostGIS version: %', PostGIS_Version();
  RAISE NOTICE 'CivicPulse AI database initialized successfully.';
END
$$;

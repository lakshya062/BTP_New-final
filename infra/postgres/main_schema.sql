-- Main-system database schema (aggregator per gym)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE OR REPLACE FUNCTION set_updated_at_timestamp()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS main_system (
  main_system_id UUID PRIMARY KEY,
  gym_id UUID NOT NULL,
  timezone TEXT NOT NULL DEFAULT 'UTC',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gym_owner_account (
  owner_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  full_name TEXT NOT NULL,
  password_hash TEXT,
  status TEXT NOT NULL DEFAULT 'ACTIVE',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gym_owner_access (
  owner_id UUID NOT NULL REFERENCES gym_owner_account(owner_id) ON DELETE CASCADE,
  gym_id UUID NOT NULL,
  access_level TEXT NOT NULL DEFAULT 'OWNER',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (owner_id, gym_id)
);

CREATE TABLE IF NOT EXISTS members (
  user_id UUID PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  email TEXT DEFAULT 'NA',
  membership TEXT DEFAULT 'NA',
  joined_on DATE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS exercise_data (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES members(user_id) ON DELETE CASCADE,
  username_snapshot TEXT,
  exercise TEXT NOT NULL,
  set_count INTEGER NOT NULL DEFAULT 0,
  sets_reps JSONB NOT NULL DEFAULT '[]'::jsonb,
  rep_data JSONB NOT NULL DEFAULT '[]'::jsonb,
  timestamp TIMESTAMPTZ NOT NULL,
  date DATE NOT NULL,
  gym_id UUID NOT NULL,
  main_system_id UUID NOT NULL,
  mirror_id UUID NOT NULL DEFAULT gen_random_uuid(),
  source_node TEXT NOT NULL DEFAULT 'MAIN',
  equipment_type TEXT NOT NULL DEFAULT 'FREE_WEIGHT',
  machine_id UUID,
  sync_status TEXT NOT NULL DEFAULT 'PENDING',
  synced_at TIMESTAMPTZ,
  source_edge_ip TEXT,
  source_payload_hash TEXT,
  source_batch_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS edge_ingest_log (
  ingest_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  edge_ip TEXT NOT NULL,
  mirror_id UUID,
  batch_id UUID,
  payload_hash TEXT,
  records_received INTEGER NOT NULL DEFAULT 0,
  records_applied INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'RECEIVED',
  error_text TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS server_upload_batch (
  server_upload_batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  gym_id UUID NOT NULL,
  main_system_id UUID NOT NULL,
  business_date_local DATE NOT NULL,
  payload_hash TEXT UNIQUE,
  status TEXT NOT NULL DEFAULT 'PENDING',
  session_count INTEGER NOT NULL DEFAULT 0,
  set_count INTEGER NOT NULL DEFAULT 0,
  rep_count INTEGER NOT NULL DEFAULT 0,
  sent_at_utc TIMESTAMPTZ,
  acked_at_utc TIMESTAMPTZ,
  error_text TEXT,
  created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_main_members_username ON members(username);
CREATE INDEX IF NOT EXISTS idx_main_exercise_user_id ON exercise_data(user_id);
CREATE INDEX IF NOT EXISTS idx_main_exercise_timestamp ON exercise_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_main_exercise_sync_status ON exercise_data(sync_status);
CREATE INDEX IF NOT EXISTS idx_main_exercise_gym_date ON exercise_data(gym_id, date);
CREATE INDEX IF NOT EXISTS idx_main_exercise_mirror ON exercise_data(mirror_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_main_owner_access_gym ON gym_owner_access(gym_id);

DROP TRIGGER IF EXISTS trg_main_system_updated_at ON main_system;
CREATE TRIGGER trg_main_system_updated_at
BEFORE UPDATE ON main_system
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

DROP TRIGGER IF EXISTS trg_gym_owner_account_updated_at ON gym_owner_account;
CREATE TRIGGER trg_gym_owner_account_updated_at
BEFORE UPDATE ON gym_owner_account
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

DROP TRIGGER IF EXISTS trg_main_members_updated_at ON members;
CREATE TRIGGER trg_main_members_updated_at
BEFORE UPDATE ON members
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

DROP TRIGGER IF EXISTS trg_main_exercise_data_updated_at ON exercise_data;
CREATE TRIGGER trg_main_exercise_data_updated_at
BEFORE UPDATE ON exercise_data
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

DROP TRIGGER IF EXISTS trg_edge_ingest_log_updated_at ON edge_ingest_log;
CREATE TRIGGER trg_edge_ingest_log_updated_at
BEFORE UPDATE ON edge_ingest_log
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

DROP TRIGGER IF EXISTS trg_server_upload_batch_updated_at ON server_upload_batch;
CREATE TRIGGER trg_server_upload_batch_updated_at
BEFORE UPDATE ON server_upload_batch
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

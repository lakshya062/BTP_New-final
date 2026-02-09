-- Edge database schema for mirror runtime
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE OR REPLACE FUNCTION set_updated_at_timestamp()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

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
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES members(user_id) ON DELETE CASCADE,
  username_snapshot TEXT,
  exercise TEXT NOT NULL,
  set_count INTEGER NOT NULL DEFAULT 0,
  sets_reps JSONB NOT NULL DEFAULT '[]'::jsonb,
  rep_data JSONB NOT NULL DEFAULT '[]'::jsonb,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  date DATE NOT NULL DEFAULT CURRENT_DATE,
  gym_id UUID,
  main_system_id UUID,
  mirror_id UUID DEFAULT gen_random_uuid(),
  source_node TEXT NOT NULL DEFAULT 'EDGE',
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

CREATE TABLE IF NOT EXISTS edge_sync_outbox (
  outbox_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  batch_id UUID NOT NULL,
  destination TEXT NOT NULL DEFAULT 'MAIN',
  payload_hash TEXT UNIQUE,
  payload_json JSONB NOT NULL,
  session_count INTEGER NOT NULL DEFAULT 0,
  set_count INTEGER NOT NULL DEFAULT 0,
  rep_count INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'PENDING',
  retry_count INTEGER NOT NULL DEFAULT 0,
  next_retry_at TIMESTAMPTZ,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_members_username ON members(username);
CREATE INDEX IF NOT EXISTS idx_exercise_user_id ON exercise_data(user_id);
CREATE INDEX IF NOT EXISTS idx_exercise_timestamp ON exercise_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_exercise_name ON exercise_data(exercise);
CREATE INDEX IF NOT EXISTS idx_exercise_sync_status ON exercise_data(sync_status);
CREATE INDEX IF NOT EXISTS idx_exercise_mirror_gym_date ON exercise_data(mirror_id, gym_id, date);

DROP TRIGGER IF EXISTS trg_members_updated_at ON members;
CREATE TRIGGER trg_members_updated_at
BEFORE UPDATE ON members
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

DROP TRIGGER IF EXISTS trg_exercise_data_updated_at ON exercise_data;
CREATE TRIGGER trg_exercise_data_updated_at
BEFORE UPDATE ON exercise_data
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

DROP TRIGGER IF EXISTS trg_edge_sync_outbox_updated_at ON edge_sync_outbox;
CREATE TRIGGER trg_edge_sync_outbox_updated_at
BEFORE UPDATE ON edge_sync_outbox
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

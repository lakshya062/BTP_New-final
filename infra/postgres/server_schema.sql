-- Manual server schema bootstrap for Django app `gym`
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS gym_gym (
    gym_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    location_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    api_key VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gym_mainsystem (
    main_system_id UUID PRIMARY KEY,
    gym_id UUID NOT NULL REFERENCES gym_gym(gym_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL DEFAULT 'Main Controller',
    timezone VARCHAR(64) NOT NULL DEFAULT 'UTC',
    api_key VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_heartbeat TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS gym_gymowneraccount (
    owner_id UUID PRIMARY KEY,
    email VARCHAR(254) NOT NULL UNIQUE,
    full_name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255),
    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gym_gymowneraccess (
    id BIGSERIAL PRIMARY KEY,
    owner_id UUID NOT NULL REFERENCES gym_gymowneraccount(owner_id) ON DELETE CASCADE,
    gym_id UUID NOT NULL REFERENCES gym_gym(gym_id) ON DELETE CASCADE,
    access_level VARCHAR(32) NOT NULL DEFAULT 'OWNER',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(owner_id, gym_id)
);

CREATE TABLE IF NOT EXISTS gym_mirror (
    mirror_id UUID PRIMARY KEY,
    gym_id UUID NOT NULL REFERENCES gym_gym(gym_id) ON DELETE CASCADE,
    main_system_id UUID REFERENCES gym_mainsystem(main_system_id) ON DELETE SET NULL,
    mirror_type VARCHAR(32) NOT NULL DEFAULT 'FREE_SPACE',
    machine_id UUID,
    station_label VARCHAR(128),
    status VARCHAR(50) NOT NULL DEFAULT 'Active',
    ip_address INET,
    last_ping TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gym_user (
    user_id UUID PRIMARY KEY,
    username VARCHAR(150) NOT NULL UNIQUE,
    email VARCHAR(254) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(150),
    last_name VARCHAR(150),
    phone VARCHAR(20),
    gender VARCHAR(20),
    date_of_birth DATE,
    height_cm DOUBLE PRECISION NOT NULL,
    weight_kg DOUBLE PRECISION,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_staff BOOLEAN NOT NULL DEFAULT FALSE,
    is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gym_userfacedata (
    face_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES gym_user(user_id) ON DELETE CASCADE,
    encoding_vector DOUBLE PRECISION[] NOT NULL,
    raw_image_path VARCHAR(512),
    model_version VARCHAR(50) NOT NULL DEFAULT 'v1.0'
);

CREATE TABLE IF NOT EXISTS gym_exercise (
    exercise_id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    cv_config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gym_workoutsession (
    session_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES gym_user(user_id) ON DELETE CASCADE,
    mirror_id UUID REFERENCES gym_mirror(mirror_id) ON DELETE SET NULL,
    gym_id UUID REFERENCES gym_gym(gym_id) ON DELETE SET NULL,
    main_system_id UUID REFERENCES gym_mainsystem(main_system_id) ON DELETE SET NULL,
    source_record_id UUID UNIQUE,
    source_batch_id UUID,
    gym_local_date DATE,
    status VARCHAR(32) NOT NULL DEFAULT 'COMPLETED',
    start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    total_calories DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS gym_workoutset (
    set_id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES gym_workoutsession(session_id) ON DELETE CASCADE,
    exercise_id UUID NOT NULL REFERENCES gym_exercise(exercise_id) ON DELETE CASCADE,
    session_set_number INTEGER NOT NULL,
    set_number INTEGER NOT NULL,
    reps_completed INTEGER NOT NULL DEFAULT 0,
    reps_target INTEGER NOT NULL DEFAULT 0,
    avg_form_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    video_url VARCHAR(512),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gym_repdetail (
    rep_id UUID PRIMARY KEY,
    workout_set_id UUID NOT NULL REFERENCES gym_workoutset(set_id) ON DELETE CASCADE,
    rep_number INTEGER NOT NULL,
    duration_seconds DOUBLE PRECISION NOT NULL,
    is_valid BOOLEAN NOT NULL DEFAULT TRUE,
    started_at_utc TIMESTAMPTZ,
    ended_at_utc TIMESTAMPTZ,
    weight_lbs DOUBLE PRECISION,
    weight_kg DOUBLE PRECISION,
    start_angle_deg DOUBLE PRECISION,
    end_angle_deg DOUBLE PRECISION,
    rom_deg DOUBLE PRECISION,
    telemetry_data JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS gym_ingestionbatch (
    batch_id UUID PRIMARY KEY,
    gym_id UUID NOT NULL REFERENCES gym_gym(gym_id) ON DELETE CASCADE,
    main_system_id UUID NOT NULL REFERENCES gym_mainsystem(main_system_id) ON DELETE CASCADE,
    payload_hash VARCHAR(128) NOT NULL UNIQUE,
    schema_version INTEGER NOT NULL DEFAULT 1,
    generated_at_utc TIMESTAMPTZ NOT NULL,
    received_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(32) NOT NULL DEFAULT 'RECEIVED',
    session_count INTEGER NOT NULL DEFAULT 0,
    set_count INTEGER NOT NULL DEFAULT 0,
    rep_count INTEGER NOT NULL DEFAULT 0,
    error_text TEXT
);

CREATE INDEX IF NOT EXISTS gym_workout_set_session_ex_set_idx
    ON gym_workoutset(session_id, exercise_id, set_number);
CREATE INDEX IF NOT EXISTS gym_workout_set_session_sess_set_idx
    ON gym_workoutset(session_id, session_set_number);
CREATE INDEX IF NOT EXISTS gym_repdetail_set_rep_idx
    ON gym_repdetail(workout_set_id, rep_number);
CREATE INDEX IF NOT EXISTS gym_workoutsession_user_start_idx
    ON gym_workoutsession(user_id, start_time DESC);
CREATE INDEX IF NOT EXISTS gym_workoutsession_gym_date_idx
    ON gym_workoutsession(gym_id, gym_local_date);

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DB_NAME="${SMART_MIRROR_PG_DB:-gym_edge_db}"
DB_USER="${SMART_MIRROR_PG_USER:-${SMART_MIRROR_DEFAULT_PG_USER:-${PGUSER:-$USER}}}"
DB_PASSWORD="${SMART_MIRROR_PG_PASSWORD:-${SMART_MIRROR_DEFAULT_PG_PASSWORD:-admin}}"
DB_HOST="${SMART_MIRROR_PG_HOST:-localhost}"
DB_PORT="${SMART_MIRROR_PG_PORT:-5432}"
SUDO_PASSWORD="${SMART_MIRROR_SUDO_PASSWORD:-}"

run_privileged() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
    return
  fi

  if sudo -n true >/dev/null 2>&1; then
    sudo "$@"
    return
  fi

  if [ -n "$SUDO_PASSWORD" ]; then
    printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' "$@"
    return
  fi

  echo "sudo access is required for PostgreSQL setup, but no non-interactive sudo password is available." >&2
  echo "Set SMART_MIRROR_SUDO_PASSWORD or configure passwordless sudo for this user." >&2
  return 1
}

run_as_postgres() {
  if [ "$(id -u)" -eq 0 ]; then
    if command -v runuser >/dev/null 2>&1; then
      runuser -u postgres -- "$@"
      return
    fi
    su -s /bin/bash postgres -c "$(printf '%q ' "$@")"
    return
  fi
  run_privileged -u postgres "$@"
}

admin_psql() {
  local db="$1"
  shift
  if [[ "$DB_HOST" == "localhost" || "$DB_HOST" == "127.0.0.1" || "$DB_HOST" == "::1" ]]; then
    run_as_postgres psql -p "$DB_PORT" -d "$db" "$@"
  else
    run_as_postgres psql -h "$DB_HOST" -p "$DB_PORT" -d "$db" "$@"
  fi
}

if ! command -v psql >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    run_privileged apt-get update
    run_privileged apt-get install -y postgresql postgresql-contrib
    run_privileged systemctl enable --now postgresql || true
  elif command -v dnf >/dev/null 2>&1; then
    run_privileged dnf install -y postgresql-server postgresql-contrib
    run_privileged postgresql-setup --initdb || true
    run_privileged systemctl enable --now postgresql || true
  elif command -v yum >/dev/null 2>&1; then
    run_privileged yum install -y postgresql-server postgresql-contrib
    run_privileged postgresql-setup initdb || true
    run_privileged systemctl enable --now postgresql || true
  else
    echo "Unsupported package manager for automatic PostgreSQL installation." >&2
    exit 1
  fi
fi

if command -v systemctl >/dev/null 2>&1; then
  run_privileged systemctl start postgresql || true
fi

if ! id postgres >/dev/null 2>&1; then
  echo "postgres system user is missing after installation." >&2
  exit 1
fi

export PGPASSWORD="$DB_PASSWORD"

DB_USER_SQL="${DB_USER//\'/\'\'}"
DB_NAME_SQL="${DB_NAME//\'/\'\'}"
DB_PASSWORD_SQL="${DB_PASSWORD//\'/\'\'}"
DB_USER_IDENT="${DB_USER//\"/\"\"}"

ROLE_EXISTS="$(
  admin_psql postgres -tAc \
    "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER_SQL}'" || true
)"
if [ "$ROLE_EXISTS" != "1" ]; then
  admin_psql postgres -c \
    "CREATE ROLE \"${DB_USER}\" LOGIN PASSWORD '${DB_PASSWORD_SQL}'"
else
  admin_psql postgres -c \
    "ALTER ROLE \"${DB_USER}\" WITH LOGIN PASSWORD '${DB_PASSWORD_SQL}'"
fi

DB_EXISTS="$(
  admin_psql postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname='${DB_NAME_SQL}'" || true
)"
if [ "$DB_EXISTS" != "1" ]; then
  admin_psql postgres -c \
    "CREATE DATABASE \"${DB_NAME}\" OWNER \"${DB_USER}\""
else
  admin_psql postgres -c \
    "ALTER DATABASE \"${DB_NAME}\" OWNER TO \"${DB_USER}\""
fi

admin_psql "$DB_NAME" -c \
  "GRANT ALL PRIVILEGES ON DATABASE \"${DB_NAME}\" TO \"${DB_USER}\"" >/dev/null 2>&1 || true

if ! psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" \
  -f "$ROOT_DIR/resources/sql/edge_schema.sql" >/dev/null 2>&1; then
  admin_psql "$DB_NAME" \
    -f "$ROOT_DIR/resources/sql/edge_schema.sql"
fi

# Ensure runtime role owns existing schema objects. Without this, edge app startup
# can fail with "must be owner of table members" on ALTER TABLE migrations.
admin_psql "$DB_NAME" -v ON_ERROR_STOP=1 -c \
  "DO \$\$
   DECLARE
     target_role text := '${DB_USER_SQL}';
     obj record;
   BEGIN
     EXECUTE format('ALTER SCHEMA public OWNER TO %I', target_role);
     FOR obj IN
       SELECT tablename
       FROM pg_tables
       WHERE schemaname = 'public'
     LOOP
       EXECUTE format('ALTER TABLE public.%I OWNER TO %I', obj.tablename, target_role);
     END LOOP;
     FOR obj IN
       SELECT sequence_name
       FROM information_schema.sequences
       WHERE sequence_schema = 'public'
     LOOP
       EXECUTE format('ALTER SEQUENCE public.%I OWNER TO %I', obj.sequence_name, target_role);
     END LOOP;
   END
   \$\$;
   GRANT USAGE, CREATE ON SCHEMA public TO \"${DB_USER_IDENT}\";
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO \"${DB_USER_IDENT}\";
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO \"${DB_USER_IDENT}\";"

if ! psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" \
  -c "ALTER TABLE exercise_data ALTER COLUMN mirror_id SET DEFAULT gen_random_uuid();
      UPDATE exercise_data SET mirror_id = gen_random_uuid() WHERE mirror_id IS NULL;" >/dev/null 2>&1; then
  admin_psql "$DB_NAME" \
    -c "ALTER TABLE exercise_data ALTER COLUMN mirror_id SET DEFAULT gen_random_uuid();
        UPDATE exercise_data SET mirror_id = gen_random_uuid() WHERE mirror_id IS NULL;"
fi

echo "Edge PostgreSQL setup complete (${DB_NAME})."

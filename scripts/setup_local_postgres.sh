#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PSQL_USER="${PSQL_USER:-$(id -un)}"
PSQL_HOST="${PSQL_HOST:-localhost}"
PSQL_PORT="${PSQL_PORT:-5432}"
APP_DB_USER="${APP_DB_USER:-${GYM_DB_USER:-${GYM_DEFAULT_PG_USER:-$PSQL_USER}}}"
APP_DB_PASSWORD="${APP_DB_PASSWORD:-${GYM_DB_PASSWORD:-${GYM_DEFAULT_PG_PASSWORD:-admin}}}"
APP_DB_HOST="${APP_DB_HOST:-${GYM_DB_HOST:-localhost}}"
APP_DB_PORT="${APP_DB_PORT:-${GYM_DB_PORT:-5432}}"

is_local_host() {
  local host="${1:-}"
  [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "::1" || -z "$host" ]]
}

repair_stale_postmaster_pid() {
  local data_dir="$1"
  local pid_file="$data_dir/postmaster.pid"
  [[ -f "$pid_file" ]] || return 0

  local pid
  pid="$(head -n 1 "$pid_file" 2>/dev/null | tr -d '[:space:]')"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    return 0
  fi

  local cmdline
  cmdline="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if kill -0 "$pid" >/dev/null 2>&1 && [[ -z "$cmdline" ]]; then
    echo "Process $pid is active but could not inspect command; keeping $pid_file."
    return 0
  fi
  if [[ "$cmdline" == *"/postgres"* || "$cmdline" == postgres:* ]]; then
    return 0
  fi

  echo "Removing stale PostgreSQL lock file at $pid_file (pid $pid is '$cmdline')."
  if ! rm -f "$pid_file"; then
    echo "Warning: could not remove stale lock file $pid_file; continuing." >&2
  fi
}

can_connect_bootstrap() {
  local user="$1"
  local host="$2"
  if [[ -n "$host" ]]; then
    psql -U "$user" -h "$host" -p "$PSQL_PORT" -d postgres -c "SELECT 1;" >/dev/null 2>&1
  else
    psql -U "$user" -p "$PSQL_PORT" -d postgres -c "SELECT 1;" >/dev/null 2>&1
  fi
}

bootstrap_psql() {
  local db="$1"
  shift
  if [[ -n "$BOOTSTRAP_HOST" ]]; then
    psql -U "$BOOTSTRAP_USER" -h "$BOOTSTRAP_HOST" -p "$PSQL_PORT" -d "$db" "$@"
  else
    psql -U "$BOOTSTRAP_USER" -p "$PSQL_PORT" -d "$db" "$@"
  fi
}

if ! command -v psql >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing PostgreSQL with Homebrew..."
    brew install postgresql@17
  else
    echo "psql not found and Homebrew is unavailable." >&2
    exit 1
  fi
fi

if command -v brew >/dev/null 2>&1; then
  brew services start postgresql@17 >/dev/null 2>&1 || true
  if command -v pg_isready >/dev/null 2>&1 && ! pg_isready -h "$PSQL_HOST" -p "$PSQL_PORT" >/dev/null 2>&1; then
    brew_prefix="$(brew --prefix 2>/dev/null || true)"
    if [[ -n "$brew_prefix" ]]; then
      repair_stale_postmaster_pid "$brew_prefix/var/postgresql@17"
      brew services restart postgresql@17 >/dev/null 2>&1 || true
    fi
  fi
fi

BOOTSTRAP_USER=""
BOOTSTRAP_HOST="$PSQL_HOST"
bootstrap_candidates=("$PSQL_USER" "$APP_DB_USER" "postgres")

for candidate in "${bootstrap_candidates[@]}"; do
  [[ -n "$candidate" ]] || continue

  if can_connect_bootstrap "$candidate" "$PSQL_HOST"; then
    BOOTSTRAP_USER="$candidate"
    BOOTSTRAP_HOST="$PSQL_HOST"
    break
  fi

  if is_local_host "$PSQL_HOST" && can_connect_bootstrap "$candidate" ""; then
    BOOTSTRAP_USER="$candidate"
    BOOTSTRAP_HOST=""
    break
  fi
done

if [[ -z "$BOOTSTRAP_USER" ]]; then
  echo "Cannot connect to PostgreSQL using bootstrap users: ${bootstrap_candidates[*]}." >&2
  exit 1
fi

echo "Checking PostgreSQL connectivity as bootstrap user '$BOOTSTRAP_USER'..."
bootstrap_psql postgres -c "SELECT 1;" >/dev/null

APP_DB_USER_SQL="${APP_DB_USER//\'/\'\'}"
APP_DB_PASSWORD_SQL="${APP_DB_PASSWORD//\'/\'\'}"

echo "Ensuring application role '$APP_DB_USER' exists..."
ROLE_EXISTS="$(
  bootstrap_psql postgres -tAc \
    "SELECT 1 FROM pg_roles WHERE rolname='${APP_DB_USER_SQL}'" || true
)"
if [[ "$ROLE_EXISTS" != "1" ]]; then
  bootstrap_psql postgres -c \
    "CREATE ROLE \"${APP_DB_USER}\" LOGIN PASSWORD '${APP_DB_PASSWORD_SQL}'"
else
  bootstrap_psql postgres -c \
    "ALTER ROLE \"${APP_DB_USER}\" WITH LOGIN PASSWORD '${APP_DB_PASSWORD_SQL}'"
fi

ensure_database() {
  local db_name="$1"
  local db_name_sql="${db_name//\'/\'\'}"
  local db_exists
  db_exists="$(
    bootstrap_psql postgres -tAc \
      "SELECT 1 FROM pg_database WHERE datname='${db_name_sql}'" || true
  )"
  if [[ "$db_exists" != "1" ]]; then
    bootstrap_psql postgres -c \
      "CREATE DATABASE \"${db_name}\" OWNER \"${APP_DB_USER}\""
  fi
  bootstrap_psql postgres -c \
    "ALTER DATABASE \"${db_name}\" OWNER TO \"${APP_DB_USER}\"" >/dev/null
}

echo "Ensuring databases exist (gym_edge_db, gym_main_db, gym_server_db)..."
ensure_database "gym_edge_db"
ensure_database "gym_main_db"
ensure_database "gym_server_db"

export PGPASSWORD="$APP_DB_PASSWORD"
app_psql_cmd=(psql -U "$APP_DB_USER" -h "$APP_DB_HOST" -p "$APP_DB_PORT")
"${app_psql_cmd[@]}" -d postgres -c "SELECT 1;" >/dev/null

echo "Applying edge schema..."
"${app_psql_cmd[@]}" -d gym_edge_db -f infra/postgres/edge_schema.sql
"${app_psql_cmd[@]}" -d gym_edge_db -c \
  "ALTER TABLE exercise_data ALTER COLUMN mirror_id SET DEFAULT gen_random_uuid();
   UPDATE exercise_data SET mirror_id = gen_random_uuid() WHERE mirror_id IS NULL;" >/dev/null

echo "Applying main schema..."
"${app_psql_cmd[@]}" -d gym_main_db -f infra/postgres/main_schema.sql
"${app_psql_cmd[@]}" -d gym_main_db -c \
  "ALTER TABLE exercise_data ALTER COLUMN mirror_id SET DEFAULT gen_random_uuid();
   UPDATE exercise_data SET mirror_id = gen_random_uuid() WHERE mirror_id IS NULL;" >/dev/null

echo "Server database (gym_server_db) created. Django migrations will manage schema."
echo "PostgreSQL setup complete."

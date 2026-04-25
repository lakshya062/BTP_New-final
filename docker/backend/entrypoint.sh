#!/usr/bin/env sh
set -eu

export GYM_DB_ENGINE="${GYM_DB_ENGINE:-postgres}"
export GYM_DB_NAME="${GYM_DB_NAME:-gym_server_db}"
export GYM_DB_USER="${GYM_DB_USER:-smartgym}"
export GYM_DB_PASSWORD="${GYM_DB_PASSWORD:-smartgym}"
export GYM_DB_HOST="${GYM_DB_HOST:-postgres}"
export GYM_DB_PORT="${GYM_DB_PORT:-5432}"
export SMART_GYM_DB_WAIT_TIMEOUT="${SMART_GYM_DB_WAIT_TIMEOUT:-60}"

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

python - <<'PY'
import os
import sys
import time

import psycopg2

host = os.environ["GYM_DB_HOST"]
port = int(os.environ["GYM_DB_PORT"])
name = os.environ["GYM_DB_NAME"]
user = os.environ["GYM_DB_USER"]
password = os.environ.get("GYM_DB_PASSWORD", "")
timeout = int(os.environ.get("SMART_GYM_DB_WAIT_TIMEOUT", "60"))

deadline = time.time() + max(timeout, 1)
last_error = "unknown database error"

while time.time() < deadline:
    try:
        connection = psycopg2.connect(
            dbname=name,
            user=user,
            password=password,
            host=host,
            port=port,
        )
        connection.close()
        print(f"PostgreSQL is ready at {host}:{port}.")
        break
    except Exception as exc:  # pragma: no cover - container startup path
        last_error = str(exc).strip() or exc.__class__.__name__
        print(f"Waiting for PostgreSQL at {host}:{port}...", flush=True)
        time.sleep(2)
else:
    sys.stderr.write(
        "PostgreSQL is not reachable for the backend container.\n"
        f"Expected database: {name} on {host}:{port} as {user}.\n"
        "Start the full stack with docker compose, or run a postgres container on the same Docker network.\n"
        f"Last error: {last_error}\n"
    )
    sys.exit(1)
PY

export SMART_GYM_SKIP_AUTO_MIGRATE=1
python manage.py migrate --noinput
exec python manage.py runserver 0.0.0.0:8000

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PSQL_USER="${PSQL_USER:-$(id -un)}"
PSQL_HOST="${PSQL_HOST:-}"
PSQL_PORT="${PSQL_PORT:-5432}"

psql_cmd=(psql -U "$PSQL_USER")
if [[ -n "$PSQL_HOST" ]]; then
  psql_cmd+=( -h "$PSQL_HOST" -p "$PSQL_PORT" )
fi

"${psql_cmd[@]}" -d postgres -f infra/postgres/drop_databases.sql

echo "Dropped gym_edge_db, gym_main_db, gym_server_db."

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.centralized.yml"
ENV_FILE="${SCRIPT_DIR}/.env"
START_MIRROR=0

if [[ "${1:-}" == "--start-mirror" ]]; then
  START_MIRROR=1
fi

IMAGE_BUNDLE="$(find "${SCRIPT_DIR}" -maxdepth 1 -name 'smart-gym-centralized-images-*.tar' | head -n 1)"

if [[ -z "${IMAGE_BUNDLE}" ]]; then
  echo "No smart-gym-centralized-images-*.tar archive was found in ${SCRIPT_DIR}."
  exit 1
fi

docker load -i "${IMAGE_BUNDLE}"

docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d postgres backend

if [[ "${START_MIRROR}" -eq 1 ]]; then
  export DISPLAY="${DISPLAY:-:0}"
  xhost +local:docker >/dev/null 2>&1 || true
  docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" --profile mirror up -d mirror
  echo "Backend, database, and mirror containers are starting."
else
  echo "Backend and database containers are starting."
  echo "Run ./run_centralized_linux.sh --start-mirror when X11/camera passthrough is available."
fi

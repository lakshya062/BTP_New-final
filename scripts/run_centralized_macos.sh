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
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d postgres

if [[ "${START_MIRROR}" -eq 1 ]]; then
  export DISPLAY="${DISPLAY:-host.docker.internal:0}"
  docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" --profile mirror up -d mirror
  echo "Database and mirror containers are starting."
  echo "On macOS, the mirror container needs XQuartz plus camera/device access exposed to Docker."
else
  echo "Database container is starting."
  echo "Run ./run_centralized_macos.sh --start-mirror when XQuartz/camera passthrough is ready."
fi

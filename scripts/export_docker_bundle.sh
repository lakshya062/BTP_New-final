#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_PLATFORM="${TARGET_PLATFORM:-linux/amd64}"
TARGET_ARCH="${TARGET_PLATFORM##*/}"
IMAGE_ARCHIVE_NAME="smart-gym-centralized-images-${TARGET_ARCH}.tar"
BUNDLE_DIR="${ROOT_DIR}/dist/centralized_docker_bundle_${TARGET_ARCH}"
BUNDLE_TGZ="${ROOT_DIR}/dist/smart-gym-centralized-bundle-${TARGET_ARCH}.tgz"

MIRROR_IMAGE="smart-gym-mirror:portable-${TARGET_ARCH}"
POSTGRES_IMAGE="smart-gym-postgres:portable-${TARGET_ARCH}"

mkdir -p "${BUNDLE_DIR}"

docker buildx build \
  --network host \
  --platform "${TARGET_PLATFORM}" \
  -f "${ROOT_DIR}/docker/mirror/Dockerfile" \
  -t "${MIRROR_IMAGE}" \
  --load \
  "${ROOT_DIR}"

docker buildx build \
  --platform "${TARGET_PLATFORM}" \
  -f "${ROOT_DIR}/docker/postgres/Dockerfile" \
  -t "${POSTGRES_IMAGE}" \
  --load \
  "${ROOT_DIR}"

docker save \
  "${MIRROR_IMAGE}" \
  "${POSTGRES_IMAGE}" \
  -o "${BUNDLE_DIR}/${IMAGE_ARCHIVE_NAME}"

cp "${ROOT_DIR}/docker-compose.centralized.yml" "${BUNDLE_DIR}/docker-compose.centralized.yml"
cp "${ROOT_DIR}/scripts/run_centralized_linux.sh" "${BUNDLE_DIR}/run_centralized_linux.sh"
cp "${ROOT_DIR}/scripts/run_centralized_macos.sh" "${BUNDLE_DIR}/run_centralized_macos.sh"
cp "${ROOT_DIR}/scripts/run_centralized_windows.ps1" "${BUNDLE_DIR}/run_centralized_windows.ps1"
cp "${ROOT_DIR}/docs/docker_centralized.md" "${BUNDLE_DIR}/docker_centralized.md"
cp "${ROOT_DIR}/docs/runtime_setup.md" "${BUNDLE_DIR}/runtime_setup.md"

cat > "${BUNDLE_DIR}/.env" <<EOF
SMART_GYM_MIRROR_IMAGE=${MIRROR_IMAGE}
SMART_GYM_POSTGRES_IMAGE=${POSTGRES_IMAGE}
SMART_GYM_DB_USER=smartgym
SMART_GYM_DB_PASSWORD=smartgym
EOF

tar -czf "${BUNDLE_TGZ}" -C "${ROOT_DIR}/dist" "$(basename "${BUNDLE_DIR}")"

echo "Created:"
echo "  ${BUNDLE_DIR}/${IMAGE_ARCHIVE_NAME}"
echo "  ${BUNDLE_TGZ}"

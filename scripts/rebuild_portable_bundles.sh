#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

build_amd64=1
build_arm64=1
start_local=0
start_mirror=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/rebuild_portable_bundles.sh [options]

Options:
  --amd64        Build only the Windows/Linux x86_64 bundle
  --arm64        Build only the Apple Silicon / ARM bundle
  --host-only    Build only the current machine architecture
  --all          Build both amd64 and arm64 bundles (default)
  --start        Rebuild and then start backend + postgres for the host architecture
  --start-mirror Rebuild and then start backend + postgres + mirror for the host architecture
  -h, --help     Show this help text

Examples:
  ./scripts/rebuild_portable_bundles.sh
  ./scripts/rebuild_portable_bundles.sh --host-only --start
  ./scripts/rebuild_portable_bundles.sh --amd64
EOF
}

detect_host_arch() {
  case "$(uname -m)" in
    arm64|aarch64)
      printf 'arm64\n'
      ;;
    x86_64|amd64)
      printf 'amd64\n'
      ;;
    *)
      printf 'unsupported\n'
      ;;
  esac
}

host_arch="$(detect_host_arch)"
if [[ "${host_arch}" == "unsupported" ]]; then
  echo "Unsupported host architecture: $(uname -m)"
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --amd64)
      build_amd64=1
      build_arm64=0
      ;;
    --arm64)
      build_amd64=0
      build_arm64=1
      ;;
    --host-only)
      if [[ "${host_arch}" == "arm64" ]]; then
        build_amd64=0
        build_arm64=1
      else
        build_amd64=1
        build_arm64=0
      fi
      ;;
    --all)
      build_amd64=1
      build_arm64=1
      ;;
    --start)
      start_local=1
      ;;
    --start-mirror)
      start_local=1
      start_mirror=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ "${start_local}" -eq 1 ]]; then
  if [[ "${host_arch}" == "arm64" ]]; then
    build_arm64=1
  else
    build_amd64=1
  fi
fi

build_bundle() {
  local arch="$1"
  local platform="linux/${arch}"

  echo ""
  echo "==> Building ${arch} portable bundle"
  (
    cd "${ROOT_DIR}"
    TARGET_PLATFORM="${platform}" ./scripts/export_docker_bundle.sh
  )
}

if [[ "${build_amd64}" -eq 1 ]]; then
  build_bundle "amd64"
fi

if [[ "${build_arm64}" -eq 1 ]]; then
  build_bundle "arm64"
fi

if [[ "${start_local}" -eq 1 ]]; then
  bundle_dir="${ROOT_DIR}/dist/centralized_docker_bundle_${host_arch}"

  case "$(uname -s)" in
    Darwin)
      runner="${bundle_dir}/run_centralized_macos.sh"
      ;;
    Linux)
      runner="${bundle_dir}/run_centralized_linux.sh"
      ;;
    *)
      echo "Automatic local start from this script is only supported on macOS and Linux."
      exit 1
      ;;
  esac

  echo ""
  echo "==> Starting local ${host_arch} stack"
  if [[ "${start_mirror}" -eq 1 ]]; then
    "${runner}" --start-mirror
  else
    "${runner}"
  fi
fi

echo ""
echo "Done."
if [[ "${build_amd64}" -eq 1 ]]; then
  echo "  AMD64 bundle: ${ROOT_DIR}/dist/smart-gym-centralized-bundle-amd64.tgz"
fi
if [[ "${build_arm64}" -eq 1 ]]; then
  echo "  ARM64 bundle: ${ROOT_DIR}/dist/smart-gym-centralized-bundle-arm64.tgz"
fi

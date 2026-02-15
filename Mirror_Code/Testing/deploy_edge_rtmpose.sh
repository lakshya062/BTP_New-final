#!/usr/bin/env bash
set -euo pipefail

# Placeholder credentials (replace before real use).
EDGE_HOST="192.168.1.37"
EDGE_PORT="22"
EDGE_USER="lakshya"
EDGE_PASSWORD="LAKSHYA@3738"

# Destination folder on edge device.
REMOTE_DIR="/home/${EDGE_USER}/rtmpose_testing"
REMOTE_VENV="${REMOTE_DIR}/rtmpose_env"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PY_FILE="${SCRIPT_DIR}/rtmpose_realtime_edge_linux.py"
LOCAL_REQ_FILE="${SCRIPT_DIR}/requirements.txt"

if ! command -v sshpass >/dev/null 2>&1; then
  echo "Error: sshpass is required for password-based SSH automation."
  echo "Install it first, then re-run this script."
  exit 1
fi

if [[ ! -f "${LOCAL_PY_FILE}" ]]; then
  echo "Error: missing file ${LOCAL_PY_FILE}"
  exit 1
fi

if [[ ! -f "${LOCAL_REQ_FILE}" ]]; then
  echo "Error: missing file ${LOCAL_REQ_FILE}"
  exit 1
fi

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

echo "Creating remote folder: ${REMOTE_DIR}"
sshpass -p "${EDGE_PASSWORD}" ssh "${SSH_OPTS[@]}" -p "${EDGE_PORT}" \
  "${EDGE_USER}@${EDGE_HOST}" "mkdir -p '${REMOTE_DIR}'"

echo "Copying RTMPose edge files to ${EDGE_USER}@${EDGE_HOST}:${REMOTE_DIR}"
sshpass -p "${EDGE_PASSWORD}" scp "${SSH_OPTS[@]}" -P "${EDGE_PORT}" \
  "${LOCAL_PY_FILE}" "${LOCAL_REQ_FILE}" \
  "${EDGE_USER}@${EDGE_HOST}:${REMOTE_DIR}/"

echo "Installing dependencies on edge device (sudo password == SSH password)..."
EDGE_PASSWORD_B64="$(printf '%s' "${EDGE_PASSWORD}" | base64)"
sshpass -p "${EDGE_PASSWORD}" ssh "${SSH_OPTS[@]}" -p "${EDGE_PORT}" \
  "${EDGE_USER}@${EDGE_HOST}" \
  "EDGE_PASSWORD_B64='${EDGE_PASSWORD_B64}' REMOTE_DIR='${REMOTE_DIR}' REMOTE_VENV='${REMOTE_VENV}' bash -s" <<'REMOTE_SETUP'
set -euo pipefail

EDGE_PASSWORD="$(printf '%s' "${EDGE_PASSWORD_B64}" | base64 -d)"

run_sudo() {
  printf '%s\n' "${EDGE_PASSWORD}" | sudo -S -p '' "$@"
}

need_system_pkgs=0
if ! command -v python3 >/dev/null 2>&1; then
  need_system_pkgs=1
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
  need_system_pkgs=1
fi
if ! python3 -m pip --version >/dev/null 2>&1; then
  need_system_pkgs=1
fi

if [[ "${need_system_pkgs}" -eq 1 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    # Use only official Ubuntu/Pop repositories for this setup run to avoid
    # unrelated third-party repo signature issues breaking deployment.
    TMP_APT_DIR="$(mktemp -d)"
    TMP_SOURCES="${TMP_APT_DIR}/sources.list"
    touch "${TMP_SOURCES}"

    append_official_sources() {
      local src_file="$1"
      [[ -f "${src_file}" ]] || return 0
      grep -E '^[[:space:]]*deb([[:space:]]|\[).*(archive\.ubuntu\.com|security\.ubuntu\.com|ports\.ubuntu\.com|apt\.pop-os\.org)' \
        "${src_file}" >> "${TMP_SOURCES}" || true
    }

    append_official_sources /etc/apt/sources.list
    for src in /etc/apt/sources.list.d/*.list; do
      append_official_sources "${src}"
    done

    if [[ ! -s "${TMP_SOURCES}" && -f /etc/apt/sources.list ]]; then
      cp /etc/apt/sources.list "${TMP_SOURCES}"
    fi

    APT_FILTER_OPTS=(
      -o "Dir::Etc::sourcelist=${TMP_SOURCES}"
      -o "Dir::Etc::sourceparts=-"
      -o "APT::Get::List-Cleanup=0"
    )

    run_sudo apt-get "${APT_FILTER_OPTS[@]}" update
    run_sudo apt-get "${APT_FILTER_OPTS[@]}" install -y \
      python3 python3-venv python3-pip libgl1 libglib2.0-0

    rm -rf "${TMP_APT_DIR}"
  elif command -v dnf >/dev/null 2>&1; then
    run_sudo dnf install -y python3 python3-pip python3-virtualenv mesa-libGL glib2
  else
    echo "Error: no supported package manager found for auto-install (apt-get/dnf)." >&2
    exit 1
  fi
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "Error: python3-venv is still unavailable on edge device." >&2
  exit 1
fi

python3 -m venv "${REMOTE_VENV}"
"${REMOTE_VENV}/bin/pip" install --upgrade pip
"${REMOTE_VENV}/bin/pip" install -r "${REMOTE_DIR}/requirements.txt"
"${REMOTE_VENV}/bin/python" -c "import cv2, onnxruntime, rtmlib; print('Edge dependency check passed')"
REMOTE_SETUP

echo "Transfer + setup complete."
echo "Run on edge device with:"
echo "${REMOTE_VENV}/bin/python ${REMOTE_DIR}/rtmpose_realtime_edge_linux.py --show-fps"

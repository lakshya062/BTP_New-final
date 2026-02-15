#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY_BIN="./rtmpose_env/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  echo "Missing virtualenv python: $PY_BIN" >&2
  exit 1
fi

RETARGET_CONFIG="${1:-human_body_autorig.retarget.json}"
UDP_HOST="${UDP_HOST:-127.0.0.1}"
UDP_PORT="${UDP_PORT:-7000}"
CAMERA_ID="${CAMERA_ID:-0}"

exec "$PY_BIN" rtmpose_to_godot_udp.py \
  --retarget-config "$RETARGET_CONFIG" \
  --camera-id "$CAMERA_ID" \
  --udp-host "$UDP_HOST" \
  --udp-port "$UDP_PORT" \
  --show-preview \
  --show-fps


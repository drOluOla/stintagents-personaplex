#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required. Set your Hugging Face token in RunPod env vars."
  exit 1
fi

export PERSONAPLEX_HOST="${PERSONAPLEX_HOST:-0.0.0.0}"
export PERSONAPLEX_PORT="${PERSONAPLEX_PORT:-8998}"
export BRIDGE_PORT="${BRIDGE_PORT:-8000}"
export PERSONAPLEX_WS_URL="ws://127.0.0.1:${PERSONAPLEX_PORT}/api/chat"

ensure_blackwell_torch() {
  if ! python3 - <<'PY'
import sys

try:
    import torch
except Exception:
    sys.exit(1)

if not torch.cuda.is_available():
    sys.exit(0)

arches = set(torch.cuda.get_arch_list())
sys.exit(0 if "sm_120" in arches else 1)
PY
  then
    echo "Installing Blackwell-compatible PyTorch nightly (sm_120 support)..."
    uv pip install --system --upgrade --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio

    python3 - <<'PY'
import torch

if torch.cuda.is_available() and "sm_120" not in set(torch.cuda.get_arch_list()):
    raise SystemExit("PyTorch still does not report sm_120 support after nightly install")
PY
  fi
}

ensure_blackwell_torch

python3 -m moshi.server \
  --host "${PERSONAPLEX_HOST}" \
  --port "${PERSONAPLEX_PORT}" \
  --static none \
  ${PERSONAPLEX_EXTRA_ARGS:-} &

MOSHI_PID=$!

cleanup() {
  kill "${MOSHI_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for moshi.server to be ready (up to 120s)..."
for i in {1..120}; do
  if curl -sS "http://127.0.0.1:${PERSONAPLEX_PORT}/" >/dev/null 2>&1; then
    echo "moshi.server is ready (${i}s)"
    break
  fi
  if ! kill -0 "${MOSHI_PID}" 2>/dev/null; then
    echo "ERROR: moshi.server exited before becoming ready."
    exit 1
  fi
  sleep 1
done

# Start the HTTP bridge in the background so we can monitor both processes.
python3 /app/presonaplex.py serve-http --host 0.0.0.0 --port "${BRIDGE_PORT}" &
BRIDGE_PID=$!

cleanup() {
  kill "${MOSHI_PID}" "${BRIDGE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Both services running. moshi.server PID=${MOSHI_PID}, bridge PID=${BRIDGE_PID}"
echo "PersonaPlex WS : ws://0.0.0.0:${PERSONAPLEX_PORT}/api/chat"
echo "HTTP bridge    : http://0.0.0.0:${BRIDGE_PORT}/v1/respond"

# Exit (and let RunPod restart the pod) if either service dies.
wait -n "${MOSHI_PID}" "${BRIDGE_PID}"
echo "ERROR: A service exited unexpectedly. Shutting down container."
exit 1

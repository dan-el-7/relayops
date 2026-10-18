#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Project venv missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi
mkdir -p .run
if [[ -f .run/api.pid ]] && kill -0 "$(cat .run/api.pid)" 2>/dev/null; then
  echo "RelayOps API already running (PID $(cat .run/api.pid))."
  exit 0
fi
setsid -f bash -c 'echo "$$" > "$1"; shift; exec "$@"' _ \
  "$project_root/.run/api.pid" \
  "$project_root/.venv/bin/python" -m uvicorn backend.main:app \
  --host "${RELAYOPS_HOST:-127.0.0.1}" \
  --port "${RELAYOPS_PORT:-8000}" \
  > .run/api.log 2>&1 < /dev/null
for _ in {1..30}; do
  if [[ -s .run/api.pid ]] && kill -0 "$(cat .run/api.pid)" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
echo "RelayOps API starting at http://${RELAYOPS_HOST:-127.0.0.1}:${RELAYOPS_PORT:-8000} (PID $(cat .run/api.pid)). Log: .run/api.log"

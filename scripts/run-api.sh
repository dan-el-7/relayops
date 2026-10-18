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

uvicorn_bin="$project_root/.venv/bin/uvicorn"
if [[ ! -x "$uvicorn_bin" ]]; then
  echo "Project venv missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi

uvicorn_args=(
  backend.main:app
  --host "${RELAYOPS_HOST:-127.0.0.1}"
  --port "${RELAYOPS_PORT:-8000}"
)
if [[ "${RELAYOPS_RELOAD:-0}" == "1" ]]; then
  uvicorn_args+=(--reload)
fi

exec "$uvicorn_bin" "${uvicorn_args[@]}"

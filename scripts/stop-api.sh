#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pid_file="$project_root/.run/api.pid"
if [[ ! -f "$pid_file" ]]; then
  echo "No background RelayOps API PID file found."
  exit 0
fi
api_pid="$(cat "$pid_file")"
if kill -0 "$api_pid" 2>/dev/null; then
  kill "$api_pid"
  echo "Stopped RelayOps API process $api_pid."
else
  echo "RelayOps API process $api_pid is not running."
fi
rm -f "$pid_file"

#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ ! -x .venv/bin/python ]]; then
  if ! command -v python3.12 >/dev/null 2>&1; then
    echo "CPython 3.12 is required for the hash-locked release environment." >&2
    exit 1
  fi
  python3.12 -m venv .venv
fi

python_version="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_version" != "3.12" || "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "The hash-locked release environment targets CPython 3.12 on Linux x86_64." >&2
  exit 2
fi

.venv/bin/python -m pip --disable-pip-version-check install --require-hashes -r requirements.lock
.venv/bin/python scripts/setup-embedding-model.py
echo "Python environment and pinned FastEmbed model assets are ready."

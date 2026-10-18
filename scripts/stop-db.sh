#!/usr/bin/env bash
set -euo pipefail

if podman container exists relayops-falkordb; then
  podman stop relayops-falkordb
else
  echo "No relayops-falkordb container exists."
fi
echo "The relayops-falkordb-data volume was preserved."

#!/usr/bin/env bash
set -euo pipefail

if ! command -v podman >/dev/null 2>&1; then
  echo "Podman is required to run the local FalkorDB container." >&2
  exit 1
fi

if podman container exists relayops-falkordb; then
  podman start relayops-falkordb >/dev/null
else
  podman volume exists relayops-falkordb-data || podman volume create relayops-falkordb-data >/dev/null
  podman run -d --name relayops-falkordb \
    -p 6387:6379 \
    -v relayops-falkordb-data:/data \
    --restart=unless-stopped \
    docker.io/falkordb/falkordb@sha256:d998ed0dfd91da6208f257dfd61f5330335bd6e7006cfc0fd1ddd761269e2634 >/dev/null
fi

echo "FalkorDB image: docker.io/falkordb/falkordb@sha256:d998ed0dfd91da6208f257dfd61f5330335bd6e7006cfc0fd1ddd761269e2634"
echo "FalkorDB container is running at 127.0.0.1:6387. Persistent data is in relayops-falkordb-data."

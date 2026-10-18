#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python="$project_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Project venv missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi
if [[ "$("$python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" || "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "The pinned CUDA server assets target Linux x86_64 and the project CPython 3.12 runtime." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver utilities are not visible. Keep the CPU provider or install this optional path on an NVIDIA host." >&2
  exit 2
fi
if ! nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader >/dev/null 2>&1; then
  echo "No usable NVIDIA GPU is visible to this process." >&2
  exit 2
fi

# The base helper supplies the exact model and verified CPU fallback.
./scripts/setup-local-llm.sh

asset_root="$project_root/.local-llm/cuda/downloads"
install_root="$project_root/.local-llm/cuda/b11430"
mkdir -p "$asset_root"

download_and_verify() {
  local filename="$1"
  local url="$2"
  local bytes="$3"
  local sha256="$4"
  local archive="$asset_root/$filename"
  if [[ ! -f "$archive" ]] || [[ "$(stat -c '%s' "$archive")" != "$bytes" ]] || \
     [[ "$(sha256sum "$archive" | cut -d' ' -f1)" != "$sha256" ]]; then
    local partial="$archive.partial"
    rm -f "$partial"
    curl --fail --location --retry 3 --output "$partial" "$url"
    if [[ "$(stat -c '%s' "$partial")" != "$bytes" ]] || \
       [[ "$(sha256sum "$partial" | cut -d' ' -f1)" != "$sha256" ]]; then
      rm -f "$partial"
      echo "Release asset size or SHA-256 verification failed: $filename" >&2
      return 1
    fi
    mv "$partial" "$archive"
  fi
  echo "Verified $filename ($bytes bytes, SHA-256 $sha256)"
}

download_and_verify \
  'llama-b11430-bin-ubuntu-cuda-13.4-x64.tar.gz' \
  'https://github.com/ggml-org/llama.cpp/releases/download/b11430/llama-b11430-bin-ubuntu-cuda-13.4-x64.tar.gz' \
  '152518127' \
  'cbfe295c3d18b398aa3031af7344efbef90e4dd66cf18af0d70a8e1a238a7277'
download_and_verify \
  'cudart-llama-b11430-bin-ubuntu-cuda-13.4-x64.tar.gz' \
  'https://github.com/ggml-org/llama.cpp/releases/download/b11430/cudart-llama-b11430-bin-ubuntu-cuda-13.4-x64.tar.gz' \
  '440236535' \
  '3186e8912a818b41dcc5fd94ee0d1a4a7d2b7c6a319f421d5123fad32f7a65d7'

if [[ ! -e "$install_root" ]]; then
  "$python" scripts/extract-local-llm-gpu-assets.py \
    --assets-dir "$asset_root" \
    --install-dir "$install_root"
fi

server="$install_root/llama-b11430/llama-server"
cuda_libs="$install_root/cudart-llama-b11430-bin-ubuntu-cuda-13.4-x64"
device_list="$(LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$server" --list-devices 2>&1)" || {
  printf '%s\n' "$device_list" >&2
  echo "llama.cpp could not load its CUDA backend. Check the host NVIDIA driver and available CUDA device." >&2
  exit 1
}
printf '%s\n' "$device_list"
if [[ "$device_list" != *"CUDA0:"* ]]; then
  echo "The CUDA backend loaded, but llama.cpp did not detect a CUDA device." >&2
  exit 1
fi

printf '\nGPU assets are ready. Add these entries to the untracked .env file and restart the API:\n'
printf 'RELAYOPS_AGENT_MODE=local\n'
printf 'RELAYOPS_LOCAL_LLM_SERVER=%s\n' "${server#$project_root/}"
printf 'RELAYOPS_LOCAL_LLM_LIB_DIR=%s\n' "${cuda_libs#$project_root/}"
echo "The API will only report the local CUDA provider ready after its worker confirms CUDA and GPU layer offload."

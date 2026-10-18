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
  echo "The pinned llama.cpp CPU wheel currently supports Linux x86_64 with the project Python 3.12 runtime." >&2
  echo "Set RELAYOPS_LOCAL_MODEL and install a compatible llama-cpp-python wheel for other platforms." >&2
  exit 2
fi

cache="$project_root/.local-llm"
wheels="$cache/wheels"
models="$cache/models"
mkdir -p "$wheels" "$models"
wheel="$wheels/llama_cpp_python-0.3.36-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
if [[ ! -f "$wheel" ]]; then
  curl --fail --location --retry 3 --output "$wheel" \
    'https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36/llama_cpp_python-0.3.36-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'
fi
echo 'c8a3e98093e950d7a770f794842e36f2db82b2933e8762ed7b5ffd435011e8e4  '"$wheel" | sha256sum --check --status || {
  echo "llama-cpp-python wheel checksum mismatch." >&2
  exit 1
}
"$python" -m pip install --require-hashes -r requirements-local-llm.lock
if ! "$python" -c 'import importlib.metadata; import llama_cpp; assert importlib.metadata.version("llama-cpp-python") == "0.3.36"' >/dev/null 2>&1; then
  "$python" -m pip install --no-deps "$wheel"
fi

model="$models/Qwen2.5-1.5B-Instruct-Q4_K_S.gguf"
if [[ ! -f "$model" ]] || [[ "$(sha256sum "$model" | cut -d' ' -f1)" != "cd382e90a417faa648ea012ebae8b97f94b3836be02947b0522886339ad1e1d7" ]]; then
  partial="$model.partial"
  rm -f "$partial"
  curl --fail --location --retry 3 --output "$partial" \
    'https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/9eadc66189c7641e1ddd226b8267a9119b2ce2d4/Qwen2.5-1.5B-Instruct-Q4_K_S.gguf'
  echo 'cd382e90a417faa648ea012ebae8b97f94b3836be02947b0522886339ad1e1d7  '"$partial" | sha256sum --check --status || {
    rm -f "$partial"
    echo "GGUF model checksum mismatch." >&2
    exit 1
  }
  mv "$partial" "$model"
fi

echo "Verified local CPU model at $model"
echo "Select it by setting RELAYOPS_AGENT_MODE=local (and restart the API)."

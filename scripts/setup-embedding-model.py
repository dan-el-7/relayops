#!/usr/bin/env python3
"""Fetch and verify the pinned, keyless BGE small ONNX model assets."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.model_assets import (  # noqa: E402
    BGE_MODEL_FILE,
    BGE_MODEL_SHA256,
    BGE_MODEL_SIZE,
    BGE_REPO_ID,
    BGE_REVISION,
    BGE_SNAPSHOT_FILES,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Run ./scripts/setup.sh first to install huggingface-hub.") from exc

    cache_dir = Path(
        os.getenv("RELAYOPS_MODEL_CACHE") or ROOT / ".model-cache"
    ).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(snapshot_download(
        repo_id=BGE_REPO_ID,
        revision=BGE_REVISION,
        cache_dir=str(cache_dir),
        allow_patterns=list(BGE_SNAPSHOT_FILES),
    ))
    model_file = snapshot_path / BGE_MODEL_FILE
    if not model_file.is_file():
        raise SystemExit(f"Pinned model file is missing: {model_file}")
    if model_file.stat().st_size != BGE_MODEL_SIZE or sha256(model_file) != BGE_MODEL_SHA256:
        raise SystemExit("Pinned model file failed expected size/SHA256 verification")
    missing = [name for name in BGE_SNAPSHOT_FILES if not (snapshot_path / name).is_file()]
    if missing:
        raise SystemExit(f"Pinned tokenizer/config files are missing: {', '.join(missing)}")
    print(f"Verified {BGE_REPO_ID}@{BGE_REVISION}")
    print(f"Model: {model_file} ({BGE_MODEL_SIZE} bytes, sha256={BGE_MODEL_SHA256})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the Vercel static frontend and fetch the hash-pinned embedding model."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PINNED_PNPM = "11.25.0"


def run(*command: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    frontend = ROOT / "frontend"
    pnpm = None
    if shutil.which("pnpm"):
        version = subprocess.run(
            ["pnpm", "--version"], cwd=frontend, check=True,
            text=True, capture_output=True,
        ).stdout.strip()
        if version == PINNED_PNPM:
            pnpm = ["pnpm"]
    if pnpm is None and shutil.which("corepack"):
        pnpm = ["corepack", "pnpm"]
    if pnpm is None and shutil.which("npx"):
        pnpm = ["npx", "--yes", f"pnpm@{PINNED_PNPM}"]
    if pnpm is None:
        raise SystemExit(
            f"Vercel build needs pnpm {PINNED_PNPM} (or Corepack/npx) on PATH."
        )

    run(*pnpm, "install", "--frozen-lockfile", cwd=frontend)
    run(*pnpm, "build", cwd=frontend)

    # The app's Python fallback serves frontend/dist for client-side routes.
    # Copy the same build into public/ so Vercel can serve assets from its CDN.
    dist = frontend / "dist"
    if not (dist / "index.html").is_file():
        raise SystemExit("Vite build did not produce frontend/dist/index.html")
    public = ROOT / "public"
    if public.exists():
        # Keep the tracked sentinel so Vercel detects the public directory
        # before this build hook runs; deleting it breaks static output.
        for child in public.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        public.mkdir(parents=True)
    shutil.copytree(dist, public, dirs_exist_ok=True)
    for source_name, public_name in (("relayops-demo.mp4", "demo.mp4"),):
        source = ROOT / "docs" / source_name
        if not source.is_file():
            raise SystemExit(f"Required review media is missing: {source}")
        shutil.copy2(source, public / public_name)
        # The FastAPI fallback serves assets from frontend/dist inside the
        # function bundle; retain the media there as well as in Vercel public.
        shutil.copy2(source, dist / public_name)

    # snapshot_download is pinned by backend/model_assets.py and the setup
    # script checks both file size and SHA256 before packaging the model.
    run(sys.executable, "scripts/setup-embedding-model.py", cwd=ROOT)
    model_file = (
        ROOT
        / ".model-cache/models--Qdrant--bge-small-en-v1.5-onnx-Q/snapshots/"
        / "aa8f8b060edb00e03bfdd08813a2949946c8ba55/model_optimized.onnx"
    )
    if not model_file.is_file():
        raise SystemExit("Verified BGE model asset is missing after setup.")
    print(f"Vercel build assets ready: {dist}, {public}, {model_file}")


if __name__ == "__main__":
    main()

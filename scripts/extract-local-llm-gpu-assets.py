#!/usr/bin/env python3
"""Safely unpack the pinned official llama.cpp CUDA release assets."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


ASSETS = (
    (
        "llama-b11430-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "llama-b11430",
    ),
    (
        "cudart-llama-b11430-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "cudart-llama-b11430-bin-ubuntu-cuda-13.4-x64",
    ),
)


def _extract_checked(archive: Path, expected_root: str, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise ValueError(f"Empty release archive: {archive.name}")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != expected_root
                or not (member.isfile() or member.isdir() or member.issym())
            ):
                raise ValueError(f"Unexpected archive member: {member.name}")
        # Python 3.12's data filter rejects escaping links and unsafe modes.
        tar.extractall(destination, members=members, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--install-dir", required=True, type=Path)
    args = parser.parse_args()
    args.assets_dir = args.assets_dir.resolve()
    args.install_dir = args.install_dir.resolve()

    if args.install_dir.exists():
        raise SystemExit(f"GPU asset destination already exists: {args.install_dir}")
    args.install_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=args.install_dir.parent) as temp:
        staging = Path(temp)
        for archive_name, expected_root in ASSETS:
            archive = args.assets_dir / archive_name
            if not archive.is_file():
                raise SystemExit(f"Verified release asset is missing: {archive}")
            _extract_checked(archive, expected_root, staging)

        binary = staging / "llama-b11430" / "llama-server"
        runtime_dir = staging / "cudart-llama-b11430-bin-ubuntu-cuda-13.4-x64"
        if not binary.is_file() or not (binary.stat().st_mode & 0o111):
            raise SystemExit("Release archive did not contain an executable llama-server")
        for lib in ("libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"):
            if not (runtime_dir / lib).is_file():
                raise SystemExit(f"CUDA runtime archive is missing {lib}")
        staging.rename(args.install_dir)

    print(f"Installed verified llama-server assets under {args.install_dir}")


if __name__ == "__main__":
    main()

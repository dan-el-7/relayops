#!/usr/bin/env python3
"""Render an exact, hashed requirements lock from pip's resolver report."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: lock-from-pip-report.py PIP_REPORT.json")
    report_path = Path(sys.argv[1])
    report = json.loads(report_path.read_text())
    packages = []
    seen = set()
    for item in report.get("install", []):
        metadata = item.get("metadata", {})
        name = str(metadata.get("name", "")).replace("_", "-")
        version = str(metadata.get("version", ""))
        url = str(item.get("download_info", {}).get("url", ""))
        sha256 = item.get("download_info", {}).get("archive_info", {}).get("hashes", {}).get("sha256")
        if not name or not version or not sha256 or url.startswith("file:"):
            raise SystemExit(f"missing public SHA256 metadata or non-public source for {name or '<unnamed>'}")
        normalized = name.lower().replace("-", "")
        if normalized in seen:
            raise SystemExit(f"duplicate normalized package name: {name}")
        seen.add(normalized)
        packages.append((name, version, sha256))
    if not packages:
        raise SystemExit("pip report contained no distributions")
    print("# RelayOps runtime lock for CPython 3.12 on Linux x86_64.")
    print(f"# Exact resolver snapshot: {len(packages)} public distributions with SHA256 hashes.")
    for name, version, digest in sorted(packages, key=lambda item: item[0].lower()):
        print(f"{name}=={version} --hash=sha256:{digest}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one bounded synthetic concurrency check on disposable FalkorDB graphs.

This is an engineering smoke benchmark, not a capacity or production claim. It
uses direct MemoryEngine calls and a fresh tenant registry so demo data is never
read or changed. Run only after the recorded demo capture has finished and the
machine is on AC power.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.graph_store import tenant_graph_name
from backend.memory import MemoryConfig, MemoryEngine

MAX_WORKERS = 8
READ_REQUESTS = 24
WRITE_REQUESTS = 12
CLAIM_RACERS = 8


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def ac_online() -> str | None:
    for path in (Path("/sys/class/power_supply/AC/online"), Path("/sys/class/power_supply/ACAD/online")):
        try:
            return path.read_text().strip()
        except OSError:
            pass
    return None


def gpu_snapshot() -> dict:
    command = [
        "nvidia-smi", "--query-gpu=name,pstate,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        fields = [part.strip() for part in rows[0].split(",")] if rows else []
        keys = ["name", "pstate", "power_draw_w", "utilization_gpu_percent", "memory_used_mib"]
        return dict(zip(keys, fields)) if fields else {"available": False, "reason": "no GPU rows"}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def summarize(latencies: list[float], started: float, ended: float, errors: list[str]) -> dict:
    duration = max(ended - started, 1e-9)
    return {
        "requests": len(latencies) + len(errors),
        "successes": len(latencies),
        "errors": len(errors),
        "error_samples": errors[:5],
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "max_ms": round(max(latencies, default=0.0), 2),
        "elapsed_s": round(duration, 3),
        "successful_ops_per_s": round(len(latencies) / duration, 2),
    }


def main() -> None:
    if ac_online() != "1":
        raise SystemExit("AC power was not confirmed; refusing to run the benchmark")

    suffix = uuid.uuid4().hex[:10]
    tenant = f"bench-{suffix}"
    other_tenant = f"benchiso-{suffix}"
    tenants = [tenant, other_tenant]
    graphs = [tenant_graph_name(item, tenants) for item in tenants]
    config = MemoryConfig(
        host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
        port=int(os.getenv("FALKORDB_PORT", "6387")),
        username=os.getenv("FALKORDB_USERNAME") or None,
        password=os.getenv("FALKORDB_PASSWORD") or None,
        tenant_registry={tenant: "Disposable concurrency benchmark", other_tenant: "Disposable isolation check"},
        embedding_mode="fastembed",
        category_index_mode="auto",
    )
    engine = MemoryEngine(config)
    if engine.embeddings.mode != "fastembed" or engine.embeddings.dimension != 384:
        raise SystemExit(f"Expected FastEmbed 384d, received {engine.embeddings.mode}/{engine.embeddings.dimension}")

    try:
        engine.ensure_seeded(tenant)
        engine.ensure_seeded(other_tenant)
        engine.create_memory(other_tenant, {
            "id": f"isolation-{suffix}", "kind": "episode",
            "text": f"Private synthetic tenant canary {suffix}; must never appear in the first tenant search.",
            "source_ref": "synthetic://concurrency/isolation", "category_ids": ["incident_history"],
        })

        jobs: list[tuple[str, callable]] = []
        read_jobs = []
        write_jobs = []
        for index in range(READ_REQUESTS):
            query = f"checkout database connection pool incident recovery sample {index % 4}"

            def read(query=query):
                return engine.search(tenant, query, k=5, mode="hierarchical_fallback")

            read_jobs.append(("read", read))
        for index in range(WRITE_REQUESTS):
            memory_id = f"write-{suffix}-{index}"

            def write(index=index, memory_id=memory_id):
                return engine.create_memory(tenant, {
                    "id": memory_id, "kind": "episode",
                    "text": f"Synthetic concurrent episode {index}: checkout pool recovery observation; no customer data.",
                    "source_ref": f"synthetic://concurrency/{memory_id}",
                    "category_ids": ["incident_history", "service_health"],
                })

            write_jobs.append(("write", write))
        # Interleave operations so the first wave includes reads and writes.
        for index in range(max(len(read_jobs), len(write_jobs))):
            if index < len(read_jobs):
                jobs.append(read_jobs[index])
            if index < len(write_jobs):
                jobs.append(write_jobs[index])

        by_kind = {"read": [], "write": []}
        errors = {"read": [], "write": []}
        lock = threading.Lock()

        def invoke(kind: str, operation):
            started = time.perf_counter()
            try:
                result = operation()
                if result.get("db_mode") != "falkordb":
                    raise RuntimeError("operation did not report FalkorDB")
            except Exception as exc:
                with lock:
                    errors[kind].append(f"{type(exc).__name__}: {exc}")
            else:
                elapsed_ms = (time.perf_counter() - started) * 1000
                with lock:
                    by_kind[kind].append(elapsed_ms)

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(invoke, kind, operation) for kind, operation in jobs]
            concurrent.futures.wait(futures, timeout=120)
            for future in futures:
                if future.done():
                    future.result()
        ended = time.perf_counter()
        if any(not future.done() for future in futures):
            raise RuntimeError("bounded read/write phase exceeded 120 seconds")

        # Eight simultaneous claims against one disposable open task and the same
        # observed version. The graph mutation itself performs the version check.
        claim_latencies: list[float] = []
        claim_winners: list[str] = []
        claim_losses: list[str] = []

        def claim(index: int) -> None:
            claim_started = time.perf_counter()
            try:
                result = engine.claim_task(tenant, "task-042-db-pool", f"benchmark-{index}", expected_version=0)
                claim_winners.append(str(result["state"].get("owner", f"benchmark-{index}")))
            except Exception as exc:
                claim_losses.append(f"{type(exc).__name__}: {exc}")
            finally:
                claim_latencies.append((time.perf_counter() - claim_started) * 1000)

        claim_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=CLAIM_RACERS) as pool:
            list(pool.map(claim, range(CLAIM_RACERS)))
        claim_ended = time.perf_counter()
        isolated = engine.search(tenant, f"private synthetic tenant canary {suffix}", k=10, mode="global")
        leaked = any(row.get("id") == f"isolation-{suffix}" for row in isolated["results"])

        report = {
            "schema_version": 1,
            "run_id": suffix,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "classification": "bounded synthetic engineering smoke benchmark; not a production capacity claim",
            "protocol": {
                "max_concurrency": MAX_WORKERS,
                "read_requests": READ_REQUESTS,
                "episode_write_requests": WRITE_REQUESTS,
                "atomic_claim_racers": CLAIM_RACERS,
                "claim_expected_version": 0,
                "tenant_graphs": graphs,
                "workload": "FastEmbed vector search and synthetic episode creation in parallel; claim race on one seeded disposable task",
            },
            "environment": {
                "ac_online": ac_online(),
                "gpu": gpu_snapshot(),
                "cpu_governor": Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
                if Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists() else None,
                "embedding": {"mode": engine.embeddings.mode, "model": engine.config.embedding_model, "dimension": engine.embeddings.dimension},
            },
            "results": {
                "reads": summarize(by_kind["read"], started, ended, errors["read"]),
                "episode_writes": summarize(by_kind["write"], started, ended, errors["write"]),
                "atomic_claims": {
                    "requests": CLAIM_RACERS,
                    "successes": len(claim_winners),
                    "errors": len(claim_losses),
                    "error_samples": claim_losses[:5],
                    "latency_sample_count": len(claim_latencies),
                    "latency_population": "all claim attempts, including conflicts",
                    "p50_ms": round(percentile(claim_latencies, 0.50), 2),
                    "p95_ms": round(percentile(claim_latencies, 0.95), 2),
                    "max_ms": round(max(claim_latencies, default=0.0), 2),
                    "elapsed_s": round(max(claim_ended - claim_started, 1e-9), 3),
                    "successful_ops_per_s": round(len(claim_winners) / max(claim_ended - claim_started, 1e-9), 2),
                    "winner_count": len(claim_winners),
                    "winner_actors": claim_winners,
                    "loss_count": len(claim_losses),
                },
            },
            "checks": {
                "exactly_one_claim_winner": len(claim_winners) == 1 and len(claim_losses) == CLAIM_RACERS - 1,
                "cross_tenant_isolation": not leaked,
                "all_read_write_requests_succeeded": not errors["read"] and not errors["write"] and len(by_kind["read"]) == READ_REQUESTS and len(by_kind["write"]) == WRITE_REQUESTS,
                "cleanup": "pending",
            },
        }
        if leaked:
            raise AssertionError("Cross-tenant canary appeared in disposable benchmark search")
        if len(claim_winners) != 1 or len(claim_losses) != CLAIM_RACERS - 1:
            raise AssertionError(f"Expected one atomic claim winner; winners={len(claim_winners)}, losses={len(claim_losses)}")
        if errors["read"] or errors["write"]:
            raise AssertionError("One or more bounded read/write operations failed")

        report["checks"]["cleanup"] = "pending"
    finally:
        cleanup_errors = []
        if engine.store._client is not None:
            for graph_name in graphs:
                try:
                    engine.store._client.select_graph(graph_name).delete()
                except Exception as exc:
                    cleanup_errors.append(f"{graph_name}: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            print(json.dumps({"cleanup_errors": cleanup_errors}), file=sys.stderr)
        if "report" in locals():
            report["checks"]["cleanup"] = "failed" if cleanup_errors else "passed"
            if cleanup_errors:
                report["checks"]["cleanup_errors"] = cleanup_errors
            output = ROOT / "backend" / "evaluation" / f"plugged_in_concurrency_report_{suffix}.json"
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            print(f"Wrote {output}")


if __name__ == "__main__":
    main()

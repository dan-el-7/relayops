#!/usr/bin/env python3
"""Run the fixed synthetic retrieval comparison against disposable FalkorDB graphs.

The fixture is intentionally small and manually authored. This is a reproducible
engineering comparison, not a representative production benchmark. It performs
no parameter fitting and deletes only the randomly named graphs it creates.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.memory import MemoryConfig, MemoryEngine


FIXTURE = ROOT / "backend" / "evaluation" / "heldout_retrieval_cases.json"
MODES = ("global", "hierarchical", "hierarchical_fallback")
K = 8
CANDIDATE_CAP = 32
EVIDENCE_TOKEN_BUDGET = 192
REPEATS = 3


def read_only_runtime_metadata() -> dict:
    ac_online = None
    for path in (Path("/sys/class/power_supply/AC/online"), Path("/sys/class/power_supply/ACAD/online")):
        try:
            ac_online = path.read_text().strip()
            break
        except OSError:
            pass
    try:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,pstate,power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        gpu = gpu_result.stdout.strip().splitlines()[0].strip() if gpu_result.stdout.strip() else None
    except Exception as exc:
        gpu = f"unavailable: {type(exc).__name__}: {exc}"
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        governor = governor_path.read_text().strip()
    except OSError:
        governor = None
    return {"ac_online": ac_online, "gpu_snapshot_csv": gpu, "cpu_governor": governor}

CORPUS = [
    ("heldout-incident-01", "episode", "Checkout latency rose after database workers exhausted their connection pool; draining stale workers returned p95 latency toward baseline.", ["incident_history", "metrics_alerts"]),
    ("heldout-incident-02", "episode", "Payment authorization retries produced duplicate capture events during the April gateway outage; disabling the retry loop stopped new duplicates.", ["incident_history", "customer_impact"]),
    ("heldout-incident-03", "episode", "The notification queue accumulated messages when its downstream mail relay slowed, delaying customer emails until the relay recovered.", ["incident_history", "service_health"]),
    ("heldout-pool-old", "fact", "The checkout database worker pool maximum was 18 connections before the capacity correction.", ["service_health", "deployments_changes"]),
    ("heldout-health-02", "fact", "Catalog cache health is green; median read latency is 14 milliseconds and no evictions are currently elevated.", ["service_health", "metrics_alerts"]),
    ("heldout-health-03", "fact", "Inventory API uses two retries with exponential backoff capped at 400 milliseconds.", ["service_health", "deployments_changes"]),
    ("heldout-change-01", "episode", "Release rel-2026.10.12 changed checkout connection acquisition from bounded pooling to eager worker allocation.", ["deployments_changes", "incident_history"]),
    ("heldout-change-02", "episode", "Payment release pay-7.14.2 was rolled back after authorization failures increased across two regions.", ["deployments_changes", "incident_history"]),
    ("heldout-change-03", "fact", "Autoscaling policy checkout-workers-v3 raised the maximum replicas from 12 to 20 after capacity review.", ["deployments_changes", "service_health"]),
    ("heldout-procedure-01", "procedure", "Approved pool recovery: confirm saturation, drain the oldest idle workers, verify active connections decline, then check checkout p95 stays below 700 ms for ten minutes.", ["runbooks_procedures", "service_health"]),
    ("heldout-procedure-02", "procedure", "Credential rotation: notify the service owner, create a replacement secret, roll consumers gradually, verify successful authentication, then revoke the exposed credential.", ["runbooks_procedures", "ownership_escalations"]),
    ("heldout-procedure-03", "procedure", "A load test is complete only after throughput stabilizes, error rate remains below 0.5 percent, and p95 latency returns within the approved service objective.", ["runbooks_procedures", "metrics_alerts"]),
    ("heldout-impact-01", "episode", "During the checkout latency event, conversion fell by approximately six percent for customers using the web checkout flow.", ["customer_impact", "metrics_alerts"]),
    ("heldout-impact-02", "episode", "The retry incident affected 43 billing accounts with a duplicate charge; all duplicate captures were identified for refund.", ["customer_impact", "incident_history"]),
    ("heldout-impact-03", "episode", "Northwind Market, an enterprise account, reported the longest checkout interruption at 17 minutes.", ["customer_impact", "ownership_escalations"]),
    ("heldout-metric-01", "fact", "Checkout p95 reached 870 milliseconds while the paging threshold was 700 milliseconds over a five minute window.", ["metrics_alerts", "service_health"]),
    ("heldout-metric-02", "fact", "Worker CPU peaked at 88 percent during the connection saturation event, with a five minute average of 82 percent.", ["metrics_alerts", "incident_history"]),
    ("heldout-metric-03", "procedure", "The queue-drain dashboard confirms recovery when oldest-message age falls below 30 seconds and the pending count trends downward for three samples.", ["metrics_alerts", "service_health"]),
    ("heldout-owner-01", "fact", "Payments Platform owns the Checkout API and receives the primary escalation for checkout payment-service incidents.", ["ownership_escalations", "service_health"]),
    ("heldout-owner-02", "fact", "Database Reliability is the on-call group for connection pool capacity and database saturation escalations.", ["ownership_escalations", "handoff_tasks"]),
    ("heldout-owner-03", "fact", "The production operations lead must approve a runbook change before the procedure is marked approved for responders.", ["ownership_escalations", "runbooks_procedures"]),
    ("heldout-handoff-01", "handoff", "Maya owns draining stale checkout workers; the next responder must verify connection recovery and observe p95 for ten minutes.", ["handoff_tasks", "runbooks_procedures"]),
    ("heldout-handoff-02", "handoff", "The incident commander is waiting for the customer success lead to confirm the customer-impact interval before sending the final update.", ["handoff_tasks", "customer_impact"]),
    ("heldout-handoff-03", "handoff", "The capacity follow-up is blocked until Database Reliability confirms the safe pool ceiling and assigns an on-call reviewer.", ["handoff_tasks", "ownership_escalations"]),
]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def limited_context(engine: MemoryEngine, rows: list[dict]) -> tuple[list[str], int]:
    selected: list[str] = []
    used = 0
    for row in rows:
        text = f"{row.get('kind', '')} {row.get('id', '')}: {row.get('text', '')}"
        count = int(engine.embeddings._model.token_count(text))
        if used + count > EVIDENCE_TOKEN_BUDGET:
            continue
        selected.append(str(row.get("id")))
        used += count
    return selected, used


def main() -> None:
    fixture = json.loads(FIXTURE.read_text())
    cases = fixture["cases"]
    if len(cases) != 24 or len(CORPUS) != 24:
        raise SystemExit("Expected exactly 24 fixed cases and 24 corpus records")
    gold_ids = {record[0] for record in CORPUS} | {"heldout-pool-current"}
    if any(case.get("type", "positive") == "positive" and case["gold_id"] not in gold_ids for case in cases):
        raise SystemExit("Fixture references an absent corpus record")

    tenant = f"eval-{uuid.uuid4().hex[:10]}"
    other_tenant = f"eval-{uuid.uuid4().hex[:10]}"
    graphs = [f"relayops_{tenant}", f"relayops_{other_tenant}"]
    config = MemoryConfig(
        host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
        port=int(os.getenv("FALKORDB_PORT", "6387")),
        username=os.getenv("FALKORDB_USERNAME"),
        password=os.getenv("FALKORDB_PASSWORD"),
        tenant_registry={tenant: "Disposable retrieval evaluation", other_tenant: "Disposable isolation check"},
        embedding_mode="fastembed",
        category_index_mode="auto",
    )
    engine = MemoryEngine(config)
    if engine.embeddings.mode != "fastembed" or engine.embeddings.dimension != 384:
        raise SystemExit(f"Expected semantic FastEmbed 384d, got {engine.embeddings.mode}/{engine.embeddings.dimension}")

    try:
        engine.ensure_seeded(tenant)
        engine.ensure_seeded(other_tenant)
        for memory_id, kind, text, category_ids in CORPUS:
            engine.create_memory(tenant, {
                "id": memory_id,
                "kind": kind,
                "text": text,
                "source_ref": f"synthetic://retrieval-eval/{memory_id}",
                "category_ids": category_ids,
                "slot_key": "eval.checkout.pool.max" if memory_id == "heldout-pool-old" else None,
            })
        current = engine.create_memory(tenant, {
            "id": "heldout-pool-current",
            "kind": "fact",
            "text": "The current checkout database worker pool maximum is 24 connections after the approved capacity correction.",
            "source_ref": "synthetic://retrieval-eval/pool-correction",
            "category_ids": ["service_health", "deployments_changes"],
            "slot_key": "eval.checkout.pool.max",
        })["item"]
        engine.create_memory(other_tenant, {
            "id": "other-tenant-canary",
            "kind": "fact",
            "text": "Private tenant canary phrase cobalt-orbit-719; it must not be visible in another tenant graph.",
            "source_ref": "synthetic://tenant-isolation",
            "category_ids": ["service_health"],
        })

        # Warm all code paths once; warm-ups are excluded from latency statistics.
        for mode in MODES:
            engine.search(tenant, cases[0]["query"], k=K, mode=mode, candidate_cap=CANDIDATE_CAP)

        observations: dict[str, list[dict]] = {mode: [] for mode in MODES}
        for repeat in range(REPEATS):
            for case in cases:
                for mode in MODES:
                    started = time.perf_counter()
                    result = engine.search(
                        tenant,
                        case["query"],
                        k=K,
                        mode=mode,
                        candidate_cap=CANDIDATE_CAP,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    ids = [str(row.get("id")) for row in result["results"]]
                    context_ids, context_tokens = limited_context(engine, result["results"])
                    observations[mode].append({
                        "case_id": case["id"],
                        "case_type": case.get("type", "positive"),
                        "gold_id": case.get("gold_id"),
                        "gold_categories": case.get("gold_categories", []),
                        "routed_categories": [item["id"] for item in result["routed_categories"]],
                        "rank": (ids.index(case["gold_id"]) + 1) if case.get("gold_id") in ids else None,
                        "context_hit": case.get("gold_id") in context_ids if case.get("gold_id") else False,
                        "context_tokens": context_tokens,
                        "leaf_candidates_returned": int(result.get("leaf_candidates_returned", 0)),
                        "top_score": float(result["results"][0].get("score", 0)) if result["results"] else None,
                        "forbidden_exposed": bool(set(case.get("forbidden_ids", [])).intersection(ids)),
                        "result_count": len(ids),
                        "fallback_used": bool(result["fallback_used"]),
                        "elapsed_ms": elapsed_ms,
                    })

        # Tenant isolation: a unique phrase held only by the second graph cannot cross the boundary.
        isolated = engine.search(tenant, "cobalt-orbit-719 private canary phrase", k=K, mode="global", candidate_cap=CANDIDATE_CAP)
        if any(row.get("id") == "other-tenant-canary" for row in isolated["results"]):
            raise AssertionError("Cross-tenant memory leaked into retrieval results")

        # Current correction is retrievable and its superseded revision is not.
        correction_query = "What is the current checkout database worker connection maximum?"
        correction_global = engine.search(tenant, correction_query, k=K, mode="global", candidate_cap=CANDIDATE_CAP)
        correction_routed = engine.search(tenant, correction_query, k=K, mode="hierarchical_fallback", candidate_cap=CANDIDATE_CAP)
        for name, result in (("global", correction_global), ("routed_fallback", correction_routed)):
            result_ids = {row["id"] for row in result["results"]}
            if "heldout-pool-current" not in result_ids or "heldout-pool-old" in result_ids:
                raise AssertionError(f"Correction current-only behavior failed in {name}: {result_ids}")

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fixture_version": fixture["version"],
            "tenant_graphs": graphs,
            "provider": {"mode": engine.embeddings.mode, "model": engine.config.embedding_model, "dimension": engine.embeddings.dimension},
            "settings": {"queries": len(cases), "repeats": REPEATS, "k": K, "candidate_cap_per_query": CANDIDATE_CAP, "final_context_budget_bge_tokens": EVIDENCE_TOKEN_BUDGET},
            "environment_read_only": read_only_runtime_metadata(),
            "results": {},
            "checks": {"cross_tenant_isolation": "passed", "superseded_fact_exclusion_global_and_fallback": "passed", "current_fact_id": current["id"]},
        }
        for mode, values in observations.items():
            grouped: dict[str, list[dict]] = {}
            for item in values:
                grouped.setdefault(item["case_id"], []).append(item)
            positive_cases = [case for case in cases if case.get("type", "positive") == "positive"]
            negative_cases = [case for case in cases if case.get("type") in {"unknown", "scope"}]
            case_medians = []
            for case in positive_cases:
                rows = grouped[case["id"]]
                case_medians.append({
                    "gold_id": case["gold_id"],
                    "rank": rows[0]["rank"],
                    "context_hit": rows[0]["context_hit"],
                    "routed_categories": rows[0]["routed_categories"],
                    "gold_categories": case["gold_categories"],
                })
            hits5 = sum(item["rank"] is not None and item["rank"] <= 5 for item in case_medians)
            evidence_hits = sum(item["context_hit"] for item in case_medians)
            reciprocal_ranks = [1 / item["rank"] if item["rank"] else 0.0 for item in case_medians]
            top2_coverage = None
            if mode != "global":
                top2_coverage = sum(
                    bool(set(item["routed_categories"][:2]).intersection(item["gold_categories"]))
                    for item in case_medians
                ) / len(positive_cases)
            negative_unknown = [case for case in negative_cases if case.get("type") == "unknown"]
            negative_scope = [case for case in negative_cases if case.get("type") == "scope"]
            report["results"][mode] = {
                "positive_queries": len(positive_cases),
                "recall_at_5": round(hits5 / len(positive_cases), 4),
                "mrr_at_8": round(statistics.mean(reciprocal_ranks), 4),
                "evidence_recall_under_192_bge_tokens": round(evidence_hits / len(positive_cases), 4),
                "top2_gold_category_coverage": round(top2_coverage, 4) if top2_coverage is not None else None,
                "p50_elapsed_ms": round(percentile([row["elapsed_ms"] for row in values], 0.50), 2),
                "p95_elapsed_ms": round(percentile([row["elapsed_ms"] for row in values], 0.95), 2),
                "mean_leaf_candidates_returned": round(statistics.mean(row["leaf_candidates_returned"] for row in values), 2),
                "fallback_rate": round(sum(row["fallback_used"] for row in values) / len(values), 4),
                "negative_cases": {
                    "unknown_queries": len(negative_unknown),
                    "unknown_with_any_results": sum(grouped[case["id"]][0]["result_count"] > 0 for case in negative_unknown),
                    "unknown_top1_score_range": [
                        round(min((grouped[case["id"]][0]["top_score"] for case in negative_unknown if grouped[case["id"]][0]["top_score"] is not None), default=0.0), 4),
                        round(max((grouped[case["id"]][0]["top_score"] for case in negative_unknown if grouped[case["id"]][0]["top_score"] is not None), default=0.0), 4),
                    ],
                    "scope_queries": len(negative_scope),
                    "scope_forbidden_exposures": sum(grouped[case["id"]][0]["forbidden_exposed"] for case in negative_scope),
                    "note": "Search is a retrieval API, not an abstention classifier; unknown results are measured, not counted as correct answers.",
                },
                "cases": case_medians,
            }
        output = ROOT / "backend" / "evaluation" / "latest_retrieval_eval.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print(f"Wrote {output}")
    finally:
        if engine.store._client is not None:
            for graph_name in graphs:
                try:
                    engine.store._client.select_graph(graph_name).delete()
                except Exception:
                    pass


if __name__ == "__main__":
    main()

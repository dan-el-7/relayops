"""API safety and workflow tests against disposable FalkorDB tenant graphs.

Run with ``FALKORDB_INTEGRATION=1``. No Acme or Northstar graph is read or
mutated; each test fixture selects unpredictable, isolated tenant IDs.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.agents import AgentRuntime
from backend.coordination import CoordinationService
from backend.graph_store import GraphConnectionConfig, GraphStore
from backend.memory import MemoryConfig, MemoryEngine


pytestmark = pytest.mark.skipif(
    os.getenv("FALKORDB_INTEGRATION") != "1",
    reason="requires FALKORDB_INTEGRATION=1 and a live FalkorDB server",
)


@pytest.fixture
def api_fixture(monkeypatch):
    # These API safety invariants exercise deterministic workflow behavior.
    # Keep them independent of an operator's live LLM provider environment.
    monkeypatch.setenv("RELAYOPS_AGENT_MODE", "scripted")
    suffix = uuid.uuid4().hex[:10]
    tenant_a = f"test-a-{suffix}"
    tenant_b = f"test-b-{suffix}"
    registry = {tenant_a: "Isolated Test A", tenant_b: "Isolated Test B"}
    monkeypatch.setattr(main, "TENANT_REGISTRY", registry)

    memory = MemoryEngine(MemoryConfig(
        host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
        port=int(os.getenv("FALKORDB_PORT", "6387")),
        username=os.getenv("FALKORDB_USERNAME"),
        password=os.getenv("FALKORDB_PASSWORD"),
        tenant_registry=registry,
        embedding_mode="demo",
        category_index_mode="global",
    ))
    for tenant in registry:
        memory.ensure_seeded(tenant)

    main.app.state.memory = memory
    main.app.state.agents = AgentRuntime(memory)
    main.app.state.coordination = CoordinationService(memory)
    main.app.state.startup_error = None
    client = TestClient(main.app)
    try:
        yield client, memory, tenant_a, tenant_b
    finally:
        client.close()
        for tenant in registry:
            try:
                memory.store._client.select_graph(f"relayops_{tenant}").delete()
            except Exception:
                pass


def test_api_validates_tenants_payloads_and_tenant_scoped_ids(api_fixture):
    client, memory, tenant_a, tenant_b = api_fixture

    assert client.get("/api/state?tenant_id=unknown").status_code == 404
    assert client.get("/api/state?tenant_id=../../acme-retail").status_code == 404
    assert client.post("/api/memories", json={
        "tenant_id": tenant_a, "kind": "episode", "text": "valid enough text", "extra": True,
    }).status_code == 422
    assert client.post("/api/memories/search", json={
        "tenant_id": tenant_a, "query": "valid query", "category_ids": ["../../graph"],
    }).status_code == 422
    assert client.post("/api/agents/not-an-agent/run", json={"tenant_id": tenant_a}).status_code == 404

    foreign_memory = client.post("/api/memories", json={
        "tenant_id": tenant_a,
        "kind": "fact",
        "text": "Private canary value for tenant A only.",
        "source_ref": "test://private/a",
        "slot_key": "private.canary",
    })
    assert foreign_memory.status_code == 200, foreign_memory.text
    foreign_id = foreign_memory.json()["item"]["id"]
    search_b = client.post("/api/memories/search", json={
        "tenant_id": tenant_b, "query": "Private canary value for tenant A only.", "mode": "global",
    })
    assert search_b.status_code == 200, search_b.text
    assert foreign_id not in {item["id"] for item in search_b.json()["results"]}
    correct_foreign = client.post("/api/facts/correct", json={
        "tenant_id": tenant_b, "fact_id": foreign_id, "corrected_text": "Wrong tenant correction attempt.",
    })
    assert correct_foreign.status_code == 422

    graph_a = memory._graph(tenant_a)
    memory.store.query(graph_a, "CREATE (:Task {id:$id, status:'open', version:0})", {"id": f"foreign-{foreign_id}"})
    cross_tenant_task = client.post(f"/api/tasks/foreign-{foreign_id}/claim", json={"tenant_id": tenant_b})
    assert cross_tenant_task.status_code == 404


def test_api_fact_correction_persists_history_and_excludes_old_vector(api_fixture):
    client, memory, tenant_a, _ = api_fixture
    created = client.post("/api/memories", json={
        "tenant_id": tenant_a,
        "kind": "fact",
        "text": "The isolated system worker cap is 11.",
        "source_ref": "test://worker-cap/old",
        "category_ids": ["service_health"],
        "incident_id": "inc-checkout-042",
        "slot_key": "test.worker_cap",
    })
    assert created.status_code == 200, created.text
    old_id = created.json()["item"]["id"]
    corrected = client.post("/api/facts/correct", json={
        "tenant_id": tenant_a,
        "fact_id": old_id,
        "corrected_text": "The isolated system worker cap is 17 after verification.",
        "source_ref": "test://worker-cap/new",
    })
    assert corrected.status_code == 200, corrected.text
    current_id = corrected.json()["current_fact"]["id"]

    state = client.get(f"/api/state?tenant_id={tenant_a}").json()
    revisions = {item["id"]: item for item in state["facts"]}
    assert revisions[old_id]["status"] == "superseded"
    assert revisions[current_id]["status"] == "active"

    search = client.post("/api/memories/search", json={
        "tenant_id": tenant_a,
        "query": "isolated system worker cap verified",
        "mode": "global",
    })
    assert search.status_code == 200, search.text
    assert old_id not in {item["id"] for item in search.json()["results"]}

    restarted = MemoryEngine(memory.config)
    persisted = restarted.state(tenant_a)
    persisted_revisions = {item["id"]: item for item in persisted["facts"]}
    assert persisted_revisions[old_id]["status"] == "superseded"
    assert persisted_revisions[current_id]["status"] == "active"


def test_api_concurrent_claim_has_one_winner_and_runbook_approval_persists(api_fixture):
    client, memory, tenant_a, _ = api_fixture
    path = "/api/tasks/task-042-db-pool/claim"

    def claim(actor: str):
        return client.post(path, json={
            "tenant_id": tenant_a, "actor": actor, "expected_version": 0,
        })

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(claim, [f"operator-{index}" for index in range(8)]))
    assert sum(response.status_code == 200 for response in responses) == 1
    assert all(response.status_code in {200, 409} for response in responses)

    approval = client.post("/api/runbooks/runbook-checkout-pool-recovery/approve", json={
        "tenant_id": tenant_a, "actor": "test-approver",
    })
    assert approval.status_code == 200, approval.text
    assert approval.json()["state"]["status"] == "approved"
    graph = memory._graph(tenant_a)
    event_count = memory.store.rows(memory.store.query(
        graph,
        "MATCH (e:ActionEvent {runbook_id:'runbook-checkout-pool-recovery'}) RETURN count(e) AS count",
    ))[0]["count"]
    repeated = client.post("/api/runbooks/runbook-checkout-pool-recovery/approve", json={
        "tenant_id": tenant_a, "actor": "test-approver",
    })
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["state"]["version"] == approval.json()["state"]["version"]
    assert memory.store.rows(memory.store.query(
        graph,
        "MATCH (e:ActionEvent {runbook_id:'runbook-checkout-pool-recovery'}) RETURN count(e) AS count",
    ))[0]["count"] == event_count
    state = client.get(f"/api/state?tenant_id={tenant_a}").json()
    assert state["runbook"]["approval_status"] == "approved"


def test_repeated_investigator_runs_do_not_nest_their_own_summaries(api_fixture):
    client, _, tenant_a, _ = api_fixture
    path = "/api/agents/investigator/run"
    body = {"tenant_id": tenant_a, "incident_id": "inc-checkout-042"}
    first = client.post(path, json=body)
    assert first.status_code == 200, first.text
    second = client.post(path, json=body)
    assert second.status_code == 200, second.text

    first_ids = [item["id"] for item in first.json()["evidence"]]
    second_ids = [item["id"] for item in second.json()["evidence"]]
    assert first_ids == second_ids
    assert first_ids
    assert all(not str(item.get("source_ref", "")).startswith("relayops://agent/") for item in second.json()["evidence"])


def test_api_simulated_action_requires_fresh_task_and_retry_is_idempotent(api_fixture):
    client, memory, tenant_a, _ = api_fixture
    created = client.post("/api/memories", json={
        "tenant_id": tenant_a,
        "kind": "fact",
        "text": "A correction that makes the pending task stale.",
        "category_ids": ["service_health"],
        "incident_id": "inc-checkout-042",
        "slot_key": "test.freshness_gate",
    })
    assert created.status_code == 200, created.text
    correction = client.post("/api/facts/correct", json={
        "tenant_id": tenant_a,
        "fact_id": created.json()["item"]["id"],
        "corrected_text": "Verified correction for the pending task freshness gate.",
    })
    assert correction.status_code == 200, correction.text

    action_path = "/api/tasks/task-042-db-pool/simulate-action"
    action_body = {"tenant_id": tenant_a, "action": "complete", "actor": "test-operator"}
    stale_action = client.post(action_path, json=action_body)
    assert stale_action.status_code in {409, 422}

    refresh = client.post("/api/tasks/task-042-db-pool/refresh", json={"tenant_id": tenant_a})
    assert refresh.status_code == 200, refresh.text
    first = client.post(action_path, json=action_body)
    assert first.status_code == 200, first.text
    first_version = first.json()["state"]["version"]
    graph = memory._graph(tenant_a)
    event_count = memory.store.rows(memory.store.query(
        graph, "MATCH (e:ActionEvent {task_id:'task-042-db-pool'}) RETURN count(e) AS count",
    ))[0]["count"]

    repeated = client.post(action_path, json=action_body)
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["state"]["version"] == first_version
    assert memory.store.rows(memory.store.query(
        graph, "MATCH (e:ActionEvent {task_id:'task-042-db-pool'}) RETURN count(e) AS count",
    ))[0]["count"] == event_count

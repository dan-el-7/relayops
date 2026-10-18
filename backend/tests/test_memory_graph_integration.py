"""Integration checks against an actual FalkorDB module.

Set FALKORDB_INTEGRATION=1 to run these checks. They use a unique disposable
tenant graph and delete only that test graph during teardown.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.graph_store import GraphConnectionConfig, GraphStore, GraphUnavailable, UnknownTenant
from backend.memory import CATEGORIES, MemoryConfig, MemoryEngine


pytestmark = pytest.mark.skipif(
    os.getenv("FALKORDB_INTEGRATION") != "1",
    reason="requires FALKORDB_INTEGRATION=1 and a live FalkorDB server",
)


class _ThreeDimensionalSemanticProvider:
    """Dependency-free stand-in to exercise 256d -> 3d schema migration."""

    mode = "fastembed"
    ready = True
    semantic = True
    dimension = 3
    error = None

    @staticmethod
    def embed(text: str) -> list[float]:
        return [float((len(text) % 7) + 1), 1.0, 0.5]


def _connection() -> tuple[GraphStore, object, str]:
    store = GraphStore(GraphConnectionConfig(
        host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
        port=int(os.getenv("FALKORDB_PORT", "6387")),
        username=os.getenv("FALKORDB_USERNAME"),
        password=os.getenv("FALKORDB_PASSWORD"),
    ))
    graph_name = f"relayops_test_{uuid.uuid4().hex[:10]}"
    graph = store._client.select_graph(graph_name)
    graph.query("RETURN 1 AS ready")
    return store, graph, graph_name


def _drop(store: GraphStore, graph_name: str) -> None:
    if store._client is not None:
        try:
            store._client.select_graph(graph_name).delete()
        except Exception:
            # A test may have failed before creating this disposable graph.
            pass


def test_category_label_add_remove_updates_vector_index_membership():
    store, graph, graph_name = _connection()
    try:
        store.ensure_vector_index(graph, "CatProbe", "embedding", 3)
        vector = [1.0, 0.0, 0.0]
        store.query(graph, "CREATE (m:Memory {id:'probe', embedding:vecf32($v)})", {"v": vector})

        assert store.vector_search(graph, "CatProbe", "embedding", vector, 5) == []

        store.query(graph, "MATCH (m:Memory {id:'probe'}) SET m:CatProbe")
        added = store.vector_search(graph, "CatProbe", "embedding", vector, 5)
        assert [item["id"] for item in added] == ["probe"]
        assert added[0]["score"] == pytest.approx(1.0)

        store.query(graph, "MATCH (m:Memory {id:'probe'}) REMOVE m:CatProbe")
        assert store.vector_search(graph, "CatProbe", "embedding", vector, 5) == []
    finally:
        _drop(store, graph_name)


def test_model_category_override_is_whitelisted_and_bounded():
    tenant = f"test-route-{uuid.uuid4().hex[:8]}"
    config = MemoryConfig(
        host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
        port=int(os.getenv("FALKORDB_PORT", "6387")),
        tenant_registry={tenant: "Category route test"},
        embedding_mode="demo",
        category_index_mode="auto",
    )
    engine = MemoryEngine(config)
    graph_name = f"relayops_{tenant}"
    try:
        engine.ensure_seeded(tenant)
        routed = engine.search(
            tenant,
            "current checkout recovery procedure",
            mode="hierarchical",
            category_ids=["runbooks_procedures", "service_health"],
            candidate_cap=24,
        )
        assert [item["id"] for item in routed["routed_categories"]] == [
            "runbooks_procedures", "service_health",
        ]
        assert routed["search_strategy"] == "category_hnsw"
        assert routed["leaf_candidates_returned"] <= 24

        with pytest.raises(ValueError, match="Unknown category ID"):
            engine.search(
                tenant, "ignored", mode="hierarchical", category_ids=["arbitrary_graph_label"]
            )
        with pytest.raises(ValueError, match="at most three"):
            engine.search(
                tenant,
                "ignored",
                mode="hierarchical",
                category_ids=["incident_history", "service_health", "metrics_alerts", "handoff_tasks"],
            )
    finally:
        _drop(engine.store, graph_name)


def test_vector_property_updates_change_observed_cosine_distance():
    store, graph, graph_name = _connection()
    try:
        store.ensure_vector_index(graph, "CurrentMemory", "embedding", 3)
        store.query(graph, "CREATE (:CurrentMemory:Memory {id:'probe', embedding:vecf32([1.0,0.0,0.0])})")
        exact = store.vector_search(graph, "CurrentMemory", "embedding", [1.0, 0.0, 0.0], 2)
        assert exact[0]["score"] == pytest.approx(1.0)

        store.query(graph, "MATCH (m:Memory {id:'probe'}) SET m.embedding=vecf32([0.0,1.0,0.0])")
        orthogonal = store.vector_search(graph, "CurrentMemory", "embedding", [1.0, 0.0, 0.0], 2)
        assert orthogonal[0]["score"] == pytest.approx(0.0, abs=0.02)
    finally:
        _drop(store, graph_name)


def test_seeded_engine_persists_tenant_isolation_and_fact_correction():
    tenant = f"test-{uuid.uuid4().hex[:8]}"
    second_tenant = f"test-{uuid.uuid4().hex[:8]}"
    config = MemoryConfig(
        host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
        port=int(os.getenv("FALKORDB_PORT", "6387")),
        tenant_registry={tenant: "Test Tenant", second_tenant: "Second Tenant"},
        embedding_mode="demo",
        category_index_mode="auto",
    )
    engine = MemoryEngine(config)
    graphs = [f"relayops_{tenant}", f"relayops_{second_tenant}"]
    try:
        engine.ensure_seeded(tenant)
        engine.ensure_seeded(second_tenant)
        created = engine.create_memory(tenant, {
            "kind": "fact", "text": "The worker cap is 15 for this test tenant.",
            "source_ref": "test://fact/1", "category_ids": ["service_health"],
            "service_id": "svc-checkout-api", "incident_id": "inc-checkout-042",
            "slot_key": "test.worker_cap",
        })["item"]
        fact_id = created["id"]

        correction = engine.correct_fact(
            tenant, fact_id, "The corrected worker cap is 18 for this test tenant.", "test://fact/2"
        )
        assert correction["superseded_fact_id"] == fact_id
        current_id = correction["current_fact"]["id"]

        # A fresh process-level engine sees committed graph state, not an object cache.
        restarted = MemoryEngine(config)
        state = restarted.state(tenant)
        assert any(fact["id"] == current_id for fact in state["facts"])
        old_fact = next(fact for fact in state["facts"] if fact["id"] == fact_id)
        assert old_fact["status"] == "superseded"

        graph = restarted._graph(tenant)
        stale_task = restarted.store.rows(restarted.store.query(
            graph, "MATCH (t:Task {id:'task-042-db-pool'}) RETURN t.is_stale AS stale"
        ))
        assert stale_task and stale_task[0]["stale"] is True
        with pytest.raises(ValueError, match="state changed"):
            restarted.simulate_action(tenant, "task-042-db-pool", "start", "test-operator")
        refreshed = restarted.refresh_task(tenant, "task-042-db-pool", "test-operator")["state"]
        assert current_id in refreshed["fact_snapshot_ids"]
        fresh_task = restarted.store.rows(restarted.store.query(
            graph, "MATCH (t:Task {id:'task-042-db-pool'}) RETURN t.is_stale AS stale"
        ))
        assert fresh_task[0]["stale"] is False

        first_action = restarted.simulate_action(
            tenant, "task-042-db-pool", "start", "test-operator", expected_version=refreshed["version"]
        )["state"]
        second_action = restarted.simulate_action(
            tenant, "task-042-db-pool", "start", "test-operator", expected_version=refreshed["version"]
        )["state"]
        assert first_action["version"] == second_action["version"]
        assert second_action["idempotent"] is True
        action_events = restarted.store.rows(restarted.store.query(
            graph, "MATCH (e:ActionEvent {task_id:'task-042-db-pool'}) RETURN count(e) AS n"
        ))
        assert action_events[0]["n"] == 1

        approved = restarted.approve_runbook(
            tenant, "runbook-checkout-pool-recovery", "test-operator", expected_version=1
        )["state"]
        approved_retry = restarted.approve_runbook(
            tenant, "runbook-checkout-pool-recovery", "test-operator", expected_version=1
        )["state"]
        assert approved["version"] == approved_retry["version"]
        assert approved_retry["idempotent"] is True
        approval_events = restarted.store.rows(restarted.store.query(
            graph, "MATCH (e:ActionEvent {runbook_id:'runbook-checkout-pool-recovery', action:'approve'}) RETURN count(e) AS n"
        ))
        assert approval_events[0]["n"] == 1
        timeline = restarted.state(tenant)["timeline"]
        assert {"task_action", "runbook_approval"}.issubset({event.get("kind") for event in timeline})

        other_graph = restarted._graph(second_tenant)
        other_rows = restarted.store.rows(restarted.store.query(
            other_graph, "MATCH (m:Memory {id:$id}) RETURN m.id AS id", {"id": current_id}
        ))
        assert other_rows == []

        with pytest.raises(UnknownTenant):
            restarted.state("not-configured")

        def attempt_claim(actor: str) -> bool:
            try:
                restarted.claim_task(second_tenant, "task-042-db-pool", actor, expected_version=0)
                return True
            except ValueError:
                return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt_claim, ["operator-a", "operator-b"]))
        assert sum(outcomes) == 1
    finally:
        client = engine.store._client
        if client is not None:
            for graph_name in graphs:
                try:
                    client.select_graph(graph_name).delete()
                except Exception:
                    pass


def test_embedding_space_change_reindexes_without_losing_fact_history():
    tenant = f"test-{uuid.uuid4().hex[:8]}"
    config = MemoryConfig(
        host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
        port=int(os.getenv("FALKORDB_PORT", "6387")),
        tenant_registry={tenant: "Migration Test"},
        embedding_mode="demo",
    )
    first = MemoryEngine(config)
    graph_name = f"relayops_{tenant}"
    try:
        first.ensure_seeded(tenant)
        old_state = first.state(tenant)
        assert any(fact["id"] == "fact-db-pool-v1" for fact in old_state["facts"])

        changed = MemoryEngine(config)
        changed.embeddings = _ThreeDimensionalSemanticProvider()
        changed._category_vectors = {
            category["id"]: changed.embeddings.embed(category["description"])
            for category in CATEGORIES
        }
        changed.ensure_seeded(tenant)
        graph = changed._graph(tenant)

        indexes = changed.store.index_state(graph)
        memory_indexes = [index for index in indexes if index.get("properties") == ["embedding"]]
        assert len(memory_indexes) == 9
        assert all(index["options"]["embedding"]["dimension"] == 3 for index in memory_indexes)

        after = changed.state(tenant)
        old_revision = next(fact for fact in after["facts"] if fact["id"] == "fact-db-pool-v1")
        assert old_revision["status"] == "superseded"
        assert any(fact["id"] == "fact-db-pool-current" for fact in after["facts"])
        # Old model vectors have been replaced, but the current fact remains indexed.
        current = changed.store.vector_search(
            graph, "CurrentMemory", "embedding", changed.embeddings.embed("database pool maximum connections"), 20
        )
        assert "fact-db-pool-current" in {row["id"] for row in current}
        assert "fact-db-pool-v1" not in {row["id"] for row in current}
    finally:
        try:
            first.store._client.select_graph(graph_name).delete()
        except Exception:
            pass


def test_retrieval_smoke_check_tracks_current_fact_revision_after_correction():
    tenant = f"test-{uuid.uuid4().hex[:8]}"
    config = MemoryConfig(
        host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
        port=int(os.getenv("FALKORDB_PORT", "6387")),
        tenant_registry={tenant: "Evaluation Test"},
        embedding_mode="demo",
    )
    engine = MemoryEngine(config)
    graph_name = f"relayops_{tenant}"
    try:
        engine.ensure_seeded(tenant)
        before = engine.evaluate(tenant)
        before_pool = next(check for check in before["checks"] if check["check"] == "pool setting current")
        assert before_pool["expected_id"] == "fact-db-pool-current"
        assert before_pool["rank"] == 1

        graph = engine._graph(tenant)
        old_id = "fact-db-pool-current"
        old_status = engine.store.rows(engine.store.query(
            graph, "MATCH (m:Memory {id:$id}) RETURN m.status AS status", {"id": old_id}
        ))[0]["status"]
        assert old_status == "active"

        corrected = engine.correct_fact(
            tenant, old_id,
            "Checkout API database pool max_connections is 24 after approved change CHG-8821.",
            "change://db-pool/CHG-8821-correction",
        )["current_fact"]
        after = engine.evaluate(tenant)
        after_pool = next(check for check in after["checks"] if check["check"] == "pool setting current")
        assert after_pool["expected_id"] == corrected["id"]
        assert after_pool["expected_id"] != old_id
        assert after_pool["rank"] == 1

        history = engine.store.rows(engine.store.query(
            graph,
            "MATCH (m:Memory {kind:'fact', slot_key:'checkout.db_pool.max_connections'}) "
            "RETURN m.id AS id, m.status AS status",
        ))
        assert len(history) == 3  # initial 12, seed current 20, corrected 24
        assert sum(item["status"] == "active" for item in history) == 1
        assert next(item for item in history if item["id"] == old_id)["status"] == "superseded"

        retrieved = engine.search(
            tenant, "What is the current checkout DB max_connections setting?", k=8, mode="global"
        )["results"]
        retrieved_ids = {item["id"] for item in retrieved}
        assert corrected["id"] in retrieved_ids
        assert old_id not in retrieved_ids
    finally:
        try:
            engine.store._client.select_graph(graph_name).delete()
        except Exception:
            pass

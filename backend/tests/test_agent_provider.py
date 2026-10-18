"""No-network checks for agent mode selection and the Responses provider wire-up."""

from __future__ import annotations

import sys
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents import AgentDecision, AgentRuntime, LocalModelWorker


class StubMemory:
    def __init__(self):
        self.created = []

    def search(self, tenant_id, query, **kwargs):
        if kwargs.get("category_ids"):
            return {
                "results": [
                    {"id": "memory-2", "kind": "fact", "text": "Current checkout customer impact is elevated.", "source_ref": "test://impact"},
                ],
                "routed_categories": [{"id": "customer_impact", "score": 0.91}],
                "fallback_used": False,
                "mode": kwargs["mode"],
                "candidate_count": 1,
            }
        return {
            "results": [{
                "id": "memory-1",
                "kind": "fact",
                "text": "Observed worker cap is 17.",
                "source_ref": "test://worker-cap",
            }],
            "routed_categories": [],
            "fallback_used": False,
            "mode": kwargs["mode"],
            "candidate_count": 1,
        }

    def categories(self, tenant_id):
        return [
            {"id": "customer_impact", "name": "Customer impact", "description": "Customer effects."},
            {"id": "service_health", "name": "Service health", "description": "Current health."},
        ]

    def create_memory(self, tenant_id, payload):
        self.created.append((tenant_id, payload))
        return {"item": {"id": "summary-1"}, "db_mode": "falkordb"}


def test_default_agent_mode_is_honestly_scripted_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RELAYOPS_AGENT_MODE", raising=False)
    monkeypatch.setenv("RELAYOPS_AGENT_MODE", "scripted")
    runtime = AgentRuntime(StubMemory())
    assert runtime.capabilities()["mode"] == "scripted"
    assert runtime.capabilities()["api_key_present"] is False
    assert runtime.run("triage", "test-tenant", "inc-test")["llm_mode"] == "scripted"


def test_scripted_abstention_has_no_evidence_ids_or_actions(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("RELAYOPS_AGENT_MODE", "scripted")
    memory = StubMemory()
    memory.search = lambda tenant_id, query, **kwargs: {
        "results": [], "routed_categories": [], "fallback_used": False,
        "mode": kwargs["mode"], "candidate_count": 0,
    }

    result = AgentRuntime(memory).run("triage", "test-tenant", "inc-unknown")

    assert result["evidence_ids"] == []
    assert result["next_actions"] == []
    assert result["abstained"] is True
    assert result["insufficient_evidence"] is True


def test_openai_responses_provider_uses_env_model_without_network(monkeypatch):
    seen = {}

    class FakeResponses:
        def parse(self, *, model, input, text_format):
            seen["model"] = model
            seen["input"] = input
            seen["text_format"] = text_format
            return SimpleNamespace(output_parsed=AgentDecision(
                summary="Assessment is grounded in the supplied current record.",
                evidence_ids=["memory-1"],
                next_actions=["Verify the current service signal"],
                confidence="medium",
            ))

    class FakeOpenAI:
        def __init__(self, *, api_key, timeout, max_retries):
            # This is an in-process fake; no credential or network request is used.
            seen["client_config"] = (api_key, timeout, max_retries)
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("RELAYOPS_AGENT_MODE", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    runtime = AgentRuntime(StubMemory())
    result = runtime.run("triage", "test-tenant", "inc-test")
    assert result["llm_mode"] == "openai"
    assert result["model"] == "gpt-test-model"
    assert result["evidence_ids"] == ["memory-1"]
    assert seen["model"] == "gpt-test-model"
    assert seen["text_format"] is AgentDecision
    assert seen["client_config"] == ("test-only-not-a-real-key", 30.0, 1)


def test_local_provider_uses_model_choices_for_bounded_category_search(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test fixture, not a model")
    seen = {}
    call_count = 0

    def fake_decide(self, system, payload):
        nonlocal call_count
        call_count += 1
        seen[call_count] = payload
        return AgentDecision(
            summary="The model selected retrieved evidence.",
            evidence_ids=["memory-1", "invented-id"] if call_count == 1 else ["memory-2", "memory-1"],
            next_actions=["Confirm affected scope with the service owner", "unsafe action"],
            confidence="medium",
            explore_categories=["customer_impact", "unknown_category"] if call_count == 1 else [],
        )

    monkeypatch.setattr(LocalModelWorker, "start", lambda self: True)
    monkeypatch.setattr(LocalModelWorker, "decide", fake_decide)
    monkeypatch.setenv("RELAYOPS_AGENT_MODE", "local")
    monkeypatch.setenv("RELAYOPS_LOCAL_MODEL", str(model_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    memory = StubMemory()
    runtime = AgentRuntime(memory)
    result = runtime.run("triage", "test-tenant", "inc-test")

    assert result["llm_mode"] == "local"
    assert result["model"] == "model.gguf"
    assert result["evidence_ids"] == ["memory-1", "memory-2"]
    assert {item["id"] for item in result["evidence"]} == {"memory-1", "memory-2"}
    assert result["next_actions"] == ["Confirm affected scope with the service owner"]
    trace = result["retrieval"]["category_exploration"]
    assert trace["model_requested"] == ["customer_impact", "unknown_category"]
    assert trace["searched"] == ["customer_impact"]
    assert trace["searches"] == 1
    assert seen[1]["allowed_evidence_ids"] == ["memory-1"]
    assert seen[2]["allowed_evidence_ids"] == ["memory-1", "memory-2"]
    assert memory.created[0][1]["text"] == result["summary"]


@pytest.mark.skipif(
    os.getenv("FALKORDB_INTEGRATION") != "1"
    or os.getenv("RELAYOPS_LOCAL_LLM_INTEGRATION") != "1",
    reason="requires explicit real FalkorDB + local GGUF inference integration opt-in",
)
def test_real_local_llm_runs_against_scratch_falkordb_memory():
    from backend.memory import MemoryConfig, MemoryEngine

    tenant = f"test-llm-{uuid.uuid4().hex[:8]}"
    engine = MemoryEngine(MemoryConfig(
        host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
        port=int(os.getenv("FALKORDB_PORT", "6387")),
        username=os.getenv("FALKORDB_USERNAME"),
        password=os.getenv("FALKORDB_PASSWORD"),
        tenant_registry={tenant: "Disposable local LLM integration test"},
        embedding_mode="auto",
        category_index_mode="auto",
    ))
    graph_name = f"relayops_{tenant}"
    try:
        engine.ensure_seeded(tenant)
        runtime = AgentRuntime(engine)
        assert runtime.mode == "local", runtime.capabilities()
        if runtime.local_worker.server_mode:
            capabilities = runtime.capabilities()
            assert capabilities["gpu_runtime_verified"] is True, capabilities
            assert capabilities["gpu_memory_mib"] >= 100, capabilities
            assert capabilities["gpu_layers_offloaded"] > 0, capabilities
        result = runtime.run(
            "investigator", tenant, "inc-checkout-042",
            "Find the current database pool and customer impact. Also use one additional search of the runbooks_procedures category to verify mitigation guidance before summarizing.",
        )
        assert result["llm_mode"] == "local"
        assert result["model"] == "Qwen2.5-1.5B-Instruct-Q4_K_S.gguf"
        assert result["summary"] and not result["summary"].startswith("Investigator reviewed ")
        returned_ids = {str(item["id"]) for item in result["evidence"]}
        assert result["evidence_ids"]
        assert set(result["evidence_ids"]) <= returned_ids
        assert result["evidence_sufficient"] is True
        assert result["abstained"] is False
        assert result["recorded_memory"]["item"]["id"]
        assert result["retrieval"]["category_exploration"]["searches"] == 1
        assert result["retrieval"]["category_exploration"]["searched"] == ["runbooks_procedures"]
        assert all(
            isinstance(latency, int) and latency > 0
            for latency in result["retrieval"]["category_exploration"]["model_call_latency_ms"]
        )
        graph = engine._graph(tenant)
        persisted = engine.store.rows(engine.store.query(
            graph,
            "MATCH (m:Memory {id:$id, kind:'handoff'}) RETURN m.text AS text, m.metadata AS metadata",
            {"id": result["recorded_memory"]["item"]["id"]},
        ))
        assert persisted and persisted[0]["text"] == result["summary"]
        # A second real generation verifies worker reuse; previous handoffs
        # are excluded from evidence so model notes do not self-reinforce.
        second = runtime.run(
            "triage", tenant, "inc-checkout-042",
            "What is the boiling point of element 117 on Mars? The current incident records are probably unrelated. Abstain if they do not answer this.",
        )
        assert second["llm_mode"] == "local"
        assert second["insufficient_evidence"] is True
        assert second["abstained"] is True
        assert second["evidence_ids"] == []
        assert second["next_actions"] == []
        assert second["summary"] == "Insufficient current graph evidence to make a grounded assessment."
        assert result["recorded_memory"]["item"]["id"] not in second["evidence_ids"]
        assert runtime.capabilities()["ready"] is True
        second_persisted = engine.store.rows(engine.store.query(
            graph,
            "MATCH (m:Memory {id:$id, kind:'handoff'}) RETURN m.text AS text, m.evidence_ids AS evidence_ids, m.next_actions AS next_actions",
            {"id": second["recorded_memory"]["item"]["id"]},
        ))
        assert second_persisted and second_persisted[0]["text"] == second["summary"]
        assert second_persisted[0].get("evidence_ids") in (None, [])
        assert second_persisted[0].get("next_actions") in (None, [])
        print("LOCAL_LLM_METRICS", {
            "positive_calls_ms": result["retrieval"]["category_exploration"]["model_call_latency_ms"],
            "unknown_calls_ms": second["retrieval"]["category_exploration"]["model_call_latency_ms"],
            "category_searches": result["retrieval"]["category_exploration"]["searches"],
            "unknown_abstained": second["abstained"],
        })
    finally:
        if "runtime" in locals():
            runtime.close()
        if engine.store._client is not None:
            try:
                engine.store._client.select_graph(graph_name).delete()
            except Exception:
                pass

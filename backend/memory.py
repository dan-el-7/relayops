"""Graph-native incident memory for RelayOps.

FalkorDB is the source of truth for tenant state, memories, provenance, and
supersession. When it cannot connect, the engine reports `db_mode=demo`; it
never labels local scoring as a FalkorDB result.
"""

from __future__ import annotations

import os
import re
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingProvider, cosine_similarity
from .graph_store import GraphConnectionConfig, GraphStore, GraphUnavailable, UnknownTenant, tenant_graph_name


@dataclass
class MemoryConfig:
    host: str = "127.0.0.1"
    port: int = 6379
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    db_mode: str = "auto"
    embedding_mode: str = "auto"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    tenant_registry: dict[str, str] | list[str] | tuple[str, ...] = field(
        default_factory=lambda: {"acme-retail": "Acme Retail", "northstar-canary": "Northstar Canary"}
    )
    category_index_mode: str = "auto"
    max_categories: int = 8
    model_cache: str | None = None
    remote_embedding_api_key: str | None = None
    remote_embedding_base_url: str | None = None
    remote_embedding_model: str | None = None

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        tenants = [part.strip() for part in os.getenv("RELAYOPS_TENANTS", "acme-retail,northstar-canary").split(",") if part.strip()]
        return cls(
            host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
            port=int(os.getenv("FALKORDB_PORT", "6379")),
            username=os.getenv("FALKORDB_USERNAME") or None,
            password=os.getenv("FALKORDB_PASSWORD") or None,
            ssl=os.getenv("FALKORDB_SSL", "false").strip().lower() in {"1", "true", "yes", "on"},
            db_mode=os.getenv("RELAYOPS_DB_MODE", "auto"),
            embedding_mode=os.getenv("RELAYOPS_EMBEDDING_MODE", "auto"),
            embedding_model=os.getenv("RELAYOPS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            tenant_registry={tenant: tenant.replace("-", " ").title() for tenant in tenants},
            category_index_mode=os.getenv("ROUTER_INDEX_MODE", os.getenv("RELAYOPS_INDEX_MODE", "auto")),
            model_cache=os.getenv("RELAYOPS_MODEL_CACHE"),
            remote_embedding_api_key=os.getenv("RELAYOPS_EMBEDDING_API_KEY") if os.getenv("RELAYOPS_EMBEDDING_MODE") == "remote" else None,
            remote_embedding_base_url=os.getenv("RELAYOPS_EMBEDDING_BASE_URL"),
            remote_embedding_model=os.getenv("RELAYOPS_EMBEDDING_REMOTE_MODEL"),
        )

    @property
    def allowed_tenants(self) -> list[str]:
        if isinstance(self.tenant_registry, dict):
            return [str(key).strip().lower() for key in self.tenant_registry]
        return [str(item).strip().lower() for item in self.tenant_registry]

    def tenant_name(self, tenant_id: str) -> str:
        if isinstance(self.tenant_registry, dict):
            return str(self.tenant_registry.get(tenant_id, tenant_id))
        return tenant_id.replace("-", " ").title()


CATEGORIES: list[dict[str, str]] = [
    {"id": "incident_history", "name": "Incident history", "description": "Past incidents, symptoms, impact, timeline, and verified resolutions."},
    {"id": "service_health", "name": "Service health", "description": "Current service health, dependencies, ownership, and reliability signals."},
    {"id": "deployments_changes", "name": "Deployments and changes", "description": "Deployments, rollbacks, configuration changes, and change-related regressions."},
    {"id": "runbooks_procedures", "name": "Runbooks and procedures", "description": "Approved operational procedures, diagnostic steps, and recovery instructions."},
    {"id": "customer_impact", "name": "Customer impact", "description": "Affected customers, checkout impact, business outcomes, and communications."},
    {"id": "metrics_alerts", "name": "Metrics and alerts", "description": "Metrics, traces, logs, alerts, thresholds, and diagnostic evidence."},
    {"id": "ownership_escalations", "name": "Ownership and escalation", "description": "Service owners, responders, escalation paths, and decision authority."},
    {"id": "handoff_tasks", "name": "Handoffs and tasks", "description": "Open actions, handoffs, blockers, task claims, and next steps between responders."},
]

CATEGORY_BY_ID = {item["id"]: item for item in CATEGORIES}
CATEGORY_LABELS = {item["id"]: "Cat" + "".join(part.title() for part in item["id"].split("_")) for item in CATEGORIES}
ALL_CATEGORY_LABELS = ":".join(CATEGORY_LABELS.values())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,50}", str(key)):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
        elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
            result[str(key)] = value
    return result


def _node_value(value: Any) -> Any:
    if hasattr(value, "properties"):
        return dict(value.properties)
    if isinstance(value, dict):
        return {key: _node_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_node_value(item) for item in value]
    return value


_BM25_TOKEN = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", re.IGNORECASE)


def _bm25_rank(query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Okapi BM25 (k1=1.5, b=0.75) over the bounded current-memory corpus."""
    tokenized = [[token.casefold() for token in _BM25_TOKEN.findall(str(doc.get("text") or ""))] for doc in documents]
    query_terms = list(dict.fromkeys(token.casefold() for token in _BM25_TOKEN.findall(query)))
    if not query_terms or not documents:
        return []
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    document_terms = [set(tokens) for tokens in tokenized]
    document_frequency = {
        term: sum(term in terms for terms in document_terms) for term in query_terms
    }
    ranked: list[dict[str, Any]] = []
    k1, b = 1.5, 0.75
    for document, tokens, length in zip(documents, tokenized, lengths):
        frequencies = {term: tokens.count(term) for term in query_terms}
        score = 0.0
        for term, frequency in frequencies.items():
            if not frequency:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            norm = frequency + k1 * (1.0 - b + b * length / average_length) if average_length else frequency + k1
            score += idf * frequency * (k1 + 1.0) / norm
        if score > 0:
            ranked.append({**document, "bm25_score": score})
    ranked.sort(key=lambda item: (-item["bm25_score"], str(item.get("id") or "")))
    return ranked


def _reciprocal_rank_fusion(
    vector_ranking: list[dict[str, Any]], bm25_ranking: list[dict[str, Any]], rank_constant: int = 60
) -> list[dict[str, Any]]:
    """Fuse independent ranked lists without comparing incompatible raw scores."""
    fused: dict[str, dict[str, Any]] = {}
    for source, ranking in (("hnsw", vector_ranking), ("bm25", bm25_ranking)):
        for rank, item in enumerate(ranking, start=1):
            memory_id = str(item.get("id") or "")
            if not memory_id:
                continue
            entry = fused.setdefault(memory_id, {})
            entry["rrf_score"] = float(entry.get("rrf_score", 0.0)) + 1.0 / (rank_constant + rank)
            entry.setdefault("match_sources", []).append(source)
            entry.setdefault("retrieval_ranks", {})[source] = rank
            for key, value in item.items():
                if key == "score":
                    if source == "hnsw" and (key not in entry or float(value) > float(entry[key])):
                        entry[key] = value
                elif key not in {"rrf_score", "match_sources", "retrieval_ranks"} and key not in entry:
                    entry[key] = value
    for item in fused.values():
        item["retrieval_source"] = item["match_sources"][0] if len(item["match_sources"]) == 1 else "hybrid"
    return sorted(fused.values(), key=lambda item: (-float(item.get("rrf_score", 0.0)), str(item.get("id") or "")))


class MemoryEngine:
    def __init__(self, config: MemoryConfig | dict[str, Any] | None = None):
        if config is None:
            config = MemoryConfig.from_env()
        elif isinstance(config, dict):
            config = MemoryConfig(**config)
        self.config = config
        self.store = GraphStore(GraphConnectionConfig(
            host=config.host,
            port=config.port,
            username=config.username,
            password=config.password,
            ssl=config.ssl,
        ))
        cache_dir = config.model_cache or str(Path(__file__).resolve().parents[1] / ".model-cache")
        self.embeddings = EmbeddingProvider(
            mode=config.embedding_mode,
            model_name=config.embedding_model,
            cache_dir=cache_dir,
            remote_api_key=config.remote_embedding_api_key,
            remote_base_url=config.remote_embedding_base_url,
            remote_model=config.remote_embedding_model,
        )
        self._category_vectors: dict[str, list[float]] = {}
        if self.embeddings.ready:
            for category in CATEGORIES:
                self._category_vectors[category["id"]] = self.embeddings.embed(category["description"])
        self._demo_graphs: dict[str, dict[str, Any]] = {}
        self._initialized: set[str] = set()
        self._last_error: str | None = None

    def _validate_tenant(self, tenant_id: str) -> str:
        tenant = str(tenant_id or "").strip().lower()
        # Also validates that the graph name cannot be client-controlled.
        tenant_graph_name(tenant, self.config.allowed_tenants)
        return tenant

    def _graph(self, tenant_id: str):
        tenant = self._validate_tenant(tenant_id)
        return self.store.graph(tenant, self.config.allowed_tenants)

    def _vector_indexes_enabled(self) -> bool:
        mode = str(self.config.category_index_mode or "auto").strip().lower()
        return mode not in {"0", "false", "off", "global", "disabled"}

    def health(self) -> dict[str, Any]:
        try:
            self._graph(self.config.allowed_tenants[0])
            db_mode = "falkordb"
            self._last_error = None
        except (GraphUnavailable, UnknownTenant) as exc:
            db_mode = "demo"
            self._last_error = str(exc)
        return {
            "status": "ok" if db_mode == "falkordb" else "degraded",
            "db_mode": db_mode,
            "graph_ready": db_mode == "falkordb",
            "embedding_ready": self.embeddings.ready and self.embeddings.semantic,
            "embedding_mode": self.embeddings.mode,
            "router_index_mode": "category" if self._vector_indexes_enabled() else "global",
            "error": self._last_error,
        }

    def ensure_seeded(self, tenant_id: str) -> dict[str, Any]:
        tenant = self._validate_tenant(tenant_id)
        if tenant in self._initialized:
            return {"tenant_id": tenant, "db_mode": "falkordb"}
        try:
            graph = self._graph(tenant)
            if not self.embeddings.ready:
                raise RuntimeError(self.embeddings.error or "embedding provider unavailable")
            self._migrate_embedding_space(graph)
            # Seed the 8 stable categories and embeddings before creating their indexes.
            for category in CATEGORIES[: self.config.max_categories]:
                vector = self._category_vectors[category["id"]]
                self.store.query(
                    graph,
                    "MERGE (c:Category {id:$id}) "
                    "SET c.name=$name, c.description=$description, c.description_embedding=vecf32($embedding), c.active=true",
                    {**category, "embedding": vector},
                )
            self._seed_operational_graph(graph, tenant)
            self.store.ensure_vector_index(graph, "CurrentMemory", "embedding", self.embeddings.dimension)
            if self._vector_indexes_enabled():
                for category in CATEGORIES[: self.config.max_categories]:
                    self.store.ensure_vector_index(
                        graph, CATEGORY_LABELS[category["id"]], "embedding", self.embeddings.dimension
                    )
            self._initialized.add(tenant)
            self._last_error = None
            return {"tenant_id": tenant, "db_mode": "falkordb"}
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            # No in-memory store is reported as FalkorDB. Demo state is only created
            # explicitly by the API's degraded/demo presentation layer.
            raise

    def _embedding_signature(self) -> str:
        model = self.config.remote_embedding_model if self.embeddings.mode == "remote" else self.config.embedding_model
        return f"{self.embeddings.mode}:{model}:{self.embeddings.dimension}"

    def _migrate_embedding_space(self, graph) -> None:
        """Never mix vectors from different models/dimensions in one HNSW index."""
        signature = self._embedding_signature()
        schema_rows = self.store.rows(self.store.query(
            graph, "MATCH (s:RelayOpsSchema {id:'memory_embedding'}) RETURN s.signature AS signature"
        ))
        old_signature = schema_rows[0].get("signature") if schema_rows else None
        if old_signature == signature:
            return
        memory_rows = self.store.rows(self.store.query(
            graph,
            "MATCH (m:Memory) OPTIONAL MATCH (m)-[:IN_CATEGORY]->(c:Category) "
            "RETURN m.id AS id, m.text AS text, m.status AS status, collect(c.id) AS category_ids",
        ))
        existing_indexes = self.store.index_state(graph)
        indexed_labels = {"CurrentMemory", *CATEGORY_LABELS.values()}
        for index in existing_indexes:
            label = index.get("label")
            props = index.get("properties") or []
            if label in indexed_labels and "embedding" in props:
                self.store.query(graph, f"DROP VECTOR INDEX FOR (m:{label}) ON (m.embedding)")
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            remaining = [
                index for index in self.store.index_state(graph)
                if index.get("label") in indexed_labels and "embedding" in (index.get("properties") or [])
            ]
            if not remaining:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Timed out waiting for old FalkorDB vector indexes to drop")

        for row in memory_rows:
            memory_id = str(row.get("id") or "")
            text = str(row.get("text") or "")
            if not memory_id:
                continue
            vector = self.embeddings.embed(text)
            self.store.query(
                graph,
                "MATCH (m:Memory {id:$id}) SET m.embedding=vecf32($embedding) "
                f"REMOVE m:CurrentMemory:{ALL_CATEGORY_LABELS}",
                {"id": memory_id, "embedding": vector},
            )
            if row.get("status") == "active":
                labels = [CATEGORY_LABELS[item] for item in row.get("category_ids") or [] if item in CATEGORY_LABELS]
                label_suffix = (":" + ":".join(labels)) if labels else ""
                self.store.query(
                    graph,
                    f"MATCH (m:Memory {{id:$id}}) SET m:CurrentMemory{label_suffix}",
                    {"id": memory_id},
                )
        for category in CATEGORIES[: self.config.max_categories]:
            vector = self._category_vectors[category["id"]]
            self.store.query(
                graph,
                "MATCH (c:Category {id:$id}) SET c.description_embedding=vecf32($embedding)",
                {"id": category["id"], "embedding": vector},
            )
        self.store.query(
            graph,
            "MERGE (s:RelayOpsSchema {id:'memory_embedding'}) SET s.signature=$signature, s.updated_at=$now",
            {"signature": signature, "now": _now()},
        )

    def _seed_operational_graph(self, graph, tenant: str) -> None:
        now = _now()
        northstar = tenant == "northstar-canary"
        graph_data = {
            "incident": {
                "id": "inc-checkout-042", "title": "Checkout latency spike after pool exhaustion",
                "status": "mitigating", "severity": "SEV-2", "started_at": "2026-10-15T06:45:00+00:00",
                "summary": "Checkout API latency rose after database connection saturation; mitigation is in progress.",
                "tenant_id": tenant,
            },
            "service": {
                "id": "svc-checkout-api", "name": "Checkout API", "owner": "Payments Platform",
                "status": "degraded", "tenant_id": tenant,
            },
            "runbook": {
                "id": "runbook-checkout-pool-recovery", "name": "Checkout connection pool recovery",
                "status": "pending", "version": 1, "owner": "Payments Platform",
                "steps": ["Confirm pool saturation", "Drain the oldest workers", "Verify checkout p95 returns below 700 ms"],
                "tenant_id": tenant,
            },
            "task": {
                "id": "task-042-db-pool", "title": "Drain stale checkout workers and validate recovery",
                "status": "open", "owner": "", "version": 0, "incident_id": "inc-checkout-042",
                "service_id": "svc-checkout-api", "next_action": "Drain stale workers, then watch connection utilization for 10 minutes.",
                "blockers": ["Need a second responder to confirm the customer-impact window."],
                "updated_at": now, "is_stale": False, "tenant_id": tenant,
            },
            "handoff": {
                "id": "handoff-042-primary", "owner": "Maya Chen", "status": "active",
                "next_actions": ["Drain stale workers", "Confirm recovery with the database on-call"],
                "blockers": ["Customer-impact window is still being confirmed"],
                "source_ref": "incident://inc-checkout-042/handoff-1", "updated_at": now,
                "incident_id": "inc-checkout-042", "is_stale": False, "tenant_id": tenant,
            },
        }
        for label, props in graph_data.items():
            props_literal = ", ".join(f"{key}: ${key}" for key in props)
            self.store.query(
                graph,
                f"MERGE (n:{label.title()} {{id:$id}}) ON CREATE SET n += {{{props_literal}}}",
                props,
            )
        if northstar:
            # Make the second allowlisted graph visibly distinct while keeping
            # all content inside its own tenant graph. These descriptive fields
            # are fixture metadata; mutable operational status/version remain
            # untouched across restarts.
            self.store.query(
                graph,
                "MATCH (i:Incident {id:'inc-checkout-042'}), (s:Service {id:'svc-checkout-api'}), "
                "(t:Task {id:'task-042-db-pool'}), (h:Handoff {id:'handoff-042-primary'}) "
                "SET i.title=$incident_title, i.summary=$incident_summary, "
                "s.name=$service_name, t.title=$task_title, h.next_actions=$handoff_actions",
                {
                    "incident_title": "Northstar canary: Checkout latency spike after pool exhaustion",
                    "incident_summary": "Synthetic Northstar canary incident. Use this isolated dataset to validate tenant routing and handoff behavior.",
                    "service_name": "Northstar Checkout API",
                    "task_title": "Canary: Drain checkout workers and verify the recovery window",
                    "handoff_actions": ["Drain two canary workers", "Confirm Northstar checkout p95 remains below 700 ms"],
                },
            )
        self.store.query(
            graph,
            "MATCH (i:Incident {id:'inc-checkout-042'}), (s:Service {id:'svc-checkout-api'}), "
            "(r:Runbook {id:'runbook-checkout-pool-recovery'}), (t:Task {id:'task-042-db-pool'}), "
            "(h:Handoff {id:'handoff-042-primary'}) "
            "MERGE (i)-[:AFFECTS]->(s) MERGE (i)-[:HAS_TASK]->(t) "
            "MERGE (i)-[:USES_RUNBOOK]->(r) MERGE (i)-[:HAS_HANDOFF]->(h) "
            "MERGE (t)-[:HANDOFF_FOR]->(h)"
        )
        # Stable seed memory IDs make refreshes idempotent; only active memories get
        # CurrentMemory/category labels, while old fact revisions remain traversable.
        seed = [
            {
                "id": "mem-incident-summary", "kind": "episode",
                "text": "At 06:45 UTC checkout p95 crossed 1.8 s while database pool utilization reached 100%. The incident remains in mitigation.",
                "source_ref": "incident://inc-checkout-042/timeline/1", "created_at": "2026-10-15T06:50:00+00:00",
                "valid_from": "2026-10-15T06:45:00+00:00", "valid_to": None, "status": "active", "confidence": 0.98,
                "category_ids": ["incident_history", "metrics_alerts"], "service_id": "svc-checkout-api", "incident_id": "inc-checkout-042", "slot_key": None,
            },
            {
                "id": "fact-db-pool-current", "kind": "fact",
                "text": "Checkout API database pool max_connections is 20 after the approved change at 07:05 UTC.",
                "source_ref": "change://db-pool/CHG-8821", "created_at": "2026-10-15T07:05:00+00:00",
                "valid_from": "2026-10-15T07:05:00+00:00", "valid_to": None, "status": "active", "confidence": 0.96,
                "category_ids": ["service_health", "deployments_changes"], "service_id": "svc-checkout-api", "incident_id": "inc-checkout-042", "slot_key": "checkout.db_pool.max_connections",
            },
            {
                "id": "fact-db-pool-v1", "kind": "fact",
                "text": "Checkout API database pool max_connections was 12 before the approved change at 07:05 UTC.",
                "source_ref": "change://db-pool/CHG-8817", "created_at": "2026-10-14T19:20:00+00:00",
                "valid_from": "2026-10-14T19:20:00+00:00", "valid_to": "2026-10-15T07:05:00+00:00", "status": "superseded", "confidence": 0.94,
                "category_ids": ["service_health", "deployments_changes"], "service_id": "svc-checkout-api", "incident_id": "inc-checkout-042", "slot_key": "checkout.db_pool.max_connections",
            },
            {
                "id": "fact-circuit-breaker", "kind": "fact",
                "text": "The checkout payment circuit breaker is enabled with a 3 second open timeout; do not disable it during this incident.",
                "source_ref": "runbook://checkout/circuit-breaker", "created_at": "2026-10-15T06:58:00+00:00",
                "valid_from": "2026-10-15T06:58:00+00:00", "valid_to": None, "status": "active", "confidence": 0.99,
                "category_ids": ["runbooks_procedures", "service_health"], "service_id": "svc-checkout-api", "incident_id": "inc-checkout-042", "slot_key": "checkout.circuit_breaker.enabled",
            },
            {
                "id": "procedure-drain-workers", "kind": "procedure",
                "text": "Drain stale checkout workers in batches of two, wait 30 seconds, and confirm pool utilization declines before continuing.",
                "source_ref": "runbook://checkout/pool-recovery#step-2", "created_at": "2026-10-15T07:00:00+00:00",
                "valid_from": "2026-10-15T07:00:00+00:00", "valid_to": None, "status": "active", "confidence": 0.97,
                "category_ids": ["runbooks_procedures", "handoff_tasks"], "service_id": "svc-checkout-api", "incident_id": "inc-checkout-042", "slot_key": None,
            },
            {
                "id": "handoff-042-memory", "kind": "handoff",
                "text": "Maya Chen owns checkout mitigation. Next: drain stale workers, verify p95 below 700 ms for 10 minutes, then confirm the impact window with database on-call.",
                "source_ref": "incident://inc-checkout-042/handoff-1", "created_at": now,
                "valid_from": now, "valid_to": None, "status": "active", "confidence": 0.95,
                "category_ids": ["handoff_tasks", "ownership_escalations"], "service_id": "svc-checkout-api", "incident_id": "inc-checkout-042", "slot_key": None,
            },
            {
                "id": "episode-customer-impact", "kind": "episode",
                "text": "About 8% of checkout attempts saw elevated latency in the first 12 minutes; payment capture errors remain below the alert threshold.",
                "source_ref": "metrics://checkout/2026-10-15T06:45Z", "created_at": "2026-10-15T07:02:00+00:00",
                "valid_from": "2026-10-15T06:45:00+00:00", "valid_to": None, "status": "active", "confidence": 0.91,
                "category_ids": ["customer_impact", "metrics_alerts"], "service_id": "svc-checkout-api", "incident_id": "inc-checkout-042", "slot_key": None,
            },
        ]
        if northstar:
            seed.append({
                "id": "northstar-canary-impact", "kind": "episode",
                "text": "Northstar canary sample: 11% of checkout attempts saw elevated latency during the synthetic pool saturation window; payment capture errors stayed below the alert threshold.",
                "source_ref": "canary://northstar/metrics/2026-10-15T06:45Z", "created_at": now,
                "valid_from": now, "valid_to": None, "status": "active", "confidence": 0.90,
                "category_ids": ["customer_impact", "incident_history"], "service_id": "svc-checkout-api",
                "incident_id": "inc-checkout-042", "slot_key": None,
            })
        for item in seed:
            item["embedding"] = self.embeddings.embed(item["text"])
            self._upsert_seed_memory(graph, item)
        # Connect memory nodes to the operational graph through explicit evidence edges.
        self.store.query(
            graph,
            "MATCH (m:Memory), (i:Incident {id:m.incident_id}), (s:Service {id:m.service_id}) "
            "MERGE (m)-[:ABOUT]->(i) MERGE (m)-[:ABOUT]->(s)"
        )
        self.store.query(
            graph,
            "MATCH (new:Memory {id:'fact-db-pool-current'}), (old:Memory {id:'fact-db-pool-v1'}) "
            "MERGE (new)-[:SUPERSEDES]->(old)"
        )
        # ActionEvent foreign keys are persisted provenance; materialize them as
        # explicit graph edges so the graph explorer can show the real trail.
        self.store.query(
            graph,
            "MATCH (t:Task), (e:ActionEvent) WHERE e.task_id=t.id "
            "MERGE (t)-[:RECORDED_ACTION]->(e)"
        )
        self.store.query(
            graph,
            "MATCH (r:Runbook), (e:ActionEvent) WHERE e.runbook_id=r.id "
            "MERGE (r)-[:RECORDED_ACTION]->(e)"
        )

    def _upsert_seed_memory(self, graph, item: dict[str, Any]) -> None:
        category_ids = item["category_ids"]
        labels = [CATEGORY_LABELS[item_id] for item_id in category_ids] if item["status"] == "active" else []
        labels_clause = ":" + ":".join(["Memory"] + (["CurrentMemory"] if item["status"] == "active" else []) + labels)
        query = (
            "MERGE (m:Memory {id:$id}) "
            f"ON CREATE SET m:CurrentMemory{':' + ':'.join(labels) if labels else ''}, "
            "m.kind=$kind, m.text=$text, m.embedding=vecf32($embedding), m.source_ref=$source_ref, "
            "m.created_at=$created_at, m.valid_from=$valid_from, m.valid_to=$valid_to, m.status=$status, "
            "m.confidence=$confidence, m.service_id=$service_id, m.incident_id=$incident_id, m.slot_key=$slot_key "
            "WITH m UNWIND $category_ids AS category_id "
            "MATCH (c:Category {id:category_id}) MERGE (m)-[:IN_CATEGORY]->(c)"
        )
        self.store.query(graph, query, item)
        if item["status"] != "active":
            self.store.query(
                graph,
                f"MATCH (m:Memory {{id:$id}}) REMOVE m:CurrentMemory:{ALL_CATEGORY_LABELS}",
                {"id": item["id"]},
            )

    def state(self, tenant_id: str) -> dict[str, Any]:
        tenant = self._validate_tenant(tenant_id)
        self.ensure_seeded(tenant)
        graph = self._graph(tenant)
        incident_rows = self.store.rows(self.store.query(graph, "MATCH (i:Incident {id:'inc-checkout-042'}) RETURN i"))
        service_rows = self.store.rows(self.store.query(graph, "MATCH (s:Service {id:'svc-checkout-api'}) RETURN s"))
        runbook_rows = self.store.rows(self.store.query(graph, "MATCH (r:Runbook {id:'runbook-checkout-pool-recovery'}) RETURN r"))
        task_rows = self.store.rows(self.store.query(graph, "MATCH (t:Task {id:'task-042-db-pool'}) RETURN t"))
        handoff_rows = self.store.rows(self.store.query(graph, "MATCH (h:Handoff {id:'handoff-042-primary'}) RETURN h"))
        categories = self.categories(tenant)
        timeline_rows = self.store.rows(self.store.query(
            graph,
            "MATCH (m:Memory) WHERE m.incident_id='inc-checkout-042' "
            "RETURN m.id AS id, m.kind AS kind, m.text AS text, m.source_ref AS source_ref, "
            "m.created_at AS timestamp, m.status AS status, m.valid_from AS valid_from, m.valid_to AS valid_to "
            "ORDER BY m.created_at ASC"
        ))
        event_rows = self.store.rows(self.store.query(
            graph,
            "MATCH (e:ActionEvent) WHERE e.incident_id='inc-checkout-042' "
            "OR e.runbook_id='runbook-checkout-pool-recovery' "
            "RETURN e.id AS id, e.created_at AS timestamp, e.action AS action, e.status AS status, "
            "e.actor AS actor, e.task_id AS task_id, e.runbook_id AS runbook_id, e.incident_id AS incident_id"
        ))
        for event in event_rows:
            is_runbook = bool(event.get("runbook_id"))
            event["kind"] = "runbook_approval" if is_runbook else "task_action"
            subject = event.get("runbook_id") or event.get("task_id") or "incident"
            event["text"] = f"{event.get('action') or 'updated'} on {subject} by {event.get('actor') or 'operator'}"
            event["source_ref"] = f"runbook://{subject}" if is_runbook else f"task://{subject}"
            timeline_rows.append(event)
        timeline_rows.sort(key=lambda item: str(item.get("timestamp") or ""))
        facts = self.store.rows(self.store.query(
            graph,
            # State exposes the complete append-only fact audit trail. Retrieval
            # remains CurrentMemory-only and excludes superseded revisions.
            "MATCH (m:Memory {kind:'fact'}) "
            "RETURN m.id AS id, m.text AS text, m.source_ref AS source_ref, m.created_at AS created_at, "
            "m.valid_from AS valid_from, m.valid_to AS valid_to, m.confidence AS confidence, "
            "m.slot_key AS slot_key, m.status AS status "
            "ORDER BY m.created_at DESC"
        ))
        incident = _node_value(incident_rows[0].get("i")) if incident_rows else {}
        service = _node_value(service_rows[0].get("s")) if service_rows else {}
        runbook = _node_value(runbook_rows[0].get("r")) if runbook_rows else {}
        task = _node_value(task_rows[0].get("t")) if task_rows else {}
        handoff = _node_value(handoff_rows[0].get("h")) if handoff_rows else {}
        incident.update({"service": service, "runbook": runbook})
        agents = [
            {"id": "triage", "name": "Triage", "status": "ready", "focus": "Summarize impact and current signal"},
            {"id": "investigator", "name": "Investigator", "status": "ready", "focus": "Trace causes and corroborating evidence"},
            {"id": "resolver", "name": "Resolver", "status": "ready", "focus": "Prepare safe next actions"},
        ]
        return {
            "tenant_id": tenant,
            "backend": self.health(),
            "categories": categories,
            "incident": incident,
            "timeline": timeline_rows,
            "facts": facts,
            "task": task,
            "handoff": handoff,
            "agents": agents,
        }

    def categories(self, tenant_id: str) -> list[dict[str, Any]]:
        graph = self._graph(self._validate_tenant(tenant_id))
        result = self.store.query(
            graph,
            "MATCH (c:Category) OPTIONAL MATCH (m:Memory)-[:IN_CATEGORY]->(c) "
            "WHERE m.status='active' RETURN c.id AS id, c.name AS name, c.description AS description, "
            "count(DISTINCT m) AS memory_count ORDER BY c.name"
        )
        return self.store.rows(result)

    def _route(
        self,
        tenant_id: str,
        query: str,
        category_ids: list[str] | None,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        categories = self.categories(tenant_id)
        vector = query_vector if query_vector is not None else self.embeddings.embed(query)
        if category_ids:
            if not isinstance(category_ids, list) or len(category_ids) > 3:
                raise ValueError("category_ids must be a list of at most three category IDs")
            available_ids = {str(category.get("id")) for category in categories}
            selected: list[dict[str, Any]] = []
            seen: set[str] = set()
            for category_id in category_ids:
                category_id = str(category_id)
                if category_id not in CATEGORY_BY_ID or category_id not in self._category_vectors or category_id not in available_ids:
                    raise ValueError(f"Unknown category ID: {category_id}")
                if category_id not in seen:
                    selected.append({"id": category_id, "score": cosine_similarity(vector, self._category_vectors[category_id])})
                    seen.add(category_id)
            return selected[:3]
        ranked = [
            {"id": category["id"], "score": cosine_similarity(vector, self._category_vectors[category["id"]])}
            for category in categories
            if category.get("id") in self._category_vectors
        ]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        selected = ranked[:2]
        if len(ranked) > 2 and selected and selected[0]["score"] - selected[-1]["score"] < 0.06:
            selected.append(ranked[2])
        return selected

    def search(
        self,
        tenant_id: str,
        query: str,
        k: int = 8,
        mode: str = "hierarchical_fallback",
        category_ids: list[str] | None = None,
        candidate_cap: int | None = None,
    ) -> dict[str, Any]:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        tenant = self._validate_tenant(tenant_id)
        self.ensure_seeded(tenant)
        graph = self._graph(tenant)
        k = max(1, min(int(k), 30))
        if candidate_cap is not None:
            candidate_cap = max(1, min(int(candidate_cap), 256))
        if mode not in {"global", "hierarchical", "hierarchical_fallback"}:
            raise ValueError("mode must be global, hierarchical, or hierarchical_fallback")
        embed_start = time.perf_counter()
        query_vector = self.embeddings.embed(query)
        routed = self._route(tenant, query, category_ids, query_vector) if mode != "global" else []
        timings["route_ms"] = round((time.perf_counter() - embed_start) * 1000, 2)
        search_start = time.perf_counter()
        results: dict[str, dict[str, Any]] = {}
        fallback_used = False
        searched: list[dict[str, Any]] = []
        candidate_count = 0
        if mode == "global":
            limit = candidate_cap if candidate_cap is not None else max(k * 3, 12)
            searched = self.store.vector_search(graph, "CurrentMemory", "embedding", query_vector, limit)
            candidate_count += len(searched)
        elif self._vector_indexes_enabled():
            per_category_cap = (
                max(1, candidate_cap // max(1, len(routed)))
                if candidate_cap is not None else max(k * 2, 8)
            )
            for category in routed:
                label = CATEGORY_LABELS[category["id"]]
                rows = self.store.vector_search(graph, label, "embedding", query_vector, per_category_cap)
                candidate_count += len(rows)
                searched.extend(rows)
        else:
            # Honest fallback when category indexes are disabled: obtain global ANN
            # candidates then filter by their IN_CATEGORY edges below.
            limit = candidate_cap if candidate_cap is not None else max(k * 8, 40)
            searched = self.store.vector_search(graph, "CurrentMemory", "embedding", query_vector, limit)
            candidate_count += len(searched)
            fallback_used = True
        # BM25 ranks a bounded result window of tenant-local current memories.
        # The Cypher match/order can still scan/sort more rows before its LIMIT.
        bm25_cap = candidate_cap if candidate_cap is not None else min(max(k * 8, 40), 256)
        category_scope = [item["id"] for item in routed] if mode != "global" else None
        bm25_corpus = self.store.current_memories(
            graph,
            category_scope,
            bm25_cap,
        )
        bm25_ranked = _bm25_rank(query, bm25_corpus)
        expansion = self.store.expand_memories(
            graph,
            list(dict.fromkeys(
                [str(item["id"]) for item in searched if item.get("id")]
                + [str(item["id"]) for item in bm25_ranked if item.get("id")]
            )),
        )
        bm25_results: dict[str, dict[str, Any]] = {}
        for item in bm25_ranked:
            memory_id = str(item.get("id") or "")
            if not memory_id or item.get("status") != "active":
                continue
            enriched = dict(item)
            enriched.update(expansion.get(memory_id, {}))
            enriched["category_ids"] = enriched.get("category_ids") or []
            if category_scope and not set(enriched["category_ids"]).intersection(category_scope):
                continue
            bm25_results[memory_id] = enriched
        bm25_ranking = list(bm25_results.values())
        for item in searched:
            memory_id = str(item.get("id") or "")
            if not memory_id:
                continue
            enriched = dict(item)
            enriched.update(expansion.get(memory_id, {}))
            enriched["category_ids"] = enriched.get("category_ids") or []
            if mode != "global" and not self._vector_indexes_enabled():
                if not set(enriched["category_ids"]).intersection({r["id"] for r in routed}):
                    continue
            # Only active evidence is eligible, even if an older graph was seeded
            # before the searchable-label maintenance rule existed.
            if enriched.get("status") != "active":
                continue
            prior = results.get(memory_id)
            if prior is None or float(enriched.get("score", 0)) > float(prior.get("score", 0)):
                results[memory_id] = enriched
        vector_ranking = sorted(
            results.values(),
            key=lambda item: (-float(item.get("score", 0)), str(item.get("id") or "")),
        )
        if mode == "hierarchical_fallback" and self._vector_indexes_enabled():
            best_route_score = routed[0]["score"] if routed else 0
            if len(vector_ranking) < min(k, 4) or best_route_score < 0.18:
                fallback_used = True
                global_cap = candidate_cap if candidate_cap is not None else max(k * 3, 16)
                global_rows = self.store.vector_search(graph, "CurrentMemory", "embedding", query_vector, global_cap)
                candidate_count += len(global_rows)
                extra = self.store.expand_memories(graph, [str(item["id"]) for item in global_rows if item.get("id")])
                for item in global_rows:
                    memory_id = str(item.get("id") or "")
                    if not memory_id:
                        continue
                    enriched = dict(item)
                    enriched.update(extra.get(memory_id, {}))
                    enriched["category_ids"] = enriched.get("category_ids") or []
                    if enriched.get("status") == "active":
                        existing = results.get(memory_id)
                        if existing is None or float(enriched.get("score", 0)) > float(existing.get("score", 0)):
                            results[memory_id] = enriched
                vector_ranking = sorted(
                    results.values(),
                    key=lambda item: (-float(item.get("score", 0)), str(item.get("id") or "")),
                )
        candidates = _reciprocal_rank_fusion(vector_ranking, bm25_ranking)
        timings["retrieval_ms"] = round((time.perf_counter() - search_start) * 1000, 2)
        timings["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return {
            "query": query,
            "routed_categories": routed,
            "results": candidates[:k],
            "fallback_used": fallback_used,
            "db_mode": "falkordb",
            "embedding_mode": self.embeddings.mode,
            "search_strategy": "category_hnsw_bm25_rrf" if mode != "global" and self._vector_indexes_enabled() else "global_hnsw_bm25_rrf",
            "bm25_candidates_returned": len(bm25_ranked),
            "fusion": {"method": "reciprocal_rank_fusion", "rank_constant": 60, "lists": ["hnsw", "bm25"]},
            # Count rows returned by vector-index calls; this is not the number
            # of distance comparisons performed inside FalkorDB's HNSW index.
            "leaf_candidates_returned": candidate_count,
            "timings_ms": timings,
        }

    def create_memory(self, tenant_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        tenant = self._validate_tenant(tenant_id)
        self.ensure_seeded(tenant)
        graph = self._graph(tenant)
        kind = str(payload.get("kind", "episode"))
        if kind not in {"episode", "fact", "procedure", "handoff"}:
            raise ValueError("kind must be episode, fact, procedure, or handoff")
        text = str(payload.get("text", "")).strip()
        if len(text) < 3 or len(text) > 10000:
            raise ValueError("text must be between 3 and 10000 characters")
        vector = self.embeddings.embed(text)
        category_ids = self._resolve_categories(tenant, text, payload.get("category_ids"))
        now = _now()
        memory_id = str(payload.get("id") or f"mem-{uuid.uuid4().hex[:12]}")
        slot_key = payload.get("slot_key") if kind == "fact" else None
        labels = [CATEGORY_LABELS[item] for item in category_ids]
        labels_clause = ":".join(["Memory", "CurrentMemory", *labels])
        params: dict[str, Any] = {
            "id": memory_id, "kind": kind, "text": text, "embedding": vector,
            "source_ref": payload.get("source_ref"), "created_at": now, "valid_from": now,
            "valid_to": None, "status": "active", "confidence": float(payload.get("confidence", 0.85)),
            "service_id": payload.get("service_id"), "incident_id": payload.get("incident_id"),
            "slot_key": slot_key, "category_ids": category_ids,
            "metadata": _safe_metadata(payload.get("metadata")),
        }
        query = (
            "OPTIONAL MATCH (old:Memory {slot_key:$slot_key, kind:'fact', status:'active'}) "
            "WITH collect(old) AS previous "
            f"CREATE (m:{labels_clause} {{id:$id}}) "
            "SET m.kind=$kind, m.text=$text, m.embedding=vecf32($embedding), m.source_ref=$source_ref, "
            "m.created_at=$created_at, m.valid_from=$valid_from, m.valid_to=$valid_to, m.status=$status, "
            "m.confidence=$confidence, m.service_id=$service_id, m.incident_id=$incident_id, m.slot_key=$slot_key "
            "SET m += $metadata "
            f"FOREACH (old IN previous | SET old.status='superseded', old.valid_to=$created_at REMOVE old:CurrentMemory:{ALL_CATEGORY_LABELS}) "
            "FOREACH (old IN previous | CREATE (m)-[:SUPERSEDES]->(old)) "
            "WITH m UNWIND $category_ids AS category_id MATCH (c:Category {id:category_id}) "
            "CREATE (m)-[:IN_CATEGORY]->(c) "
            "RETURN m.id AS id, m.kind AS kind, m.text AS text, m.source_ref AS source_ref, "
            "m.status AS status, m.created_at AS created_at, m.valid_from AS valid_from, "
            "m.valid_to AS valid_to, m.confidence AS confidence, m.slot_key AS slot_key"
        )
        rows = self.store.rows(self.store.query(graph, query, params))
        if not rows:
            raise RuntimeError("FalkorDB did not create the memory")
        row = rows[0]
        row["category_ids"] = category_ids
        row["metadata"] = params["metadata"]
        return {"item": row, "db_mode": "falkordb"}

    def _resolve_categories(self, tenant: str, text: str, category_ids: list[str] | None) -> list[str]:
        if category_ids:
            chosen: list[str] = []
            for category_id in category_ids:
                category_id = str(category_id)
                if category_id not in CATEGORY_BY_ID:
                    raise ValueError(f"Unknown category ID: {category_id}")
                if category_id not in chosen:
                    chosen.append(category_id)
            return chosen[:2]
        return [item["id"] for item in self._route(tenant, text, None)[:2]]

    def correct_fact(self, tenant_id: str, fact_id: str, corrected_text: str, source_ref: str | None = None) -> dict[str, Any]:
        tenant = self._validate_tenant(tenant_id)
        self.ensure_seeded(tenant)
        graph = self._graph(tenant)
        old_rows = self.store.rows(self.store.query(
            graph,
            "MATCH (old:Memory {id:$fact_id, kind:'fact', status:'active'}) "
            "OPTIONAL MATCH (old)-[:IN_CATEGORY]->(c:Category) "
            "RETURN old.id AS id, old.text AS text, old.source_ref AS source_ref, old.slot_key AS slot_key, "
            "old.service_id AS service_id, old.incident_id AS incident_id, collect(c.id) AS category_ids",
            {"fact_id": fact_id},
        ))
        if not old_rows or not old_rows[0].get("id"):
            raise ValueError("Active fact not found")
        old = old_rows[0]
        result = self.create_memory(tenant, {
            "kind": "fact", "text": corrected_text, "source_ref": source_ref or old.get("source_ref"),
            "category_ids": old.get("category_ids") or [], "service_id": old.get("service_id"),
            "incident_id": old.get("incident_id"), "slot_key": old.get("slot_key"),
        })
        item = result["item"]
        self.store.query(
            graph,
            "MATCH (old:Memory {id:$fact_id}) "
            "OPTIONAL MATCH (t:Task {incident_id:old.incident_id}) "
            "OPTIONAL MATCH (h:Handoff {incident_id:old.incident_id}) "
            "SET t.is_stale=true, t.updated_at=$now, h.is_stale=true, h.updated_at=$now "
            "RETURN old.id AS id",
            {"fact_id": fact_id, "now": _now()},
        )
        item["is_stale"] = True
        return {"current_fact": item, "superseded_fact_id": fact_id, "db_mode": "falkordb"}

    def _task_mutation(self, tenant_id: str, task_id: str, query: str, params: dict[str, Any]) -> dict[str, Any]:
        tenant = self._validate_tenant(tenant_id)
        self.ensure_seeded(tenant)
        graph = self._graph(tenant)
        rows = self.store.rows(self.store.query(graph, query, params))
        if not rows:
            raise ValueError("Task not found or state changed; refresh and retry")
        return {"state": rows[0], "db_mode": "falkordb"}

    def claim_task(self, tenant_id: str, task_id: str, actor: str = "operator", expected_version: int | None = None) -> dict[str, Any]:
        now = _now()
        return self._task_mutation(
            tenant_id, task_id,
            "MATCH (t:Task {id:$task_id}) "
            "WHERE t.status IN ['open','available'] AND ($expected_version IS NULL OR t.version=$expected_version) "
            "SET t.status='claimed', t.owner=$actor, t.version=t.version+1, t.updated_at=$now "
            "RETURN t.id AS id, t.title AS title, t.status AS status, t.owner AS owner, t.version AS version, t.updated_at AS updated_at",
            {"task_id": task_id, "actor": actor, "expected_version": expected_version, "now": now},
        )

    def refresh_task(self, tenant_id: str, task_id: str, actor: str = "operator") -> dict[str, Any]:
        now = _now()
        return self._task_mutation(
            tenant_id, task_id,
            "MATCH (t:Task {id:$task_id}) WHERE t.status <> 'done' "
            "OPTIONAL MATCH (m:Memory {incident_id:t.incident_id, kind:'fact', status:'active'}) "
            "WITH t, collect(m.id) AS current_fact_ids "
            "SET t.updated_at=$now, t.version=t.version+1, t.is_stale=false, t.fact_snapshot_ids=current_fact_ids "
            "WITH t, current_fact_ids OPTIONAL MATCH (h:Handoff {incident_id:t.incident_id}) "
            "SET h.is_stale=false, h.updated_at=$now, h.fact_snapshot_ids=current_fact_ids "
            "RETURN t.id AS id, t.title AS title, t.status AS status, t.owner AS owner, t.version AS version, "
            "t.updated_at AS updated_at, t.is_stale AS is_stale, t.fact_snapshot_ids AS fact_snapshot_ids",
            {"task_id": task_id, "actor": actor, "now": now},
        )

    def simulate_action(
        self, tenant_id: str, task_id: str, action: str, actor: str = "operator",
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        allowed = {"start", "complete", "resolve", "block", "unblock"}
        if action not in allowed:
            raise ValueError(f"action must be one of {', '.join(sorted(allowed))}")
        status = {"start": "in_progress", "complete": "done", "resolve": "done", "block": "blocked", "unblock": "in_progress"}[action]
        now = _now()
        return self._task_mutation(
            tenant_id, task_id,
            "MATCH (t:Task {id:$task_id}) "
            "WHERE coalesce(t.is_stale,false)=false "
            "AND (t.status IN ['claimed','in_progress'] OR (t.status='done' AND t.status=$status)) "
            "AND t.owner=$actor "
            "AND ($expected_version IS NULL OR t.version=$expected_version OR t.status=$status) "
            "AND (t.status <> 'done' OR t.status=$status) "
            "WITH t, t.version AS previous_version, t.status=$status AS unchanged "
            "SET t.status=$status, "
            "t.updated_at=CASE WHEN unchanged THEN t.updated_at ELSE $now END, "
            "t.version=CASE WHEN unchanged THEN previous_version ELSE previous_version+1 END "
            "WITH t, unchanged OPTIONAL MATCH (i:Incident {id:t.incident_id}) "
            "FOREACH (_ IN CASE WHEN $status='done' THEN [1] ELSE [] END | SET i.status='monitoring') "
            "WITH t, i, unchanged "
            "MERGE (e:ActionEvent {task_id:t.id, version:t.version}) "
            "ON CREATE SET e.id=t.id + ':action:' + toString(t.version), e.incident_id=t.incident_id, e.action=$action, "
            "e.status=$status, e.actor=$actor, e.created_at=$now "
            "MERGE (t)-[:RECORDED_ACTION]->(e) "
            "RETURN t.id AS id, t.title AS title, t.status AS status, t.owner AS owner, t.version AS version, "
            "t.updated_at AS updated_at, i.status AS incident_status, unchanged AS idempotent, e.id AS event_id",
            {"task_id": task_id, "actor": actor, "action": action, "status": status,
             "expected_version": expected_version, "now": now},
        )

    def approve_runbook(self, tenant_id: str, runbook_id: str, actor: str = "operator", expected_version: int | None = None) -> dict[str, Any]:
        tenant = self._validate_tenant(tenant_id)
        self.ensure_seeded(tenant)
        graph = self._graph(tenant)
        rows = self.store.rows(self.store.query(
            graph,
            "MATCH (r:Runbook {id:$runbook_id}) "
            "WHERE (r.status IN ['draft','pending'] AND ($expected_version IS NULL OR r.version=$expected_version)) "
            "OR r.status='approved' "
            "WITH r, r.status='approved' AS unchanged "
            "SET r.status='approved', r.owner=coalesce(r.owner,$actor), "
            "r.version=CASE WHEN unchanged THEN r.version ELSE r.version+1 END, "
            "r.updated_at=CASE WHEN unchanged THEN r.updated_at ELSE $now END "
            "WITH r, unchanged MERGE (e:ActionEvent {runbook_id:r.id, action:'approve'}) "
            "ON CREATE SET e.id=r.id + ':approved', e.incident_id=r.incident_id, e.actor=$actor, e.version=r.version, e.created_at=$now "
            "MERGE (r)-[:RECORDED_ACTION]->(e) "
            "RETURN r.id AS id, r.name AS name, r.status AS status, r.version AS version, "
            "r.updated_at AS updated_at, unchanged AS idempotent, e.id AS event_id",
            {"runbook_id": runbook_id, "actor": actor, "expected_version": expected_version, "now": _now()},
        ))
        if not rows:
            raise ValueError("Runbook not found or version changed; refresh and retry")
        return {"state": rows[0], "db_mode": "falkordb"}

    def graph_snapshot(self, tenant_id: str) -> dict[str, Any]:
        tenant = self._validate_tenant(tenant_id)
        self.ensure_seeded(tenant)
        graph = self._graph(tenant)
        nodes = self.store.rows(self.store.query(graph, "MATCH (n) RETURN labels(n) AS labels, count(*) AS count"))
        edges = self.store.rows(self.store.query(graph, "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count"))
        graph_nodes = self.store.rows(self.store.query(
            graph,
            "MATCH (n) WITH n ORDER BY CASE WHEN n:Incident OR n:Service OR n:Runbook OR n:Task "
            "OR n:Handoff OR n:ActionEvent THEN 0 ELSE 1 END, id(n) "
            "RETURN id(n) AS graph_id, labels(n) AS labels, properties(n) AS properties LIMIT 80",
        ))
        graph_edges = self.store.rows(self.store.query(
            graph,
            "MATCH (a)-[r]->(b) WITH a, r, b ORDER BY CASE type(r) "
            "WHEN 'AFFECTS' THEN 0 WHEN 'HAS_TASK' THEN 1 WHEN 'USES_RUNBOOK' THEN 2 "
            "WHEN 'HAS_HANDOFF' THEN 3 WHEN 'HANDOFF_FOR' THEN 4 WHEN 'RECORDED_ACTION' THEN 5 "
            "WHEN 'SUPERSEDES' THEN 6 WHEN 'IN_CATEGORY' THEN 7 WHEN 'ABOUT' THEN 8 ELSE 9 END, id(r) "
            "RETURN id(a) AS source, id(b) AS target, type(r) AS type LIMIT 160",
        ))
        # Keep the UI payload intentionally small and never send vector embeddings.
        safe_nodes = []
        display_fields = {
            "id", "name", "title", "kind", "text", "summary", "status", "owner",
            "source_ref", "created_at", "updated_at", "valid_from", "valid_to",
            "incident_id", "service_id", "runbook_id", "task_id", "handoff_id",
            "slot_key", "version", "confidence", "is_stale", "stale", "next_action",
            "next_actions", "blockers", "steps", "description", "action", "actor",
        }
        for row in graph_nodes:
            props = row.get("properties") or {}
            safe_nodes.append({
                "graph_id": row.get("graph_id"),
                "labels": row.get("labels") or [],
                "properties": {key: value for key, value in props.items() if key in display_fields},
            })
        return {
            "tenant_id": tenant, "nodes": nodes, "edges": edges,
            "graph_nodes": safe_nodes, "graph_edges": graph_edges,
            "limits": {"nodes": 80, "edges": 160}, "db_mode": "falkordb",
        }

    def evaluate(self, tenant_id: str) -> dict[str, Any]:
        """Small seeded-corpus retrieval check; reports counts, not broad accuracy claims."""
        tenant = self._validate_tenant(tenant_id)
        self.ensure_seeded(tenant)
        graph = self._graph(tenant)
        current_pool_rows = self.store.rows(self.store.query(
            graph,
            "MATCH (m:Memory {kind:'fact', status:'active', slot_key:$slot_key}) "
            "RETURN m.id AS id ORDER BY m.valid_from DESC, m.created_at DESC LIMIT 1",
            {"slot_key": "checkout.db_pool.max_connections"},
        ))
        current_pool_fact_id = current_pool_rows[0].get("id") if current_pool_rows else None
        checks = [
            ("pool setting current", "What is the current checkout DB max_connections setting?", current_pool_fact_id),
            ("safe worker drain", "How should we safely drain stale checkout workers?", "procedure-drain-workers"),
            ("handoff owner", "Who owns mitigation and what is the next action?", "handoff-042-memory"),
        ]
        results = []
        for label, query, expected_id in checks:
            response = self.search(tenant, query, k=5, mode="hierarchical_fallback")
            ids = [item["id"] for item in response["results"]]
            results.append({
                "check": label,
                "expected_id": expected_id,
                "rank": ids.index(expected_id) + 1 if expected_id and expected_id in ids else None,
            })
        return {
            "tenant_id": tenant,
            "db_mode": "falkordb",
            "embedding_mode": self.embeddings.mode,
            "checks": results,
            "matched": sum(item["rank"] is not None for item in results),
            "total": len(results),
            "note": "Small seeded-corpus smoke check only; not a general retrieval benchmark.",
        }

    def reset_demo_tenant(self, tenant_id: str) -> dict[str, Any]:
        tenant = self._validate_tenant(tenant_id)
        if tenant != "acme-retail":
            raise ValueError("Only the isolated acme-retail demo tenant can be reset")
        if tenant not in self.config.allowed_tenants:
            raise UnknownTenant(tenant)
        if self.health()["db_mode"] != "falkordb":
            raise GraphUnavailable("Cannot reset a tenant while FalkorDB is unavailable")
        if self.store._client is None:
            raise GraphUnavailable("FalkorDB client unavailable")
        # The installed FalkorDB SDK exposes deletion on the selected Graph, not
        # on the FalkorDB client. Keep this constrained to the demo tenant.
        graph = self._graph(tenant)
        graph.delete()
        self._initialized.discard(tenant)
        return self.ensure_seeded(tenant)

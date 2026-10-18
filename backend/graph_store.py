"""FalkorDB client and graph-scoped index helpers for RelayOps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


class GraphUnavailable(RuntimeError):
    pass


class UnknownTenant(ValueError):
    pass


def tenant_graph_name(tenant_id: str, allowed_tenants: Iterable[str]) -> str:
    """Resolve only server-configured tenant IDs to a safe FalkorDB graph name."""
    tenant = str(tenant_id or "").strip().lower()
    allowed = {str(item).strip().lower() for item in allowed_tenants}
    if tenant not in allowed:
        raise UnknownTenant(f"Unknown tenant: {tenant_id}")
    slug = re.sub(r"[^a-z0-9_]+", "_", tenant).strip("_")
    if not slug or len(slug) > 48:
        raise UnknownTenant("Invalid tenant ID")
    return f"relayops_{slug}"


@dataclass(frozen=True)
class GraphConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 6379
    username: str | None = None
    password: str | None = None
    ssl: bool = False


class GraphStore:
    def __init__(self, config: GraphConnectionConfig):
        self.config = config
        self._client = None
        self._import_error: str | None = None
        try:
            from falkordb import FalkorDB

            self._client = FalkorDB(
                host=config.host,
                port=int(config.port),
                username=config.username,
                password=config.password,
                ssl=config.ssl,
                socket_timeout=3,
            )
        except Exception as exc:
            self._import_error = f"{type(exc).__name__}: {exc}"

    @property
    def import_error(self) -> str | None:
        return self._import_error

    def graph(self, tenant_id: str, allowed_tenants: Iterable[str]):
        if self._client is None:
            raise GraphUnavailable(self._import_error or "FalkorDB client is unavailable")
        name = tenant_graph_name(tenant_id, allowed_tenants)
        try:
            graph = self._client.select_graph(name)
            # A cheap Cypher probe confirms both Redis reachability and FalkorDB module health.
            graph.query("RETURN 1 AS ready")
            return graph
        except Exception as exc:
            raise GraphUnavailable(f"FalkorDB connection failed: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def query(graph, cypher: str, params: dict[str, Any] | None = None):
        try:
            if params:
                return graph.query(cypher, params=params)
            return graph.query(cypher)
        except Exception as exc:
            raise GraphUnavailable(f"FalkorDB query failed: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def rows(result) -> list[dict[str, Any]]:
        headers: list[str] = []
        for header in getattr(result, "header", []) or []:
            if isinstance(header, (list, tuple)) and len(header) > 1:
                headers.append(str(header[1]))
            else:
                headers.append(str(header))
        records: list[dict[str, Any]] = []
        for row in getattr(result, "result_set", []) or []:
            if isinstance(row, dict):
                records.append(dict(row))
                continue
            records.append({headers[index] if index < len(headers) else str(index): value for index, value in enumerate(row)})
        return records

    def index_state(self, graph) -> list[dict[str, Any]]:
        result = self.query(graph, "CALL db.indexes()")
        return self.rows(result)

    def ensure_vector_index(self, graph, label: str, property_name: str, dimension: int) -> dict[str, Any]:
        # Labels/property names originate only from the fixed server-side taxonomy.
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label):
            raise ValueError("Unsafe vector index label")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", property_name):
            raise ValueError("Unsafe vector index property")
        existing = self.index_state(graph)
        for index in existing:
            if index.get("label") == label and property_name in (index.get("properties") or []):
                return index
        cypher = (
            f"CREATE VECTOR INDEX FOR (m:{label}) ON (m.{property_name}) "
            f"OPTIONS {{dimension:{int(dimension)}, similarityFunction:'cosine', M:16, efConstruction:100, efRuntime:32}}"
        )
        self.query(graph, cypher)
        return {"label": label, "properties": [property_name], "status": "CREATED"}

    def vector_search(
        self,
        graph,
        label: str,
        property_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label):
            raise ValueError("Unsafe vector index label")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", property_name):
            raise ValueError("Unsafe vector index property")
        # label/property are a server-side allowlist, while query contents remain parameters.
        cypher = (
            f"CALL db.idx.vector.queryNodes('{label}', '{property_name}', {max(1, int(limit))}, "
            "vecf32($query_vector)) YIELD node, score "
            "RETURN node.id AS id, node.kind AS kind, node.text AS text, node.source_ref AS source_ref, "
            "node.status AS status, node.created_at AS created_at, node.valid_from AS valid_from, "
            "node.valid_to AS valid_to, node.confidence AS confidence, node.service_id AS service_id, "
            "node.incident_id AS incident_id, node.slot_key AS slot_key, score AS distance "
            "ORDER BY distance ASC"
        )
        rows = self.rows(self.query(graph, cypher, {"query_vector": query_vector}))
        for row in rows:
            distance = float(row.pop("distance", 1.0) or 0.0)
            # FalkorDB 4.22.1's observed score is cosine distance (0 identical, 1 orthogonal).
            # Convert to the conventional cosine-similarity range used by the API.
            row["score"] = 1.0 - distance
        return rows

    def expand_memories(self, graph, memory_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not memory_ids:
            return {}
        result = self.query(
            graph,
            "UNWIND $ids AS memory_id "
            "MATCH (m:Memory {id: memory_id}) "
            "OPTIONAL MATCH (m)-[:IN_CATEGORY]->(c:Category) "
            "OPTIONAL MATCH (m)-[r]->(related) "
            "WHERE type(r) IN ['ABOUT', 'EVIDENCE_FOR', 'USES_RUNBOOK', 'SUPERSEDES', 'HANDOFF_FOR'] "
            "RETURN m.id AS id, collect(DISTINCT c.id) AS category_ids, "
            "collect(DISTINCT CASE WHEN related IS NULL THEN NULL ELSE {type:type(r), id:related.id, label:labels(related)} END) AS relations",
            {"ids": memory_ids},
        )
        return {str(row["id"]): row for row in self.rows(result) if row.get("id") is not None}

    def current_memories(self, graph, category_ids: list[str] | None, limit: int) -> list[dict[str, Any]]:
        """Return a small, deterministic BM25 corpus from this tenant graph only."""
        if category_ids == []:
            return []
        params: dict[str, Any] = {"limit": max(1, int(limit))}
        category_filter = ""
        if category_ids:
            category_filter = (
                "MATCH (m)-[:IN_CATEGORY]->(c:Category) WHERE c.id IN $category_ids "
            )
            params["category_ids"] = category_ids
        query = (
            "MATCH (m:CurrentMemory:Memory {status:'active'}) "
            f"{category_filter}"
            "WITH DISTINCT m ORDER BY m.created_at DESC, m.id ASC LIMIT $limit "
            "OPTIONAL MATCH (m)-[:IN_CATEGORY]->(c:Category) "
            "RETURN m.id AS id, m.kind AS kind, m.text AS text, m.source_ref AS source_ref, "
            "m.status AS status, m.created_at AS created_at, m.valid_from AS valid_from, "
            "m.valid_to AS valid_to, m.confidence AS confidence, m.service_id AS service_id, "
            "m.incident_id AS incident_id, m.slot_key AS slot_key, collect(DISTINCT c.id) AS category_ids "
            "ORDER BY created_at DESC, id ASC"
        )
        return self.rows(self.query(graph, query, params))

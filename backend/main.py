"""RelayOps local API and static frontend server."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agents import AGENT_INFO, AgentRuntime, ProviderRateLimited
from .coordination import CoordinationService, VersionConflict
from .memory import MemoryConfig, MemoryEngine
from .schemas import (
    AgentRunRequest,
    DemoResetRequest,
    FactCorrectionRequest,
    MemoryCreateRequest,
    MemorySearchRequest,
    RunbookApprovalRequest,
    TaskActionRequest,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("relayops.api")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

TENANT_LABELS = {
    "acme-retail": "Acme Retail",
    "northstar-canary": "Northstar Canary",
}


def tenant_registry_from_env() -> dict[str, str]:
    configured = os.getenv("RELAYOPS_TENANTS", "acme-retail,northstar-canary")
    registry: dict[str, str] = {}
    for raw in configured.split(","):
        tenant_id = raw.strip().lower()
        if not tenant_id:
            continue
        registry[tenant_id] = TENANT_LABELS.get(tenant_id, tenant_id.replace("-", " ").title())
    if "acme-retail" not in registry:
        registry["acme-retail"] = TENANT_LABELS["acme-retail"]
    return registry


TENANT_REGISTRY = tenant_registry_from_env()


def demo_reset_enabled() -> bool:
    """Keep the destructive seeded-graph reset available only outside production."""
    return os.getenv("VERCEL_ENV", "").strip().lower() != "production"


def build_memory_engine() -> MemoryEngine:
    config = MemoryConfig(
        host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
        port=int(os.getenv("FALKORDB_PORT", "6387")),
        username=os.getenv("FALKORDB_USERNAME") or None,
        password=os.getenv("FALKORDB_PASSWORD") or None,
        ssl=os.getenv("FALKORDB_SSL", "false").strip().lower() in {"1", "true", "yes", "on"},
        db_mode=os.getenv("RELAYOPS_DB_MODE", "auto"),
        embedding_mode=os.getenv("RELAYOPS_EMBEDDING_MODE", "auto"),
        embedding_model=os.getenv("RELAYOPS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        tenant_registry=TENANT_REGISTRY,
        category_index_mode=os.getenv("RELAYOPS_INDEX_MODE", "auto"),
        max_categories=int(os.getenv("RELAYOPS_MAX_CATEGORIES", "8")),
    )
    return MemoryEngine(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    memory = build_memory_engine()
    app.state.memory = memory
    app.state.agents = AgentRuntime(memory)
    app.state.coordination = CoordinationService(memory)
    # Seed only server-allowlisted isolated tenant graphs. The engine's seed is
    # idempotent, so restarts preserve ongoing task/fact state.
    try:
        for tenant_id in TENANT_REGISTRY:
            memory.ensure_seeded(tenant_id)
    except Exception as exc:
        logger.exception("RelayOps startup could not verify/seed FalkorDB")
        app.state.startup_error = f"{type(exc).__name__}: {exc}"
    else:
        app.state.startup_error = None
    yield


app = FastAPI(
    title="RelayOps",
    description="Incident memory, handoffs, and safe coordination backed by FalkorDB.",
    version="0.1.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "RELAYOPS_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Demo-Tenant"],
    allow_credentials=False,
)


def tenant_id_or_404(tenant_id: str) -> str:
    normalized = (tenant_id or "").strip().lower()
    if normalized not in TENANT_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown demo tenant. Select one from /api/tenants.")
    return normalized


def services(request: Request) -> tuple[MemoryEngine, AgentRuntime, CoordinationService]:
    memory = getattr(request.app.state, "memory", None)
    agents = getattr(request.app.state, "agents", None)
    coordination = getattr(request.app.state, "coordination", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory engine is not initialized.")
    return memory, agents, coordination


def call_store(callable_, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except ProviderRateLimited as exc:
        headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
        raise HTTPException(status_code=429, detail=str(exc), headers=headers) from exc
    except VersionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    memory, _, _ = services(request)
    try:
        result = memory.health()
    except Exception as exc:
        logger.warning("Memory health check failed: %s", type(exc).__name__)
        result = {"db_mode": "unavailable", "graph_ready": False, "embedding_ready": False}
    db_connected = bool(result.get("db_connected", result.get("graph_ready", False)))
    embedding_ready = bool(result.get("embedding_ready", False))
    return {
        "status": "ok" if db_connected and embedding_ready else "degraded",
        "graph_ready": db_connected,
        "embedding_ready": embedding_ready,
    }


@app.get("/api/tenants")
def list_tenants() -> dict[str, list[dict[str, str]]]:
    return {"items": [{"id": key, "name": value} for key, value in TENANT_REGISTRY.items()]}


@app.get("/api/state")
def get_state(request: Request, tenant_id: str = Query(default="acme-retail")) -> dict[str, Any]:
    memory, agents, _ = services(request)
    tenant_id = tenant_id_or_404(tenant_id)
    state = call_store(memory.state, tenant_id)
    if not isinstance(state, dict):
        raise HTTPException(status_code=503, detail="Memory engine returned an invalid state snapshot.")
    backend_health = call_store(memory.health)
    llm = agents.capabilities()
    incident = state.get("incident") or {}
    task = state.get("task") or {}
    handoff = state.get("handoff") or {}
    raw_runbook = state.get("runbook") or incident.get("runbook") or {}
    service = incident.get("service") or {}
    state["incident"] = {
        **incident,
        "service_name": service.get("name") or incident.get("service_name"),
        "service_id": service.get("id") or incident.get("service_id"),
    }
    state["timeline"] = [
        {
            **item,
            "at": item.get("timestamp") or item.get("created_at"),
            "label": item.get("label") or str(item.get("kind", "memory")).replace("_", " ").title(),
            "detail": item.get("text", ""),
        }
        for item in state.get("timeline", [])
    ]
    state["facts"] = [
        {
            **item,
            "key": item.get("slot_key") or "incident memory",
            "value": item.get("text", ""),
            "status": item.get("status") or ("superseded" if item.get("valid_to") else "active"),
            "observed_at": item.get("created_at"),
        }
        for item in state.get("facts", [])
    ]
    state["task"] = {
        **task,
        "summary": task.get("summary") or task.get("next_action", ""),
        "next_action": task.get("next_action") or task.get("summary", ""),
        "blocked_by": task.get("blocked_by") or task.get("blockers", []),
    } if task else None
    state["handoff"] = {
        **handoff,
        "title": handoff.get("title") or "Primary incident handoff",
        "from_agent": handoff.get("from_agent") or handoff.get("owner"),
        "to_agent": handoff.get("to_agent") or "Resolver",
        "summary": handoff.get("summary") or "Shared incident context for the next responder.",
        "evidence": handoff.get("evidence") or [],
        "runbook_id": raw_runbook.get("id"),
    } if handoff else None
    state["runbook"] = {
        **raw_runbook,
        "title": raw_runbook.get("title") or raw_runbook.get("name"),
        "approval_status": raw_runbook.get("approval_status") or raw_runbook.get("status", "pending"),
    } if raw_runbook else None
    state["tenant_id"] = tenant_id
    state["backend"] = {
        **(state.get("backend") or {}),
        "mode": backend_health.get("db_mode", "unavailable"),
        "demo_reset_enabled": demo_reset_enabled(),
        "db_connected": bool(backend_health.get("db_connected", backend_health.get("graph_ready", False))),
        "embedding_provider": backend_health.get("embedding_provider", backend_health.get("embedding_mode", "unknown")),
        "embedding_ready": bool(backend_health.get("embedding_ready", False)),
        "llm_mode": llm["mode"],
        "llm_model": llm.get("model"),
        "llm_provider": llm.get("provider_name") or llm.get("provider"),
        "llm_endpoint_host": llm.get("endpoint_host"),
        "router_mode": backend_health.get("router_mode", backend_health.get("router_index_mode", "global")),
    }
    state["agents"] = []
    for agent_id, spec in AGENT_INFO.items():
        runs = [
            item for item in state.get("timeline", [])
            if str(item.get("source_ref", "")).startswith(f"relayops://agent/{agent_id}/")
        ]
        latest = max(runs, key=lambda item: str(item.get("at", "")), default=None)
        state["agents"].append({
            "id": agent_id,
            "name": spec["name"],
            "status": "complete" if latest else "ready",
            "last_run": latest.get("at") if latest else None,
            "capability": spec["capability"],
        })
    if state.get("handoff") and state.get("runbook"):
        state["handoff"].setdefault("runbook_id", state["runbook"].get("id"))
    return state


@app.get("/api/categories")
def list_categories(request: Request, tenant_id: str = Query(default="acme-retail")) -> dict[str, Any]:
    memory, _, _ = services(request)
    tenant_id = tenant_id_or_404(tenant_id)
    return {"items": call_store(memory.categories, tenant_id)}


@app.post("/api/memories/search")
def search_memories(body: MemorySearchRequest, request: Request) -> dict[str, Any]:
    memory, _, _ = services(request)
    tenant_id = tenant_id_or_404(body.tenant_id)
    return call_store(
        memory.search,
        tenant_id,
        body.query,
        k=body.k,
        mode=body.mode,
        category_ids=body.category_ids,
    )


@app.post("/api/memories")
def create_memory(body: MemoryCreateRequest, request: Request) -> dict[str, Any]:
    memory, _, _ = services(request)
    tenant_id = tenant_id_or_404(body.tenant_id)
    return call_store(memory.create_memory, tenant_id, body.model_dump(exclude={"tenant_id"}))


@app.post("/api/facts/correct")
def correct_fact(body: FactCorrectionRequest, request: Request) -> dict[str, Any]:
    memory, _, _ = services(request)
    tenant_id = tenant_id_or_404(body.tenant_id)
    return call_store(
        memory.correct_fact,
        tenant_id,
        body.fact_id,
        body.corrected_text,
        body.source_ref,
    )


@app.post("/api/agents/{agent_id}/run")
def run_agent(agent_id: str, body: AgentRunRequest, request: Request) -> dict[str, Any]:
    _, agents, _ = services(request)
    if agent_id not in AGENT_INFO:
        raise HTTPException(status_code=404, detail="Unknown RelayOps agent.")
    tenant_id = tenant_id_or_404(body.tenant_id)
    return call_store(agents.run, agent_id, tenant_id, body.incident_id, body.prompt)


def validate_task_in_tenant(memory: MemoryEngine, tenant_id: str, task_id: str) -> None:
    state = call_store(memory.state, tenant_id)
    task = state.get("task") or {}
    if str(task.get("id")) != task_id:
        raise HTTPException(status_code=404, detail="Task not found in this tenant graph.")


@app.post("/api/tasks/{task_id}/claim")
def claim_task(task_id: str, body: TaskActionRequest, request: Request) -> dict[str, Any]:
    memory, _, coordination = services(request)
    tenant_id = tenant_id_or_404(body.tenant_id)
    validate_task_in_tenant(memory, tenant_id, task_id)
    result = call_store(
        coordination.claim_task,
        tenant_id,
        task_id,
        body.actor,
        body.expected_version,
    )
    return result


@app.post("/api/tasks/{task_id}/refresh")
def refresh_task(task_id: str, body: TaskActionRequest, request: Request) -> dict[str, Any]:
    memory, _, coordination = services(request)
    tenant_id = tenant_id_or_404(body.tenant_id)
    validate_task_in_tenant(memory, tenant_id, task_id)
    result = call_store(coordination.refresh_task, tenant_id, task_id, body.actor)
    return result


@app.post("/api/tasks/{task_id}/simulate-action")
def simulate_action(task_id: str, body: TaskActionRequest, request: Request) -> dict[str, Any]:
    memory, _, coordination = services(request)
    tenant_id = tenant_id_or_404(body.tenant_id)
    validate_task_in_tenant(memory, tenant_id, task_id)
    result = call_store(
        coordination.simulate_action,
        tenant_id,
        task_id,
        body.actor,
        body.action,
        body.expected_version,
    )
    return {**result, "simulated_only": True}


@app.post("/api/runbooks/{runbook_id}/approve")
def approve_runbook(runbook_id: str, body: RunbookApprovalRequest, request: Request) -> dict[str, Any]:
    memory, _, coordination = services(request)
    tenant_id = tenant_id_or_404(body.tenant_id)
    state = call_store(memory.state, tenant_id)
    runbook = state.get("runbook") or (state.get("incident") or {}).get("runbook") or {}
    if str(runbook.get("id")) != runbook_id:
        raise HTTPException(status_code=404, detail="Runbook not found in this tenant graph.")
    result = call_store(
        coordination.approve_runbook,
        tenant_id,
        runbook_id,
        body.actor,
        body.expected_version,
    )
    return result


@app.get("/api/graph")
def graph_snapshot(request: Request, tenant_id: str = Query(default="acme-retail")) -> dict[str, Any]:
    memory, _, _ = services(request)
    tenant_id = tenant_id_or_404(tenant_id)
    return call_store(memory.graph_snapshot, tenant_id)


@app.get("/api/evaluation")
def evaluation(request: Request, tenant_id: str = Query(default="acme-retail")) -> dict[str, Any]:
    memory, _, _ = services(request)
    tenant_id = tenant_id_or_404(tenant_id)
    return call_store(memory.evaluate, tenant_id)


@app.get("/api/evaluation/concurrency")
def concurrency_evaluation() -> dict[str, Any]:
    """Return the latest saved concurrency report without querying FalkorDB."""
    report_dir = PROJECT_ROOT / "backend" / "evaluation"
    reports = sorted(
        report_dir.glob("plugged_in_concurrency_report_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        raise HTTPException(status_code=404, detail="No concurrency evaluation report is available.")
    try:
        payload = json.loads(reports[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Could not read latest concurrency report")
        raise HTTPException(status_code=503, detail="The latest concurrency report could not be read.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=503, detail="The latest concurrency report has an invalid format.")
    return payload


@app.post("/api/demo/reset")
def reset_demo(body: DemoResetRequest, request: Request) -> dict[str, Any]:
    if not demo_reset_enabled():
        raise HTTPException(status_code=403, detail="Demo reset is disabled in production.")
    memory, _, coordination = services(request)
    tenant_id = tenant_id_or_404(body.tenant_id)
    if body.confirmation != "RESET ISOLATED DEMO TENANT":
        raise HTTPException(status_code=400, detail="Explicit destructive confirmation is required.")
    result = call_store(coordination.reset_demo_tenant, tenant_id)
    return {"tenant_id": tenant_id, "reset": True, "db_mode": "falkordb", "state": result}


@app.get("/{asset_path:path}", include_in_schema=False)
def spa_fallback(asset_path: str):
    if asset_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    if not FRONTEND_DIST.exists():
        raise HTTPException(status_code=404, detail="Frontend bundle not built. Run the local frontend build.")
    candidate = (FRONTEND_DIST / asset_path).resolve()
    if candidate.is_file() and candidate.is_relative_to(FRONTEND_DIST.resolve()):
        return FileResponse(candidate)
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend bundle not found")

"""HTTP payloads for RelayOps' local coordination API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemorySearchRequest(StrictModel):
    tenant_id: str = "acme-retail"
    query: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=8, ge=1, le=30)
    mode: Literal["global", "hierarchical", "hierarchical_fallback"] = (
        "hierarchical_fallback"
    )
    category_ids: list[str] | None = None


class MemoryCreateRequest(StrictModel):
    tenant_id: str = "acme-retail"
    kind: Literal["episode", "fact", "procedure", "handoff"]
    text: str = Field(min_length=3, max_length=10000)
    source_ref: str | None = None
    category_ids: list[str] | None = None
    service_id: str | None = None
    incident_id: str | None = None
    slot_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FactCorrectionRequest(StrictModel):
    tenant_id: str = "acme-retail"
    fact_id: str = Field(min_length=1, max_length=200)
    corrected_text: str = Field(min_length=3, max_length=4000)
    source_ref: str | None = None


class AgentRunRequest(StrictModel):
    tenant_id: str = "acme-retail"
    incident_id: str = "inc-checkout-042"
    prompt: str | None = Field(default=None, max_length=2000)


class TaskActionRequest(StrictModel):
    tenant_id: str = "acme-retail"
    actor: str = "operator"
    expected_version: int | None = Field(default=None, ge=0)
    action: str | None = Field(default=None, max_length=100)


class RunbookApprovalRequest(StrictModel):
    tenant_id: str = "acme-retail"
    actor: str = "operator"
    expected_version: int | None = Field(default=None, ge=0)


class DemoResetRequest(StrictModel):
    tenant_id: str = "acme-retail"
    confirmation: Literal["RESET ISOLATED DEMO TENANT"]

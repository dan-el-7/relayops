"""Guardrails around state-changing demo operations."""

from __future__ import annotations


SAFE_SIMULATED_ACTIONS = {
    "restart-worker": "start",
    "scale-worker": "start",
    "rollback-deployment": "resolve",
    "invalidate-cache": "complete",
    "complete": "complete",
    "resolve": "resolve",
    "start": "start",
    "block": "block",
    "unblock": "unblock",
}


class VersionConflict(RuntimeError):
    pass


class CoordinationService:
    def __init__(self, memory_engine):
        self.memory = memory_engine

    def _current_version(self, tenant_id: str, key: str) -> int:
        state = self.memory.state(tenant_id)
        item = state.get(key) or {}
        if key == "runbook" and not item:
            # The frozen state contract permits the runbook to be nested under
            # incident; approve-runbook must still infer its real version when
            # older UI clients omit expected_version.
            item = (state.get("incident") or {}).get("runbook") or {}
        return int(item.get("version", 0))

    @staticmethod
    def _raise_if_conflict(result) -> None:
        if isinstance(result, dict) and (
            result.get("conflict")
            or result.get("status") == "conflict"
            or result.get("error_code") == "version_conflict"
        ):
            raise VersionConflict(str(result.get("message") or "State changed; refresh before retrying."))

    def claim_task(self, tenant_id: str, task_id: str, actor: str, expected_version: int | None):
        # Resolve the current version when older clients omit it. The store applies
        # this version check and claim in one graph query, so two clients cannot
        # both claim the same task from the same observed version.
        version = expected_version if expected_version is not None else self._current_version(tenant_id, "task")
        try:
            result = self.memory.claim_task(tenant_id, task_id, actor, version)
        except ValueError as exc:
            if "state changed" in str(exc).lower() or "version" in str(exc).lower():
                raise VersionConflict("Task changed or was already claimed; refresh before retrying.") from exc
            raise
        self._raise_if_conflict(result)
        return result

    def refresh_task(self, tenant_id: str, task_id: str, actor: str):
        return self.memory.refresh_task(tenant_id, task_id, actor)

    def simulate_action(
        self, tenant_id: str, task_id: str, actor: str, action: str | None,
        expected_version: int | None = None,
    ):
        action_key = (action or "complete").strip().lower()
        if action_key not in SAFE_SIMULATED_ACTIONS:
            raise ValueError(f"Unsupported simulated action: {action_key}")
        # The graph mutation checks this observed version atomically. `actor` is
        # a workflow label supplied by the caller, not an authenticated identity.
        version = expected_version if expected_version is not None else self._current_version(tenant_id, "task")
        result = self.memory.simulate_action(
            tenant_id, task_id, SAFE_SIMULATED_ACTIONS[action_key], actor, version
        )
        self._raise_if_conflict(result)
        return result

    def approve_runbook(self, tenant_id: str, runbook_id: str, actor: str, expected_version: int | None):
        version = expected_version if expected_version is not None else self._current_version(tenant_id, "runbook")
        result = self.memory.approve_runbook(tenant_id, runbook_id, actor, version)
        self._raise_if_conflict(result)
        return result

    def reset_demo_tenant(self, tenant_id: str):
        if tenant_id != "acme-retail":
            raise ValueError("Only the isolated acme-retail demo tenant can be reset.")
        reset = getattr(self.memory, "reset_demo_tenant", None)
        if not callable(reset):
            raise NotImplementedError("Demo reset is not supported by this memory engine.")
        return reset(tenant_id)

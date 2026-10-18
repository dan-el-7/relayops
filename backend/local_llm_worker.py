"""Private localhost worker for the optional CPU GGUF agent provider."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any

from llama_cpp import Llama

from .agents import AgentDecision
from .memory import CATEGORIES

MAX_REQUEST_BYTES = 256_000
_llm: Llama | None = None
_model_name = ""
_lock = threading.Lock()


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "RelayOpsLocalModel/1"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep health probes and requests out of logs; startup errors are handled
        # by Python's regular exception path in the parent process log.
        return

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._reply(404, {"error": "not found"})
            return
        self._reply(200, {"ready": _llm is not None, "model": _model_name})

    def do_POST(self) -> None:
        if self.path != "/decision":
            self._reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._reply(400, {"error": "invalid content length"})
            return
        if length < 1 or length > MAX_REQUEST_BYTES:
            self._reply(413, {"error": "request body size is outside the allowed range"})
            return
        try:
            request = json.loads(self.rfile.read(length))
            system = str(request["system"])
            payload = request["payload"]
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            decision = _infer(system, payload)
            self._reply(200, {"decision": decision.model_dump()})
        except Exception as exc:
            detail = " ".join(str(exc).split())[:240]
            print(f"local inference failed: {type(exc).__name__}: {detail}", flush=True)
            self._reply(500, {"error": f"{type(exc).__name__}: local model response could not be validated", "detail": detail})


def _schema(allowed_categories: list[str], allowed_actions: list[str], allowed_evidence: list[str]) -> dict[str, Any]:
    """Keep the native grammar small: avoid pathological maxLength repeats."""
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": 160},
            "evidence_ids": {"type": "array", "maxItems": 8, "items": {"type": "string", "enum": allowed_evidence}},
            "next_actions": {"type": "array", "maxItems": 5, "items": {"type": "string", "enum": allowed_actions}},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "insufficient_evidence": {"type": "boolean"},
            "explore_categories": {"type": "array", "maxItems": 3,
                "items": {"type": "string", "enum": allowed_categories},
            },
        },
        "required": ["summary", "evidence_ids", "next_actions", "confidence", "insufficient_evidence", "explore_categories"],
        "additionalProperties": False,
    }


def _infer(system: str, payload: dict[str, Any]) -> AgentDecision:
    if _llm is None:
        raise RuntimeError("model has not loaded")
    allowed_categories = [
        str(item.get("id")) for item in payload.get("available_categories", [])
        if isinstance(item, dict) and str(item.get("id")) in {item["id"] for item in CATEGORIES}
    ][:8]
    allowed_actions = [str(item) for item in payload.get("allowed_next_actions", [])][:5]
    allowed_evidence = [str(item) for item in payload.get("allowed_evidence_ids", [])][:12]
    schema = _schema(allowed_categories, allowed_actions, allowed_evidence)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
    ]
    with _lock:
        response = _llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
            temperature=0.0,
            top_p=1.0,
            seed=42,
            # The summary is discarded downstream; keep it short to leave
            # enough completion budget for a complete validated JSON object.
            max_tokens=320,
        )
    content = response["choices"][0]["message"]["content"]
    return AgentDecision.model_validate_json(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    global _llm, _model_name
    if not args.model.is_file():
        raise SystemExit(f"GGUF model does not exist: {args.model}")
    _model_name = args.model.name
    _llm = Llama(
        model_path=str(args.model),
        n_ctx=4096,
        n_threads=6,
        n_batch=256,
        verbose=False,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), WorkerHandler)
    server.daemon_threads = True
    server.serve_forever(poll_interval=0.25)


if __name__ == "__main__":
    main()

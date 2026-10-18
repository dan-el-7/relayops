"""Embedding providers used by RelayOps memory retrieval.

The normal provider is a small, keyless FastEmbed/ONNX model.  A deterministic
hash encoder exists only for an explicitly visible degraded local demo mode;
it is not represented as a semantic model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Iterable

from .model_assets import (
    BGE_MODEL_FILE,
    BGE_MODEL_NAME,
    BGE_MODEL_SHA256,
    BGE_MODEL_SIZE,
    BGE_REPO_ID,
    BGE_REVISION,
)


class EmbeddingProvider:
    def __init__(
        self,
        mode: str = "auto",
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | Path | None = None,
        remote_api_key: str | None = None,
        remote_base_url: str | None = None,
        remote_model: str | None = None,
    ) -> None:
        self.requested_mode = (mode or "auto").strip().lower()
        self.model_name = model_name
        self.cache_dir = Path(
            cache_dir
            or os.getenv("RELAYOPS_MODEL_CACHE")
            or Path(__file__).resolve().parents[1] / ".model-cache"
        )
        self.remote_api_key = remote_api_key
        self.remote_base_url = (remote_base_url or "https://api.openai.com/v1").rstrip("/")
        self.remote_model = remote_model or "text-embedding-3-small"
        self._model = None
        self._mode = "unavailable"
        self._dimension = 0
        self._error: str | None = None
        self._initialize()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def semantic(self) -> bool:
        return self._mode in {"fastembed", "remote"}

    @property
    def ready(self) -> bool:
        return self._mode != "unavailable"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def error(self) -> str | None:
        return self._error

    def _initialize(self) -> None:
        if self.requested_mode == "remote":
            if not self.remote_api_key:
                self._error = "remote embedding mode requires an explicitly supplied API key"
                return
            self._mode = "remote"
            # text-embedding-3-small defaults to 1536 dimensions.
            self._dimension = 1536
            return

        if self.requested_mode in {"auto", "fastembed"}:
            try:
                from fastembed import TextEmbedding

                self.cache_dir.mkdir(parents=True, exist_ok=True)
                model_kwargs: dict[str, str] = {}
                if self.model_name == BGE_MODEL_NAME:
                    # The default semantic encoder is pinned by a Hub revision and
                    # LFS digest. Never silently fetch a newer registry model.
                    model_dir = (
                        self.cache_dir
                        / f"models--{BGE_REPO_ID.replace('/', '--')}"
                        / "snapshots"
                        / BGE_REVISION
                    )
                    model_file = model_dir / BGE_MODEL_FILE
                    if not model_file.is_file():
                        raise RuntimeError(
                            "Pinned BGE assets are missing; run ./scripts/setup.sh or "
                            "./scripts/setup-embedding-model.sh"
                        )
                    if model_file.stat().st_size != BGE_MODEL_SIZE or self._sha256(model_file) != BGE_MODEL_SHA256:
                        raise RuntimeError("Pinned BGE model asset failed size/SHA256 verification")
                    if any(not (model_dir / name).is_file() for name in (
                        "config.json", "ort_config.json", "special_tokens_map.json",
                        "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                    )):
                        raise RuntimeError("Pinned BGE tokenizer/config assets are incomplete")
                    model_kwargs["specific_model_path"] = str(model_dir)
                self._model = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=str(self.cache_dir),
                    **model_kwargs,
                )
                self._mode = "fastembed"
                # BAAI/bge-small-en-v1.5 emits 384-dimensional vectors.
                self._dimension = 384
                return
            except Exception as exc:  # model download / ONNX setup can fail offline
                self._error = f"FastEmbed unavailable: {type(exc).__name__}: {exc}"
                if self.requested_mode == "fastembed":
                    return

        if self.requested_mode in {"auto", "demo", "deterministic"}:
            self._mode = "demo-hash"
            self._dimension = 256
            return

        self._error = f"unknown embedding mode: {self.requested_mode}"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def embed(self, text: str) -> list[float]:
        text = (text or "").strip()
        if self._mode == "fastembed":
            vectors = list(self._model.embed([text or " "]))
            return self._normalize([float(value) for value in vectors[0]])
        if self._mode == "remote":
            return self._remote_embed(text)
        if self._mode == "demo-hash":
            return self._hash_embed(text)
        raise RuntimeError(self._error or "No embedding provider is ready")

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        items = [str(text or " ") for text in texts]
        if not items:
            return []
        if self._mode == "fastembed":
            return [self._normalize([float(value) for value in vector]) for vector in self._model.embed(items)]
        if self._mode == "remote":
            return [self._remote_embed(text) for text in items]
        if self._mode == "demo-hash":
            return [self._hash_embed(text) for text in items]
        raise RuntimeError(self._error or "No embedding provider is ready")

    def _remote_embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.remote_model, "input": text or " "}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.remote_base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.remote_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
        vector = [float(value) for value in result["data"][0]["embedding"]]
        if self._dimension != len(vector):
            self._dimension = len(vector)
        return self._normalize(vector)

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic feature-hash vector for offline UI/demo only."""
        dimension = self._dimension
        vector = [0.0] * dimension
        normalized = " ".join((text or "").lower().split())
        tokens = [f"w:{token}" for token in normalized.split()]
        padded = f"  {normalized}  "
        tokens.extend(f"c:{padded[index:index + 3]}" for index in range(max(0, len(padded) - 2)))
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            integer = int.from_bytes(digest, "big")
            index = integer % dimension
            sign = -1.0 if integer & (1 << 63) else 1.0
            vector[index] += sign
        return self._normalize(vector)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = list(left)
    b = list(right)
    if not a or len(a) != len(b):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in a))
    right_norm = math.sqrt(sum(value * value for value in b))
    if not left_norm or not right_norm:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (left_norm * right_norm)

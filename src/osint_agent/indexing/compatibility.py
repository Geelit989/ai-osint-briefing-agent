"""Deterministic semantic-index compatibility identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import ollama

from osint_agent.config import settings
from osint_agent.preprocessing.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    tokenizer,
)


class CompatibilityResolutionError(RuntimeError):
    """Current semantic configuration could not be established."""


COMPATIBILITY_MANIFEST_FIELDS = frozenset(
    {
        "index_schema_version",
        "embedding_provider",
        "embedding_model",
        "embedding_model_digest",
        "embedding_dimension",
        "document_embedding_prefix",
        "query_embedding_prefix",
        "chunk_size",
        "chunk_overlap",
        "chunking_implementation_version",
        "tokenizer",
        "tokenizer_digest",
        "text_normalization_version",
        "distance_metric",
    }
)


def validate_compatibility_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject partial, legacy, or structurally invalid semantic manifests."""

    if set(manifest) != COMPATIBILITY_MANIFEST_FIELDS:
        missing = sorted(COMPATIBILITY_MANIFEST_FIELDS - set(manifest))
        extra = sorted(set(manifest) - COMPATIBILITY_MANIFEST_FIELDS)
        raise ValueError(f"invalid compatibility fields: missing={missing}, extra={extra}")
    positive_integers = (
        "index_schema_version",
        "embedding_dimension",
        "chunk_size",
        "chunking_implementation_version",
        "text_normalization_version",
    )
    if any(
        not isinstance(manifest[field], int) or manifest[field] < 1
        for field in positive_integers
    ):
        raise ValueError("compatibility version, size, and dimension must be positive")
    overlap = manifest["chunk_overlap"]
    if not isinstance(overlap, int) or not 0 <= overlap < manifest["chunk_size"]:
        raise ValueError("chunk_overlap must be nonnegative and less than chunk_size")
    required_strings = COMPATIBILITY_MANIFEST_FIELDS - set(positive_integers) - {
        "chunk_overlap"
    }
    if any(
        not isinstance(manifest[field], str) or not manifest[field]
        for field in required_strings
    ):
        raise ValueError("compatibility identifiers must be non-empty strings")
    if manifest["distance_metric"] not in {"l2", "cosine", "ip"}:
        raise ValueError("unsupported Chroma distance metric")


def canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    """Serialize a compatibility manifest with stable ordering and syntax."""

    return json.dumps(
        dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compatibility_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Return a process-stable SHA-256 digest of semantic configuration."""

    return hashlib.sha256(
        canonical_manifest_json(manifest).encode("utf-8")
    ).hexdigest()


def _normalized_model_name(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def resolve_ollama_model_digest(model_name: str | None = None) -> str:
    """Resolve a mutable Ollama tag to its locally installed content digest."""

    target = _normalized_model_name(model_name or settings.EMBEDDING_MODEL)
    try:
        response = ollama.list()
    except Exception as exc:
        raise CompatibilityResolutionError(
            "Unable to resolve the local Ollama embedding model digest"
        ) from exc

    for model in response.models:
        candidate = model.model or ""
        if _normalized_model_name(candidate) == target and model.digest:
            return model.digest
    raise CompatibilityResolutionError(
        f"Embedding model {target!r} is not locally available with a digest"
    )


def tokenizer_digest() -> str:
    """Fingerprint the effective tokenizer vocabulary and configuration."""

    try:
        serialized = tokenizer.backend_tokenizer.to_str()
    except Exception as exc:
        raise CompatibilityResolutionError(
            "Unable to serialize the configured tokenizer"
        ) from exc
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_compatibility_manifest(
    *,
    model_digest: str | None = None,
    embedding_dimension: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the minimal manifest that determines stored vector semantics."""

    manifest: dict[str, Any] = {
        "index_schema_version": settings.INDEX_SCHEMA_VERSION,
        "embedding_provider": "ollama",
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_model_digest": (
            model_digest
            if model_digest is not None
            else resolve_ollama_model_digest()
        ),
        "embedding_dimension": (
            embedding_dimension
            if embedding_dimension is not None
            else settings.EMBEDDING_DIMENSION
        ),
        "document_embedding_prefix": settings.DOCUMENT_EMBEDDING_PREFIX,
        "query_embedding_prefix": settings.QUERY_EMBEDDING_PREFIX,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunking_implementation_version": (
            settings.CHUNKING_IMPLEMENTATION_VERSION
        ),
        "tokenizer": settings.TOKENIZER_NAME,
        "tokenizer_digest": tokenizer_digest(),
        "text_normalization_version": settings.TEXT_NORMALIZATION_VERSION,
        "distance_metric": settings.CHROMA_DISTANCE_METRIC,
    }
    if overrides:
        manifest.update(overrides)
    validate_compatibility_manifest(manifest)
    return manifest


def manifest_differences(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return human-inspectable field differences."""

    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(set(stored) | set(current)):
        if stored.get(key) != current.get(key):
            differences[key] = {
                "stored": stored.get(key),
                "current": current.get(key),
            }
    return differences

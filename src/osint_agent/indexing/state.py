"""Mechanical freshness and compatibility checks for the derived index."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from osint_agent.config import settings
from osint_agent.indexing.compatibility import (
    CompatibilityResolutionError,
    build_compatibility_manifest,
    compatibility_fingerprint,
    manifest_differences,
    validate_compatibility_manifest,
)
from osint_agent.indexing.indexing_documents import chunk_metadata
from osint_agent.models.document import Document
from osint_agent.preprocessing.chunking import chunk_document
from osint_agent.storage.chroma import (
    collection_metric,
    document_collection_exists,
    get_document_collection,
)
from osint_agent.storage.sqlite import get_documents


class CorpusIndexStatus(str, Enum):
    ABSENT = "absent"
    CURRENT = "current"
    STALE = "stale"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


class CompatibilityStatus(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class IndexInspection:
    corpus_status: CorpusIndexStatus
    compatibility_status: CompatibilityStatus
    usable: bool
    reasons: tuple[str, ...]
    corpus_digest: str | None = None
    expected_chunk_count: int | None = None
    stored_compatibility_fingerprint: str | None = None
    current_compatibility_fingerprint: str | None = None
    compatibility_differences: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )


class IndexNotUsableError(RuntimeError):
    """Semantic retrieval was attempted without a verified usable index."""

    def __init__(self, inspection: IndexInspection) -> None:
        self.inspection = inspection
        detail = "; ".join(inspection.reasons) or "unverified index state"
        super().__init__(
            "Semantic index is not usable; explicit reconciliation/reindexing "
            f"is required ({inspection.corpus_status.value}, "
            f"{inspection.compatibility_status.value}): {detail}"
        )


@dataclass(frozen=True)
class ExpectedChunk:
    chunk_id: str
    document: str
    metadata: dict[str, Any]


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def authoritative_corpus_payload(documents: list[Document]) -> list[dict[str, Any]]:
    """Return exactly the authoritative fields affecting indexed evidence."""

    return [
        {
            "doc_id": document.doc_id,
            "title": document.title,
            "source": document.source,
            "provider": document.provider,
            "source_type": document.source_type,
            "published_date": _iso(document.published_date),
            "event_time": _iso(document.event_time),
            "retrieved_at": _iso(document.retrieved_at),
            "url": document.url,
            "text": document.text,
            "contradiction_group": document.contradiction_group,
            "contradiction_position": document.contradiction_position,
        }
        for document in sorted(documents, key=lambda item: item.doc_id)
    ]


def authoritative_corpus_digest(documents: list[Document]) -> str:
    payload = json.dumps(
        authoritative_corpus_payload(documents),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_chunks(
    documents: list[Document],
    corpus_digest: str,
    compatibility_digest: str,
) -> list[ExpectedChunk]:
    records: list[ExpectedChunk] = []
    for document in sorted(documents, key=lambda item: item.doc_id):
        for chunk in chunk_document(document):
            records.append(
                ExpectedChunk(
                    chunk_id=chunk.chunk_id,
                    document=chunk.text,
                    metadata=chunk_metadata(
                        document,
                        chunk,
                        corpus_digest=corpus_digest,
                        compatibility_fingerprint=compatibility_digest,
                    ),
                )
            )
    return sorted(records, key=lambda item: item.chunk_id)


def expected_chunks_digest(records: list[ExpectedChunk]) -> str:
    payload = [
        {
            "id": record.chunk_id,
            "document": record.document,
            "metadata": record.metadata,
        }
        for record in records
    ]
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _actual_chunks_digest(
    collection,
) -> tuple[str | None, int | None, str | None]:
    actual, error = _actual_records(collection)
    if error:
        return None, None, error
    payload = [
        {
            "id": chunk_id,
            "document": actual[chunk_id][0],
            "metadata": actual[chunk_id][1],
        }
        for chunk_id in sorted(actual)
    ]
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), len(actual), None


def _read_state(db_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not db_path.is_file():
        return None, "authoritative SQLite database is missing"
    try:
        with sqlite3.connect(db_path) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                """
                SELECT * FROM semantic_index_state
                WHERE state_key = 'semantic_index'
                """
            ).fetchone()
    except sqlite3.DatabaseError as exc:
        return None, f"index state metadata is unreadable: {exc}"
    if row is None:
        return None, "index state metadata is missing or legacy"
    return dict(row), None


def _load_documents(db_path: Path) -> tuple[list[Document] | None, str | None]:
    try:
        return get_documents(db_path), None
    except (sqlite3.DatabaseError, ValueError) as exc:
        return None, f"authoritative corpus is unreadable: {exc}"


def _actual_records(collection) -> tuple[dict[str, tuple[str, dict]], str | None]:
    try:
        result = collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not (len(ids) == len(documents) == len(metadatas)):
            return {}, "Chroma returned inconsistent record arrays"
        if len(set(ids)) != len(ids):
            return {}, "Chroma returned duplicate chunk identifiers"
        records = {}
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            if not isinstance(text, str) or not isinstance(metadata, dict):
                return {}, "Chroma contains malformed chunk content or metadata"
            records[chunk_id] = (text, metadata)
        return records, None
    except Exception as exc:
        return {}, f"Chroma records are unreadable: {exc}"


def observed_embedding_dimension(collection) -> tuple[int | None, str | None]:
    """Read one vector to verify the collection's effective dimensionality."""

    try:
        if collection.count() == 0:
            return None, None
        result = collection.get(limit=1, include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) != 1:
            return None, "unable to observe a stored embedding dimension"
        return len(embeddings[0]), None
    except Exception as exc:
        return None, f"stored embedding dimension is unreadable: {exc}"


def verify_derived_records(
    collection,
    expected: list[ExpectedChunk],
) -> tuple[CorpusIndexStatus, list[str], int | None]:
    """Compare actual derived membership, content, and provenance."""

    actual, error = _actual_records(collection)
    if error:
        return CorpusIndexStatus.INVALID, [error], None

    expected_by_id = {record.chunk_id: record for record in expected}
    actual_ids = set(actual)
    expected_ids = set(expected_by_id)
    reasons: list[str] = []
    if missing := sorted(expected_ids - actual_ids):
        reasons.append(f"missing expected chunks: {missing}")
    if orphaned := sorted(actual_ids - expected_ids):
        reasons.append(f"stale/orphaned chunks: {orphaned}")
    for chunk_id in sorted(actual_ids & expected_ids):
        actual_text, actual_metadata = actual[chunk_id]
        expected_record = expected_by_id[chunk_id]
        if actual_text != expected_record.document:
            reasons.append(f"chunk content differs: {chunk_id}")
        if actual_metadata != expected_record.metadata:
            reasons.append(f"chunk provenance metadata differs: {chunk_id}")
    if reasons:
        return CorpusIndexStatus.STALE, reasons, None
    return CorpusIndexStatus.CURRENT, [], None


def _evaluate_compatibility(
    row: dict[str, Any],
    stored_manifest: dict[str, Any],
    stored_fingerprint: str,
    collection,
    runtime_manifest: dict[str, Any] | None,
) -> tuple[
    CompatibilityStatus,
    str | None,
    dict[str, dict[str, Any]],
    list[str],
]:
    try:
        current_manifest = (
            runtime_manifest
            if runtime_manifest is not None
            else build_compatibility_manifest()
        )
        validate_compatibility_manifest(current_manifest)
        current_fingerprint = compatibility_fingerprint(current_manifest)
    except (CompatibilityResolutionError, TypeError, ValueError) as exc:
        return CompatibilityStatus.UNVERIFIABLE, None, {}, [str(exc)]

    differences = manifest_differences(stored_manifest, current_manifest)
    try:
        actual_metric = collection_metric(collection)
    except Exception as exc:
        return (
            CompatibilityStatus.UNVERIFIABLE,
            current_fingerprint,
            differences,
            [f"collection metric is unreadable: {exc}"],
        )
    expected_metric = stored_manifest["distance_metric"]
    if actual_metric != expected_metric:
        differences["observed_distance_metric"] = {
            "stored": expected_metric,
            "current": actual_metric,
        }

    observed_dimension, dimension_error = observed_embedding_dimension(collection)
    if dimension_error:
        return (
            CompatibilityStatus.UNVERIFIABLE,
            current_fingerprint,
            differences,
            [dimension_error],
        )
    expected_dimension = stored_manifest["embedding_dimension"]
    if collection.count() and observed_dimension != expected_dimension:
        differences["observed_embedding_dimension"] = {
            "stored": expected_dimension,
            "current": observed_dimension,
        }
    if row.get("embedding_dimension") != expected_dimension:
        differences["committed_embedding_dimension"] = {
            "stored": row.get("embedding_dimension"),
            "current": expected_dimension,
        }

    compatible = not differences and stored_fingerprint == current_fingerprint
    return (
        CompatibilityStatus.COMPATIBLE
        if compatible
        else CompatibilityStatus.INCOMPATIBLE,
        current_fingerprint,
        differences,
        [] if compatible else ["semantic compatibility configuration differs"],
    )


def inspect_index_state(
    *,
    db_path: str | Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
) -> IndexInspection:
    """Re-establish derived-state validity from durable state on every call."""

    active_path = Path(db_path or settings.DB_PATH)
    row, state_error = _read_state(active_path)
    try:
        collection_exists = document_collection_exists()
    except Exception as exc:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=CompatibilityStatus.UNVERIFIABLE,
            usable=False,
            reasons=(f"configured Chroma state is unreadable: {exc}",),
        )
    if row is not None and row.get("status") in {"in_progress", "failed"}:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INCOMPLETE,
            compatibility_status=CompatibilityStatus.UNVERIFIABLE,
            usable=False,
            reasons=(row.get("error") or "index reconciliation is incomplete",),
            stored_compatibility_fingerprint=row.get(
                "compatibility_fingerprint"
            ),
        )
    if not collection_exists:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.ABSENT,
            compatibility_status=CompatibilityStatus.UNVERIFIABLE,
            usable=False,
            reasons=("configured Chroma collection is absent",),
        )

    if state_error:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=CompatibilityStatus.UNVERIFIABLE,
            usable=False,
            reasons=(state_error,),
        )
    assert row is not None

    if row.get("status") not in {"current", "stale"}:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=CompatibilityStatus.UNVERIFIABLE,
            usable=False,
            reasons=(f"unknown index commit status: {row.get('status')!r}",),
        )

    try:
        stored_manifest = json.loads(row.get("compatibility_manifest") or "")
        if not isinstance(stored_manifest, dict):
            raise ValueError("manifest is not an object")
        validate_compatibility_manifest(stored_manifest)
        stored_fingerprint = compatibility_fingerprint(stored_manifest)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=CompatibilityStatus.UNVERIFIABLE,
            usable=False,
            reasons=(f"stored compatibility manifest is corrupt: {exc}",),
        )
    if stored_fingerprint != row.get("compatibility_fingerprint"):
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=CompatibilityStatus.UNVERIFIABLE,
            usable=False,
            reasons=("stored compatibility fingerprint does not match manifest",),
        )

    try:
        collection = get_document_collection()
    except Exception as exc:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=CompatibilityStatus.UNVERIFIABLE,
            usable=False,
            reasons=(f"configured Chroma collection is unreadable: {exc}",),
            stored_compatibility_fingerprint=stored_fingerprint,
        )
    if str(collection.id) != row.get("collection_id"):
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=CompatibilityStatus.UNVERIFIABLE,
            usable=False,
            reasons=("configured Chroma collection was replaced",),
            stored_compatibility_fingerprint=stored_fingerprint,
        )

    documents, corpus_error = _load_documents(active_path)
    if corpus_error:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=CompatibilityStatus.UNVERIFIABLE,
            usable=False,
            reasons=(corpus_error,),
            stored_compatibility_fingerprint=stored_fingerprint,
        )
    assert documents is not None
    corpus_digest = authoritative_corpus_digest(documents)
    compatibility_status, current_fingerprint, differences, compat_reasons = (
        _evaluate_compatibility(
            row,
            stored_manifest,
            stored_fingerprint,
            collection,
            runtime_manifest,
        )
    )
    stale_reasons = []
    if row.get("status") == "stale":
        stale_reasons.append(row.get("error") or "index was explicitly invalidated")
    if row.get("corpus_digest") != corpus_digest:
        stale_reasons.append("authoritative corpus identity changed")
    if stale_reasons:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.STALE,
            compatibility_status=compatibility_status,
            usable=False,
            reasons=tuple(stale_reasons + compat_reasons),
            corpus_digest=corpus_digest,
            stored_compatibility_fingerprint=stored_fingerprint,
            current_compatibility_fingerprint=current_fingerprint,
            compatibility_differences=differences,
        )

    committed_chunks_digest = row.get("expected_chunks_digest")
    committed_chunk_count = row.get("expected_chunk_count")
    if (
        not isinstance(committed_chunks_digest, str)
        or len(committed_chunks_digest) != 64
        or not isinstance(committed_chunk_count, int)
        or committed_chunk_count < 0
    ):
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=compatibility_status,
            usable=False,
            reasons=("stored expected-chunk manifest is missing or legacy",),
            corpus_digest=corpus_digest,
            stored_compatibility_fingerprint=stored_fingerprint,
            current_compatibility_fingerprint=current_fingerprint,
            compatibility_differences=differences,
        )
    actual_digest, actual_count, actual_error = _actual_chunks_digest(collection)
    if actual_error:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=compatibility_status,
            usable=False,
            reasons=(actual_error,),
            corpus_digest=corpus_digest,
            stored_compatibility_fingerprint=stored_fingerprint,
            current_compatibility_fingerprint=current_fingerprint,
            compatibility_differences=differences,
        )
    if (
        actual_digest != committed_chunks_digest
        or actual_count != committed_chunk_count
    ) and compatibility_status != CompatibilityStatus.COMPATIBLE:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.STALE,
            compatibility_status=compatibility_status,
            usable=False,
            reasons=("actual Chroma chunk manifest differs from committed state",),
            corpus_digest=corpus_digest,
            expected_chunk_count=committed_chunk_count,
            stored_compatibility_fingerprint=stored_fingerprint,
            current_compatibility_fingerprint=current_fingerprint,
            compatibility_differences=differences,
        )

    # An unchanged committed derived manifest still proves corpus freshness when
    # the current runtime cannot reproduce old semantic configuration. It is not
    # usable until explicit reconciliation creates compatible vectors.
    if compatibility_status != CompatibilityStatus.COMPATIBLE:
        return IndexInspection(
            corpus_status=CorpusIndexStatus.CURRENT,
            compatibility_status=compatibility_status,
            usable=False,
            reasons=tuple(compat_reasons),
            corpus_digest=corpus_digest,
            expected_chunk_count=committed_chunk_count,
            stored_compatibility_fingerprint=stored_fingerprint,
            current_compatibility_fingerprint=current_fingerprint,
            compatibility_differences=differences,
        )

    expected = expected_chunks(documents, corpus_digest, stored_fingerprint)
    manifest_digest = expected_chunks_digest(expected)
    if (
        row.get("expected_chunks_digest") != manifest_digest
        or row.get("expected_chunk_count") != len(expected)
    ):
        return IndexInspection(
            corpus_status=CorpusIndexStatus.INVALID,
            compatibility_status=compatibility_status,
            usable=False,
            reasons=("stored expected-chunk manifest is corrupt or legacy",),
            corpus_digest=corpus_digest,
            stored_compatibility_fingerprint=stored_fingerprint,
            current_compatibility_fingerprint=current_fingerprint,
            compatibility_differences=differences,
        )

    corpus_status, record_reasons, _ = verify_derived_records(
        collection, expected
    )
    if corpus_status != CorpusIndexStatus.CURRENT:
        return IndexInspection(
            corpus_status=corpus_status,
            compatibility_status=compatibility_status,
            usable=False,
            reasons=tuple(record_reasons + compat_reasons),
            corpus_digest=corpus_digest,
            expected_chunk_count=len(expected),
            stored_compatibility_fingerprint=stored_fingerprint,
            current_compatibility_fingerprint=current_fingerprint,
            compatibility_differences=differences,
        )
    return IndexInspection(
        corpus_status=CorpusIndexStatus.CURRENT,
        compatibility_status=compatibility_status,
        usable=compatibility_status == CompatibilityStatus.COMPATIBLE,
        reasons=tuple(compat_reasons),
        corpus_digest=corpus_digest,
        expected_chunk_count=len(expected),
        stored_compatibility_fingerprint=stored_fingerprint,
        current_compatibility_fingerprint=current_fingerprint,
        compatibility_differences=differences,
    )


def assert_index_usable(
    *,
    db_path: str | Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
) -> IndexInspection:
    inspection = inspect_index_state(
        db_path=db_path, runtime_manifest=runtime_manifest
    )
    if not inspection.usable:
        raise IndexNotUsableError(inspection)
    return inspection

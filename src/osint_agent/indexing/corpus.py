"""Explicit reconciliation of authoritative SQLite state into Chroma."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from osint_agent.config import settings
from osint_agent.indexing.compatibility import (
    build_compatibility_manifest,
    canonical_manifest_json,
    compatibility_fingerprint,
    validate_compatibility_manifest,
)
from osint_agent.indexing.embedding import embed_documents
from osint_agent.indexing.state import (
    CorpusIndexStatus,
    assert_index_usable,
    authoritative_corpus_digest,
    expected_chunks,
    expected_chunks_digest,
    inspect_index_state,
    observed_embedding_dimension,
    verify_derived_records,
)
from osint_agent.storage.chroma import (
    collection_metric,
    document_collection_exists,
    get_chroma_client,
    get_document_collection,
    recreate_document_collection,
    upsert_chunks,
)
from osint_agent.storage.sqlite import (
    _create_semantic_index_state_table,
    get_documents,
)


class IndexReconciliationError(RuntimeError):
    """The explicit index repair attempt did not safely commit."""


EMBEDDING_BATCH_SIZE = 32


@dataclass(frozen=True)
class CorpusIndexingResult:
    documents_found: int
    documents_indexed: int
    chunks_indexed: int
    chroma_count_before: int
    chroma_count_after: int
    changed: bool
    corpus_status: str
    compatibility_fingerprint: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_attempt(
    db_path: Path,
    *,
    attempt_id: str,
    corpus_digest: str,
    manifest: dict[str, Any],
    fingerprint: str,
    chunks_digest: str,
    chunk_count: int,
) -> None:
    now = _utc_now()
    with sqlite3.connect(db_path) as con:
        _create_semantic_index_state_table(con)
        con.execute(
            """
            INSERT INTO semantic_index_state (
                state_key, status, attempt_id, corpus_digest,
                compatibility_manifest, compatibility_fingerprint,
                expected_chunks_digest, expected_chunk_count, collection_id,
                embedding_dimension, error, started_at, completed_at, updated_at
            ) VALUES (
                'semantic_index', 'in_progress', ?, ?, ?, ?, ?, ?, NULL,
                ?, NULL, ?, NULL, ?
            )
            ON CONFLICT(state_key) DO UPDATE SET
                status = excluded.status,
                attempt_id = excluded.attempt_id,
                corpus_digest = excluded.corpus_digest,
                compatibility_manifest = excluded.compatibility_manifest,
                compatibility_fingerprint = excluded.compatibility_fingerprint,
                expected_chunks_digest = excluded.expected_chunks_digest,
                expected_chunk_count = excluded.expected_chunk_count,
                collection_id = NULL,
                embedding_dimension = excluded.embedding_dimension,
                error = NULL,
                started_at = excluded.started_at,
                completed_at = NULL,
                updated_at = excluded.updated_at
            """,
            (
                attempt_id,
                corpus_digest,
                canonical_manifest_json(manifest),
                fingerprint,
                chunks_digest,
                chunk_count,
                manifest["embedding_dimension"],
                now,
                now,
            ),
        )


def _write_failed(db_path: Path, attempt_id: str, error: str) -> None:
    with sqlite3.connect(db_path) as con:
        _create_semantic_index_state_table(con)
        con.execute(
            """
            UPDATE semantic_index_state
            SET status = 'failed', error = ?, updated_at = ?
            WHERE state_key = 'semantic_index' AND attempt_id = ?
            """,
            (error, _utc_now(), attempt_id),
        )


def _write_current(
    db_path: Path,
    attempt_id: str,
    collection_id: str,
) -> None:
    now = _utc_now()
    with sqlite3.connect(db_path) as con:
        _create_semantic_index_state_table(con)
        cursor = con.execute(
            """
            UPDATE semantic_index_state
            SET status = 'current', collection_id = ?, error = NULL,
                completed_at = ?, updated_at = ?
            WHERE state_key = 'semantic_index'
              AND attempt_id = ?
              AND status = 'in_progress'
            """,
            (collection_id, now, now, attempt_id),
        )
        if cursor.rowcount != 1:
            raise IndexReconciliationError(
                "reconciliation attempt lost its in-progress commit marker"
            )


def _safe_collection_count() -> int:
    if not document_collection_exists():
        return 0
    return get_document_collection().count()


def reconcile_index(
    *,
    db_path: str | Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
    embedding_function: Callable[[list[str]], list[list[float]]] | None = None,
) -> CorpusIndexingResult:
    """Rebuild and verify derived state, committing only a consistent snapshot.

    A full local rebuild is deliberately used: it is the smallest reliable way
    to remove stale/orphaned chunks and recreate incompatible collection metric
    semantics without pretending SQLite and Chroma share a transaction.
    """

    active_db_path = Path(db_path or settings.DB_PATH)
    if not active_db_path.is_file():
        raise IndexReconciliationError(
            "authoritative SQLite database is missing; initialize it first"
        )
    manifest_was_supplied = runtime_manifest is not None
    manifest = (
        runtime_manifest
        if runtime_manifest is not None
        else build_compatibility_manifest()
    )
    validate_compatibility_manifest(manifest)
    fingerprint = compatibility_fingerprint(manifest)

    inspection = inspect_index_state(
        db_path=active_db_path,
        runtime_manifest=manifest,
    )
    documents = get_documents(active_db_path)
    count_before = _safe_collection_count()
    if inspection.usable:
        return CorpusIndexingResult(
            documents_found=len(documents),
            documents_indexed=0,
            chunks_indexed=0,
            chroma_count_before=count_before,
            chroma_count_after=count_before,
            changed=False,
            corpus_status=CorpusIndexStatus.CURRENT.value,
            compatibility_fingerprint=fingerprint,
        )

    corpus_digest = authoritative_corpus_digest(documents)
    records = expected_chunks(documents, corpus_digest, fingerprint)
    chunks_digest = expected_chunks_digest(records)
    attempt_id = str(uuid.uuid4())
    _write_attempt(
        active_db_path,
        attempt_id=attempt_id,
        corpus_digest=corpus_digest,
        manifest=manifest,
        fingerprint=fingerprint,
        chunks_digest=chunks_digest,
        chunk_count=len(records),
    )

    try:
        embed = embedding_function or embed_documents
        embeddings: list[list[float]] = []
        for start in range(0, len(records), EMBEDDING_BATCH_SIZE):
            batch = records[start : start + EMBEDDING_BATCH_SIZE]
            batch_embeddings = embed([record.document for record in batch])
            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    "Chunk/embedding count mismatch in reconciliation batch: "
                    f"{len(batch)} chunks, {len(batch_embeddings)} embeddings"
                )
            embeddings.extend(batch_embeddings)
        if len(embeddings) != len(records):
            raise ValueError(
                "Chunk/embedding count mismatch during reconciliation: "
                f"{len(records)} chunks, {len(embeddings)} embeddings"
            )
        expected_dimension = manifest["embedding_dimension"]
        invalid_dimensions = [
            index
            for index, embedding in enumerate(embeddings)
            if len(embedding) != expected_dimension
        ]
        if invalid_dimensions:
            raise ValueError(
                "Observed embedding dimension does not match manifest: "
                f"expected {expected_dimension}, rows={invalid_dimensions}"
            )

        collection = recreate_document_collection(manifest["distance_metric"])
        if records:
            batch_size = get_chroma_client().get_max_batch_size()
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                upsert_chunks(
                    ids=[record.chunk_id for record in batch],
                    documents=[record.document for record in batch],
                    embeddings=embeddings[start : start + batch_size],
                    metadatas=[record.metadata for record in batch],
                )
            collection = get_document_collection()

        # Re-establish both snapshots immediately before the success commit.
        if authoritative_corpus_digest(get_documents(active_db_path)) != corpus_digest:
            raise IndexReconciliationError(
                "authoritative corpus changed during reconciliation"
            )
        current_manifest = (
            manifest if manifest_was_supplied else build_compatibility_manifest()
        )
        if compatibility_fingerprint(current_manifest) != fingerprint:
            raise IndexReconciliationError(
                "semantic configuration changed during reconciliation"
            )

        status, reasons, _ = verify_derived_records(collection, records)
        if status != CorpusIndexStatus.CURRENT:
            raise IndexReconciliationError(
                "derived chunk verification failed: " + "; ".join(reasons)
            )
        if collection_metric(collection) != manifest["distance_metric"]:
            raise IndexReconciliationError(
                "created collection metric does not match the manifest"
            )
        observed_dimension, dimension_error = observed_embedding_dimension(
            collection
        )
        if dimension_error:
            raise IndexReconciliationError(dimension_error)
        if records and observed_dimension != expected_dimension:
            raise IndexReconciliationError(
                "stored embedding dimension does not match the manifest"
            )

        _write_current(active_db_path, attempt_id, str(collection.id))
        final = assert_index_usable(
            db_path=active_db_path, runtime_manifest=manifest
        )
        return CorpusIndexingResult(
            documents_found=len(documents),
            documents_indexed=len(documents),
            chunks_indexed=len(records),
            chroma_count_before=count_before,
            chroma_count_after=collection.count(),
            changed=True,
            corpus_status=final.corpus_status.value,
            compatibility_fingerprint=fingerprint,
        )
    except Exception as exc:
        _write_failed(active_db_path, attempt_id, str(exc))
        if isinstance(exc, IndexReconciliationError):
            raise
        raise IndexReconciliationError("index reconciliation failed") from exc


def index_corpus() -> CorpusIndexingResult:
    """Existing command-compatible name for explicit reconciliation."""

    return reconcile_index()

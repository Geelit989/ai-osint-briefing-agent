from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from osint_agent.config import settings
from osint_agent.indexing.compatibility import (
    build_compatibility_manifest,
    compatibility_fingerprint,
)
from osint_agent.indexing.corpus import (
    IndexReconciliationError,
    reconcile_index,
)
from osint_agent.indexing.state import (
    CompatibilityStatus,
    CorpusIndexStatus,
    IndexNotUsableError,
    authoritative_corpus_digest,
    expected_chunks,
    expected_chunks_digest,
    inspect_index_state,
)
from osint_agent.models.document import Document
from osint_agent.retrieval.semantic import semantic_search
from osint_agent.storage.chroma import (
    document_collection_exists,
    get_chroma_client,
    get_document_collection,
)
from osint_agent.storage.insert_data import upsert_document
from osint_agent.storage.sqlite import create_db, get_documents
from osint_agent.workflow import reason_over_evidence


FIXED_TIME = datetime(2026, 8, 15, 12, tzinfo=timezone.utc)


def document(doc_id: str, text: str) -> Document:
    return Document(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        source="Fixture News",
        provider="fixture",
        source_type="news",
        published_date=FIXED_TIME,
        retrieved_at=FIXED_TIME,
        url=f"https://example.test/{doc_id}",
        raw_text=text,
        text=text,
    )


def deterministic_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vectors.append([byte / 255 for byte in digest[:3]])
    return vectors


@pytest.fixture
def index_env(tmp_path, monkeypatch):
    db_path = tmp_path / "argus.db"
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    monkeypatch.setattr(settings, "CHROMA_PATH", tmp_path / "chroma")
    monkeypatch.setattr(settings, "CHROMA_COLLECTION", "argus_test_chunks")
    create_db()
    manifest = build_compatibility_manifest(
        model_digest="sha256:fixture-model", embedding_dimension=3
    )

    def insert(item: Document) -> None:
        with sqlite3.connect(db_path) as con:
            upsert_document(con, item)

    return db_path, manifest, insert


def test_unchanged_corpus_is_current_and_reconciliation_is_idempotent(index_env):
    db_path, manifest, insert = index_env
    insert(document("doc-a", "One authoritative report."))

    first = reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    collection_id = str(get_document_collection().id)
    second = reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    inspection = inspect_index_state(runtime_manifest=manifest)

    assert first.changed is True
    assert second.changed is False
    assert str(get_document_collection().id) == collection_id
    assert inspection.corpus_status == CorpusIndexStatus.CURRENT
    assert inspection.compatibility_status == CompatibilityStatus.COMPATIBLE
    assert inspection.usable is True


def test_reconciliation_batches_embeddings_and_writes_complete_index(
    index_env, monkeypatch
):
    db_path, manifest, insert = index_env
    for index in range(5):
        insert(document(f"doc-{index}", f"Authoritative report {index}."))

    embedding_calls = []

    def recording_embeddings(texts):
        embedding_calls.append(texts)
        return deterministic_embeddings(texts)

    monkeypatch.setattr(
        "osint_agent.indexing.corpus.EMBEDDING_BATCH_SIZE", 2
    )
    result = reconcile_index(
        runtime_manifest=manifest,
        embedding_function=recording_embeddings,
    )

    documents = get_documents(db_path)
    corpus_digest = authoritative_corpus_digest(documents)
    records = expected_chunks(
        documents,
        corpus_digest,
        compatibility_fingerprint(manifest),
    )
    assert [len(batch) for batch in embedding_calls] == [2, 2, 1]
    assert [text for batch in embedding_calls for text in batch] == [
        record.document for record in records
    ]
    assert set(get_document_collection().get()["ids"]) == {
        record.chunk_id for record in records
    }
    assert result.chunks_indexed == len(records)
    assert inspect_index_state(runtime_manifest=manifest).usable is True


def test_content_addition_and_deletion_each_make_index_stale(index_env):
    db_path, manifest, insert = index_env
    insert(document("doc-a", "Version one."))
    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )

    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE documents SET raw_text = ?, cleaned_text = ? WHERE doc_id = ?",
            ("Version two.", "Version two.", "doc-a"),
        )
    assert inspect_index_state(
        runtime_manifest=manifest
    ).corpus_status == CorpusIndexStatus.STALE
    assert (
        inspect_index_state(runtime_manifest=manifest).compatibility_status
        == CompatibilityStatus.COMPATIBLE
    )

    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    insert(document("doc-b", "A newly authoritative report."))
    assert inspect_index_state(
        runtime_manifest=manifest
    ).corpus_status == CorpusIndexStatus.STALE

    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    with sqlite3.connect(db_path) as con:
        con.execute("DELETE FROM documents WHERE doc_id = 'doc-b'")
    assert inspect_index_state(
        runtime_manifest=manifest
    ).corpus_status == CorpusIndexStatus.STALE


def test_reconciliation_removes_orphaned_and_stale_chunks(index_env):
    _, manifest, insert = index_env
    insert(document("doc-a", "Current report."))
    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    collection = get_document_collection()
    collection.add(
        ids=["orphan::chunk-000::deadbeef"],
        documents=["Orphaned report."],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"doc_id": "orphan"}],
    )

    inspection = inspect_index_state(runtime_manifest=manifest)
    assert inspection.corpus_status == CorpusIndexStatus.STALE
    assert any("orphaned" in reason for reason in inspection.reasons)

    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    assert "orphan::chunk-000::deadbeef" not in get_document_collection().get()[
        "ids"
    ]
    assert inspect_index_state(runtime_manifest=manifest).usable is True


def test_equal_counts_cannot_hide_replaced_or_partial_chunks(index_env):
    _, manifest, insert = index_env
    insert(document("doc-a", "Expected report A."))
    insert(document("doc-b", "Expected report B."))
    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    collection = get_document_collection()
    result = collection.get(include=["documents", "metadatas"])
    removed_id = result["ids"][0]
    collection.delete(ids=[removed_id])
    collection.add(
        ids=["substitute::chunk-000::bad"],
        documents=[result["documents"][0]],
        embeddings=[[0.2, 0.3, 0.4]],
        metadatas=[result["metadatas"][0]],
    )

    assert collection.count() == 2
    inspection = inspect_index_state(runtime_manifest=manifest)
    assert inspection.corpus_status == CorpusIndexStatus.STALE
    assert any("missing expected" in reason for reason in inspection.reasons)
    assert any("orphaned" in reason for reason in inspection.reasons)


def test_failed_partial_reconciliation_remains_incomplete_after_reopen(
    index_env, monkeypatch
):
    _, manifest, insert = index_env
    insert(document("doc-a", "Report A."))
    insert(document("doc-b", "Report B."))

    def partial_upsert(ids, documents, embeddings, metadatas):
        get_document_collection().upsert(
            ids=ids[:1],
            documents=documents[:1],
            embeddings=embeddings[:1],
            metadatas=metadatas[:1],
        )
        raise OSError("simulated interruption")

    monkeypatch.setattr(
        "osint_agent.indexing.corpus.upsert_chunks", partial_upsert
    )
    with pytest.raises(IndexReconciliationError):
        reconcile_index(
            runtime_manifest=manifest,
            embedding_function=deterministic_embeddings,
        )

    first = inspect_index_state(runtime_manifest=manifest)
    second = inspect_index_state(runtime_manifest=manifest)
    assert first.corpus_status == CorpusIndexStatus.INCOMPLETE
    assert second.corpus_status == CorpusIndexStatus.INCOMPLETE
    assert first.usable is second.usable is False


def test_interruption_after_collection_delete_remains_incomplete(
    index_env, monkeypatch
):
    _, manifest, insert = index_env
    insert(document("doc-a", "Report A."))
    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    insert(document("doc-b", "Report B."))

    def delete_then_terminate(metric):
        get_chroma_client().delete_collection(settings.CHROMA_COLLECTION)
        raise SystemExit("simulated process termination")

    monkeypatch.setattr(
        "osint_agent.indexing.corpus.recreate_document_collection",
        delete_then_terminate,
    )
    with pytest.raises(SystemExit, match="simulated process termination"):
        reconcile_index(
            runtime_manifest=manifest,
            embedding_function=deterministic_embeddings,
        )

    assert document_collection_exists() is False
    inspection = inspect_index_state(runtime_manifest=manifest)
    assert inspection.corpus_status == CorpusIndexStatus.INCOMPLETE
    assert inspection.usable is False


def test_authoritative_change_during_attempt_cannot_commit(index_env):
    _, manifest, insert = index_env
    insert(document("doc-a", "Initial report."))

    def mutate_while_embedding(texts):
        insert(document("doc-b", "Concurrent report."))
        return deterministic_embeddings(texts)

    with pytest.raises(
        IndexReconciliationError,
        match="authoritative corpus changed",
    ):
        reconcile_index(
            runtime_manifest=manifest,
            embedding_function=mutate_while_embedding,
        )
    assert inspect_index_state(
        runtime_manifest=manifest
    ).corpus_status == CorpusIndexStatus.INCOMPLETE


def test_missing_and_corrupt_metadata_fail_closed(index_env):
    db_path, manifest, insert = index_env
    insert(document("doc-a", "Report."))
    from osint_agent.storage.chroma import create_document_collection

    # The collection is deliberately created without a commit-state row.
    create_document_collection()
    assert inspect_index_state(
        runtime_manifest=manifest
    ).corpus_status == CorpusIndexStatus.INVALID

    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE semantic_index_state SET compatibility_manifest = 'not-json'"
        )
    assert inspect_index_state(
        runtime_manifest=manifest
    ).corpus_status == CorpusIndexStatus.INVALID


def test_reconciled_empty_corpus_is_current_but_yields_insufficient_evidence(
    index_env
):
    _, manifest, _ = index_env
    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    inspection = inspect_index_state(runtime_manifest=manifest)

    assert inspection.corpus_status == CorpusIndexStatus.CURRENT
    assert inspection.expected_chunk_count == 0
    assert semantic_search("anything", runtime_manifest=manifest) == []
    assert reason_over_evidence("anything", [], 0.5, 1).status == (
        "insufficient_evidence"
    )


def test_production_semantic_search_rejects_absent_index_before_embedding(
    index_env, monkeypatch
):
    called = False

    def unexpected_embedding(query):
        nonlocal called
        called = True
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        "osint_agent.retrieval.semantic.embed_query", unexpected_embedding
    )
    with pytest.raises(IndexNotUsableError, match="reconciliation/reindexing"):
        semantic_search("query")
    assert called is False


def test_incompatible_current_corpus_is_rejected_with_diagnostics(index_env):
    _, manifest, insert = index_env
    insert(document("doc-a", "Report."))
    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    changed = {**manifest, "chunk_size": manifest["chunk_size"] + 1}

    inspection = inspect_index_state(runtime_manifest=changed)
    assert inspection.corpus_status == CorpusIndexStatus.CURRENT
    assert inspection.compatibility_status == CompatibilityStatus.INCOMPATIBLE
    assert inspection.compatibility_differences["chunk_size"] == {
        "stored": manifest["chunk_size"],
        "current": changed["chunk_size"],
    }
    with pytest.raises(IndexNotUsableError):
        semantic_search("query", runtime_manifest=changed)


def test_repair_recreates_collection_for_metric_change(index_env):
    _, manifest, insert = index_env
    insert(document("doc-a", "Report."))
    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    old_collection_id = str(get_document_collection().id)
    changed = {**manifest, "distance_metric": "cosine"}

    result = reconcile_index(
        runtime_manifest=changed,
        embedding_function=deterministic_embeddings,
    )
    inspection = inspect_index_state(runtime_manifest=changed)
    assert result.changed is True
    assert str(get_document_collection().id) != old_collection_id
    assert get_document_collection().configuration["hnsw"]["space"] == "cosine"
    assert inspection.usable is True


def test_observed_dimension_inconsistency_is_rejected(index_env):
    db_path, manifest, insert = index_env
    insert(document("doc-a", "Report."))
    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    changed = {**manifest, "embedding_dimension": 4}
    changed_fingerprint = compatibility_fingerprint(changed)
    documents = get_documents(db_path)
    corpus_digest = authoritative_corpus_digest(documents)
    records = expected_chunks(documents, corpus_digest, changed_fingerprint)
    collection = get_document_collection()
    for record in records:
        collection.update(ids=[record.chunk_id], metadatas=[record.metadata])
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            UPDATE semantic_index_state
            SET compatibility_manifest = ?, compatibility_fingerprint = ?,
                expected_chunks_digest = ?, embedding_dimension = ?
            """,
            (
                json.dumps(changed, sort_keys=True, separators=(",", ":")),
                changed_fingerprint,
                expected_chunks_digest(records),
                4,
            ),
        )

    inspection = inspect_index_state(runtime_manifest=changed)
    assert inspection.corpus_status == CorpusIndexStatus.CURRENT
    assert inspection.compatibility_status == CompatibilityStatus.INCOMPATIBLE
    assert "observed_embedding_dimension" in inspection.compatibility_differences


def test_observed_collection_metric_inconsistency_is_rejected(
    index_env, monkeypatch
):
    _, manifest, insert = index_env
    insert(document("doc-a", "Report."))
    reconcile_index(
        runtime_manifest=manifest,
        embedding_function=deterministic_embeddings,
    )
    monkeypatch.setattr(
        "osint_agent.indexing.state.collection_metric", lambda collection: "cosine"
    )

    inspection = inspect_index_state(runtime_manifest=manifest)
    assert inspection.corpus_status == CorpusIndexStatus.CURRENT
    assert inspection.compatibility_status == CompatibilityStatus.INCOMPATIBLE
    assert "observed_distance_metric" in inspection.compatibility_differences

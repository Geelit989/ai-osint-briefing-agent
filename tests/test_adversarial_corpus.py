"""Isolated integration tests: real storage/chunking; no live corpus or Ollama."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock
import zipfile

import pytest

from osint_agent.config import settings
from osint_agent.indexing.state import (
    CorpusIndexStatus, IndexNotUsableError, inspect_index_state,
)
from osint_agent.models.document import Document
from osint_agent.preprocessing.chunking import chunk_document
from osint_agent.reasoning.synthesis import OllamaReasoningModel
from osint_agent.storage.chroma import get_document_collection
from osint_agent.storage.sqlite import get_documents


@pytest.fixture
def utility():
    path = Path(__file__).resolve().parents[1] / "scripts/seed_adversarial_corpus.py"
    spec = importlib.util.spec_from_file_location("adversarial_corpus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rows():
    return [{
        "corpus_id": f"ADV-{number:03d}", "title": f"Synthetic report {number}",
        "source": f"Fixture source {number}", "provider": "argus_adversarial_v1",
        "source_type": "synthetic_adversarial",
        "published_date": "2026-08-10T12:00:00+00:00",
        "url": f"https://synthetic.argus.invalid/{number}",
        "text": f"Report {number}. " + "Evidence remains qualified. " * 180,
        "event_group": "EXTERNAL_EVENT_SENTINEL",
        "adversarial_pattern": "EXTERNAL_PATTERN_SENTINEL",
        "lineage_group": "EXTERNAL_LINEAGE_SENTINEL",
    } for number in range(1, 51)]


@pytest.fixture
def package(tmp_path, rows):
    archive = tmp_path / "supplied.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("manifest.json", json.dumps({
            "synthetic": True, "article_count": 50,
        }))
        output.writestr("argus_adversarial_corpus_50.jsonl", "\n".join(
            json.dumps(row) for row in rows
        ))
        output.writestr("evaluation_queries.jsonl", "\n".join(
            json.dumps({"query_id": key, "query": f"What was reported, {key}?",
                        "expected_behavior": "EXTERNAL_ANSWER_SENTINEL"})
            for key in ("Q01", "Q02", "Q12")
        ))
    return archive


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_DIMENSION", 3)
    monkeypatch.setattr(
        "osint_agent.indexing.compatibility.resolve_ollama_model_digest",
        lambda: "sha256:offline-fixture",
    )
    monkeypatch.setattr(
        "osint_agent.indexing.corpus.embed_documents",
        lambda texts: [[0.0, 0.0, 0.0] for text in texts],
    )
    monkeypatch.setattr(
        "osint_agent.retrieval.semantic.embed_query", lambda query: [0.0, 0.0, 0.0],
    )


def test_package_maps_full_documents_without_ground_truth(utility, package, rows):
    documents, queries, external = utility.load_package(package)
    assert len(documents) == 50
    assert len({doc.doc_id for doc in documents}) == 50
    for doc, row in zip(documents, rows):
        assert isinstance(doc, Document)
        assert doc.doc_id == row["corpus_id"]
        assert doc.raw_text == doc.text == row["text"].strip()
        assert doc.title == row["title"] and doc.source == row["source"]
        assert doc.url == row["url"]
        assert doc.published_date.isoformat() == row["published_date"]
        assert doc.provider == utility.PROVIDER
        assert doc.source_type == utility.SOURCE_TYPE
        assert doc.event_time is None
        assert doc.meta_data == {}
        assert "EXTERNAL_" not in doc.model_dump_json()
    assert documents[0].doc_id == utility.load_package(package)[0][0].doc_id
    assert external["ADV-001"]["event_group"] == "EXTERNAL_EVENT_SENTINEL"
    assert queries[0]["expected_behavior"] == "EXTERNAL_ANSWER_SENTINEL"


@pytest.mark.parametrize("field", ["text", "title", "published_date", "corpus_id"])
def test_missing_field_rejected_before_any_writes(utility, rows, field):
    del rows[0][field]
    with pytest.raises(ValueError, match=field):
        utility.validate_documents(rows)


@pytest.mark.parametrize("count", [49, 51])
def test_wrong_count_rejected(utility, rows, count):
    changed = rows[:count] if count < 50 else rows + [rows[0]]
    with pytest.raises(ValueError, match="exactly 50"):
        utility.validate_documents(changed)


def test_duplicate_identity_and_non_synthetic_provenance_rejected(utility, rows):
    rows[1]["corpus_id"] = rows[0]["corpus_id"]
    with pytest.raises(ValueError, match="unique supplied identities"):
        utility.validate_documents(rows)
    rows[0]["provider"] = "real_news"
    with pytest.raises(ValueError, match="synthetic provenance"):
        utility.validate_documents(rows)


@pytest.mark.parametrize("text", ['{"bad":', '[]', 'null'])
def test_malformed_jsonl_is_explicit(utility, text):
    with pytest.raises(ValueError, match="line 1"):
        utility.parse_jsonl(text)


def test_invalid_package_never_establishes_write_context(utility, tmp_path):
    archive = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("manifest.json", '{"synthetic": false}')
    root = tmp_path / "eval"
    with pytest.raises(ValueError, match="50 synthetic"):
        utility.run(archive, root)
    assert not root.exists()


@pytest.mark.parametrize("collision", ["root", "database", "chroma", "symlink", "hardlink"])
def test_live_paths_are_rejected_before_writes(utility, tmp_path, monkeypatch, collision):
    live = tmp_path / "live"
    live.mkdir()
    database = live / "osint_sys.db"
    database.write_bytes(b"live database sentinel")
    chroma = live / "chroma"
    chroma.mkdir()
    monkeypatch.setattr(settings, "DB_PATH", database)
    monkeypatch.setattr(settings, "CHROMA_PATH", chroma)
    root = tmp_path / "evaluation"
    if collision == "root":
        root = live
    elif collision == "database":
        monkeypatch.setattr(settings, "DB_PATH", root / "osint_sys.db")
    elif collision == "chroma":
        monkeypatch.setattr(settings, "CHROMA_PATH", root / "chroma")
    elif collision == "symlink":
        root.symlink_to(live, target_is_directory=True)
    else:
        root.mkdir()
        (root / "osint_sys.db").hardlink_to(database)
    original = settings.DB_PATH, settings.CHROMA_PATH, settings.CHROMA_COLLECTION
    with pytest.raises(ValueError, match="live persistence"):
        with utility.isolated_settings(root, "fixture"):
            pytest.fail("must reject before yielding")
    assert database.read_bytes() == b"live database sentinel"
    assert not (root / "synthetic-corpus.json").exists()
    assert original == (settings.DB_PATH, settings.CHROMA_PATH, settings.CHROMA_COLLECTION)


def test_unowned_directory_rejected_and_settings_restored(utility, tmp_path):
    root = tmp_path / "eval"
    root.mkdir()
    unrelated = root / "unrelated.txt"
    unrelated.write_text("keep")
    with pytest.raises(ValueError, match="not owned"):
        with utility.isolated_settings(root, "fixture"):
            pytest.fail("must not reuse arbitrary existing state")
    safe = tmp_path / "safe"
    original = settings.DB_PATH, settings.CHROMA_PATH, settings.CHROMA_COLLECTION
    with pytest.raises(RuntimeError, match="simulated"):
        with utility.isolated_settings(safe, "fixture"):
            raise RuntimeError("simulated failure")
    assert original == (settings.DB_PATH, settings.CHROMA_PATH, settings.CHROMA_COLLECTION)
    assert unrelated.read_text() == "keep"


def test_real_storage_chunking_reconciliation_idempotency_and_grouping(
    utility, package, tmp_path, offline, monkeypatch,
):
    documents, queries, external = utility.load_package(package)
    persist = Mock(wraps=utility.upsert_document)
    reconcile = Mock(wraps=utility.reconcile_index)
    monkeypatch.setattr(utility, "upsert_document", persist)
    monkeypatch.setattr(utility, "reconcile_index", reconcile)
    with utility.isolated_settings(tmp_path / "eval", "fixture"):
        first = utility.ingest(documents)
        second = utility.ingest(documents)
        assert first["documents"] == second["documents"] == 50
        assert first["chunk_ids"] == second["chunk_ids"]
        assert first["chunks"] == second["chunks"] > 50
        assert reconcile.call_count == 2 and persist.call_count == 100
        assert all(call.args[1] in documents for call in persist.call_args_list)
        actual = get_document_collection().get(include=["documents", "metadatas"])
        expected = {chunk.chunk_id: chunk.text for doc in get_documents()
                    for chunk in chunk_document(doc)}
        assert dict(zip(actual["ids"], actual["documents"])) == expected
        assert all("EXTERNAL_" not in json.dumps(metadata) for metadata in actual["metadatas"])
        assert utility.assert_index_usable().usable
        baseline = utility.chunk_baseline()
        assert baseline["multi_chunk_documents"] == 50
        assert baseline["chunks"] == len(expected)
        # Actual production retrieval and grouping on real multi-chunk records.
        # Filter only for this controlled grouping observation, not corpus retrieval.
        doc_id = documents[0].doc_id
        from osint_agent.models.document import EvidenceChunk
        evidence = [EvidenceChunk(chunk_id=key, doc_id=doc_id, text=text, distance=0.0)
                    for key, text, meta in zip(actual["ids"], actual["documents"], actual["metadatas"])
                    if meta["doc_id"] == doc_id]
        assessment = utility.check_retrieval_sufficiency(evidence, 0.5, 2)
        assert assessment.usable_chunk_count > 1
        assert assessment.independent_evidence_count == 1
        assert "same_doc_id" in assessment.grouping_manifest[0].grouping_reasons
        results = utility.retrieval_baseline(queries, external)
        assert len(results) == 3
        assert all(len(result["retrieved"]) == 5 for result in results)


def test_failed_embedding_blocks_all_downstream_phases(
    utility, package, tmp_path, offline, monkeypatch,
):
    def fail(texts):
        raise OSError("simulated embedding interruption")
    monkeypatch.setattr("osint_agent.indexing.corpus.embed_documents", fail)
    retrieve = Mock(side_effect=AssertionError("retrieval must not run"))
    reason = Mock(side_effect=AssertionError("reasoning must not run"))
    monkeypatch.setattr(utility, "retrieval_baseline", retrieve)
    monkeypatch.setattr(utility, "reasoning_baseline", reason)
    root = tmp_path / "eval"
    with pytest.raises(Exception, match="index reconciliation failed"):
        utility.run(package, root)
    report = json.loads((root / "baseline.json").read_text())
    assert report["failure"]["stage"] == "ingestion_first"
    assert not retrieve.called and not reason.called
    import hashlib
    with utility.isolated_settings(root, hashlib.sha256(package.read_bytes()).hexdigest()):
        inspection = inspect_index_state()
        assert inspection.corpus_status == CorpusIndexStatus.INCOMPLETE
        assert not inspection.usable


def test_certification_enforced_at_retrieval_boundary(utility, tmp_path, monkeypatch):
    retrieve = Mock(side_effect=AssertionError("must not retrieve"))
    monkeypatch.setattr(utility, "semantic_search", retrieve)
    with utility.isolated_settings(tmp_path / "eval", "fixture"):
        with pytest.raises(IndexNotUsableError):
            utility.retrieval_baseline([], {})
        with pytest.raises(IndexNotUsableError):
            utility.reasoning_baseline([])
    assert not retrieve.called


def test_ground_truth_never_reaches_production_models(
    utility, package, tmp_path, offline, monkeypatch,
):
    model_inputs = []
    def generate(self, system, user, schema):
        model_inputs.append((system, user))
        payload = json.loads(user.split("\n\n", 1)[1])
        if schema["title"] == "SynthesisDraft":
            chunk = payload["evidence"][0]
            claim = {"text": "Fixture report.", "citations": [chunk["source_id"]],
                     "supporting_quotes": [{"source_id": chunk["source_id"],
                                            "chunk_id": chunk["chunk_id"],
                                            "text": chunk["text"]}]}
            return {"title": claim, "bluf": claim, "reported_developments": [],
                    "analytic_assessments": [], "intelligence_gaps": []}
        return {"claim_id": payload["claim_id"], "status": "supported", "issues": []}
    monkeypatch.setattr(OllamaReasoningModel, "generate", generate)
    query_inputs = []
    def embed_query(query):
        query_inputs.append(query)
        return [0.0, 0.0, 0.0]
    monkeypatch.setattr("osint_agent.retrieval.semantic.embed_query", embed_query)
    report = utility.run(package, tmp_path / "eval")
    assert report["idempotent"] and report["final_inspection"]["usable"]
    assert len(report["reasoning"]) == 3
    assert all(row["result"]["status"] == "success" for row in report["reasoning"])
    assert model_inputs
    assert "EXTERNAL_" not in json.dumps(model_inputs)
    assert set(query_inputs) == {f"What was reported, {key}?" for key in ("Q01", "Q02", "Q12")}

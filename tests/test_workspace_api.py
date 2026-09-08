"""Local API boundary checks; no Ollama, Chroma, or live corpus required."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osint_agent.models.document import Document
from osint_agent.storage.insert_data import upsert_document
from osint_agent.storage.sqlite import create_db
from osint_agent.workspace.api import create_app


HEALTH = {
    "status": "degraded",
    "sqlite": {"status": "ready", "document_count": 3},
    "chroma": {"status": "unavailable", "record_count": None},
    "ollama": {"status": "unavailable", "reasoning_model": "fixture-model"},
}


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, dict[str, Document]]:
    db_path = tmp_path / "argus.sqlite"
    create_db(db_path)
    documents = {
        doc_id: Document(
            doc_id=doc_id,
            title=title,
            source="Fixture publisher",
            provider="fixture",
            source_type="news",
            published_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
            retrieved_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
            url=f"https://example.test/{doc_id}",
            raw_text=f"Authoritative raw text for {doc_id}.",
            text=f"Authoritative cleaned text for {doc_id}.",
            meta_data={"fixture": True, "preserved": [1, 2]},
        )
        for doc_id, title in [
            ("doc-a", "Alpha infrastructure reporting"),
            ("doc-b", "Beta 100% capacity_under review"),
            ("doc-z", "Unrelated culture report"),
        ]
    }
    with sqlite3.connect(db_path) as con:
        for document in documents.values():
            upsert_document(con, document)
    return db_path, documents


def run_result(
    query: str,
    n_results: int,
    run_id: str,
    created_at: str,
    *,
    status: str = "success",
    doc_id: str = "doc-a",
) -> dict:
    """A serialized application result, including unrelated retrieved evidence."""
    chunk_id = f"{doc_id}::chunk-000"
    evidence = [
        {
            "chunk_id": "doc-z::chunk-000",
            "doc_id": "doc-z",
            "text": "Unrelated culture report, retrieved first.",
            "distance": 0.99,
        },
        {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": "Operators reported service disruption.",
            "distance": 0.2,
        },
    ]
    source = {
        "source_id": "S1",
        "doc_id": doc_id,
        "chunk_ids": [chunk_id],
        "title": "Retrieved title snapshot",
        "provider": "fixture",
    }
    statement = {
        "text": "Operators reported service disruption.",
        "citations": ["S1"],
        "supporting_spans": [
            {
                "source_id": "S1",
                "chunk_id": chunk_id,
                "start": 0,
                "end": 38,
                "text": "Operators reported service disruption.",
            }
        ],
        "acknowledged_contradictions": [],
    }
    return {
        "run_id": run_id,
        "query": query,
        "created_at": created_at,
        "n_results": n_results,
        "status": status,
        "retrieval": {
            "retrieved_chunk_count": 2,
            "usable_chunk_count": 1,
            "independent_evidence_count": 1,
        },
        "sufficiency": {
            "status": "INSUFFICIENT" if status == "insufficient_evidence" else "SUFFICIENT",
            "reason": "Fixture assessment from the application boundary.",
        },
        "grouping_manifest": [
            {
                "unit_id": "U1",
                "member_chunk_ids": [chunk_id],
                "member_document_ids": [doc_id],
                "grouping_reasons": ["singleton"],
            }
        ],
        "brief": {
            "query": query,
            "generated_date": "2026-09-08",
            "title": deepcopy(statement),
            "bluf": deepcopy(statement),
            "reported_developments": [deepcopy(statement)],
            "analytic_assessments": [],
            "intelligence_gaps": [],
            "sources": [source],
            "known_contradictions": [],
        } if status == "success" else None,
        "claims": [
            {"claim_id": "bluf", **statement, "status": "supported", "issues": []}
        ] if status == "success" else [],
        "citations": ["S1"] if status == "success" else [],
        "sources": [source],
        "evidence": evidence,
        "validation": {
            "stages": [
                {"name": "retrieval", "status": "pass", "detail": None},
                {"name": "claim_support", "status": "pass" if status == "success" else "not_run", "detail": None},
            ],
        },
        "failure_detail": None if status == "success" else "Fixture fail-closed outcome.",
    }


def client_for(corpus, executor=run_result) -> TestClient:
    return TestClient(create_app(
        db_path=corpus[0],
        run_executor=executor,
        health_reader=lambda: deepcopy(HEALTH),
    ))


@pytest.mark.parametrize("path", ["/api/health", "/api/corpus/status"])
def test_health_stays_available_when_models_and_index_are_unavailable(corpus, path):
    with client_for(corpus) as client:
        response = client.get(path)
    assert response.status_code == 200
    assert response.json() == HEALTH


def test_success_serializes_and_persists_before_application_execution(corpus):
    observed = []

    def executor(query, n_results, run_id, created_at):
        with sqlite3.connect(corpus[0]) as con:
            persisted = con.execute(
                "SELECT run_id FROM analyst_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        assert persisted == (run_id,)
        assert datetime.fromisoformat(created_at).tzinfo is not None
        observed.append((query, n_results, run_id, created_at))
        return run_result(query, n_results, run_id, created_at)

    with client_for(corpus, executor) as client:
        response = client.post("/api/runs", json={"query": "Infrastructure reporting?"})
        assert response.status_code == 201
        result = response.json()
        assert result == run_result(*observed[0])
        assert result["n_results"] == 5
        assert result["brief"]["bluf"]["citations"] == ["S1"]
        assert client.get(f"/api/runs/{result['run_id']}").json() == result
    assert len(observed) == 1


@pytest.mark.parametrize("status", ["insufficient_evidence", "validation_failure", "error"])
def test_failed_outcomes_preserve_evidence_without_publishing_a_brief(corpus, status):
    def executor(query, n_results, run_id, created_at):
        result = run_result(query, n_results, run_id, created_at, status=status)
        # Even a malformed adapter result cannot publish a rejected product.
        result["brief"] = {"bluf": "Rejected model draft must not be displayed."}
        result["claims"] = [{"text": "Rejected claim", "citations": ["S1"]}]
        return result

    with client_for(corpus, executor) as client:
        response = client.post("/api/runs", json={"query": "A focused question"})
        assert response.status_code == 201
        result = response.json()
        assert result["status"] == status
        assert result["brief"] is None
        assert result["claims"] == []
        assert len(result["evidence"]) == 2
        assert result["sources"][0]["doc_id"] == "doc-a"
        assert result["failure_detail"]
        assert "Rejected model draft" not in response.text
        reopened = client.get(f"/api/runs/{result['run_id']}")
        assert reopened.json() == result
        source = client.get(f"/api/runs/{result['run_id']}/sources/S1")
        assert source.status_code == 200
        assert source.json()["chunks"][0]["doc_id"] == "doc-a"
        assert source.json()["claims"] == []


def test_unexpected_application_error_is_recorded_and_reopenable(corpus):
    def executor(query, n_results, run_id, created_at):
        raise RuntimeError("fixture runtime unavailable")

    with client_for(corpus, executor) as client:
        response = client.post("/api/runs", json={"query": "A focused question"})
        assert response.status_code == 201
        result = response.json()
        assert result["status"] == "error"
        assert result["brief"] is None
        assert result["failure_detail"]["type"] == "RuntimeError"
        assert "Traceback" not in response.text
        assert client.get(f"/api/runs/{result['run_id']}").json() == result
        history = client.get("/api/runs").json()
        assert history["total"] == 1
        assert history["items"][0]["status"] == "error"


def test_history_survives_reopening_app_without_changing_authoritative_corpus(corpus):
    with sqlite3.connect(corpus[0]) as con:
        original_rows = con.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
        original_schema = con.execute("PRAGMA table_info(documents)").fetchall()
        original_tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}

    with client_for(corpus) as client:
        result = client.post("/api/runs", json={"query": "Persistent question", "n_results": 7}).json()

    for _ in range(2):
        with client_for(corpus) as reopened:
            assert reopened.get(f"/api/runs/{result['run_id']}").json() == result
            assert reopened.get("/api/runs").json()["total"] == 1
            assert reopened.get("/api/documents/doc-a").json()["raw_text"] == corpus[1]["doc-a"].raw_text

    with sqlite3.connect(corpus[0]) as con:
        assert con.execute("SELECT * FROM documents ORDER BY doc_id").fetchall() == original_rows
        assert con.execute("PRAGMA table_info(documents)").fetchall() == original_schema
        new_tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert new_tables - original_tables == {"analyst_runs"}


def test_run_local_citations_resolve_by_mapping_not_retrieval_position(corpus):
    def executor(query, n_results, run_id, created_at):
        return run_result(query, n_results, run_id, created_at, doc_id=query)

    with client_for(corpus, executor) as client:
        for doc_id in ("doc-a", "doc-b"):
            result = client.post("/api/runs", json={"query": doc_id}).json()
            run_id = result["run_id"]
            assert client.get(f"/api/runs/{run_id}/sources").json() == {"items": result["sources"]}
            detail = client.get(f"/api/runs/{run_id}/sources/S1")
            assert detail.status_code == 200
            source = detail.json()
            assert source["source"]["doc_id"] == doc_id
            assert source["document"] == corpus[1][doc_id].model_dump(mode="json")
            assert [chunk["chunk_id"] for chunk in source["chunks"]] == [f"{doc_id}::chunk-000"]
            assert source["claims"] == result["claims"]
            document = client.get(f"/api/documents/{doc_id}", params={"run_id": run_id})
            assert document.status_code == 200
            assert document.json()["relevant_chunks"] == source["chunks"]

        assert client.get("/api/sources/S1").status_code == 404


def test_retrieved_source_remains_inspectable_when_document_is_missing(corpus):
    def executor(query, n_results, run_id, created_at):
        return run_result(query, n_results, run_id, created_at, doc_id="missing-document")

    with client_for(corpus, executor) as client:
        result = client.post("/api/runs", json={"query": "Missing document"}).json()
        response = client.get(f"/api/runs/{result['run_id']}/sources/S1")
        assert response.status_code == 200
        assert response.json()["document"] is None
        assert len(response.json()["chunks"]) == 1
        assert client.get("/api/documents/missing-document").status_code == 404


@pytest.mark.parametrize("corruption", ["invalid_metadata", "blank_text"])
def test_corrupt_stored_document_does_not_hide_recorded_citation_evidence(corpus, corruption):
    with client_for(corpus) as client:
        result = client.post("/api/runs", json={"query": "A focused question"}).json()
        with sqlite3.connect(corpus[0]) as con:
            if corruption == "invalid_metadata":
                con.execute(
                    "UPDATE documents SET meta_data = ? WHERE doc_id = ?",
                    ("{invalid JSON", "doc-a"),
                )
            else:
                con.execute(
                    "UPDATE documents SET cleaned_text = ? WHERE doc_id = ?",
                    (" \n\t ", "doc-a"),
                )

        response = client.get(f"/api/runs/{result['run_id']}/sources/S1")

        assert response.status_code == 200
        detail = response.json()
        assert detail["document"] is None
        assert "authoritative document could not be read" in detail["document_error"].lower()
        assert detail["source"] == result["sources"][0]
        assert detail["chunks"] == [result["evidence"][1]]
        assert detail["claims"] == result["claims"]
        assert "Traceback" not in response.text
        assert client.get(f"/api/runs/{result['run_id']}").json() == result


def test_run_history_is_bounded_metadata_in_newest_first_order(corpus):
    with client_for(corpus) as client:
        results = [
            client.post("/api/runs", json={"query": f"Question {i}", "n_results": i + 1}).json()
            for i in range(3)
        ]
        response = client.get("/api/runs", params={"limit": 1, "offset": 1})
        assert response.status_code == 200
        page = response.json()
        assert (page["total"], page["limit"], page["offset"]) == (3, 1, 1)
        assert len(page["items"]) == 1
        assert page["items"][0] == {
            key: results[1][key]
            for key in ("run_id", "query", "created_at", "status", "n_results")
        }
        first = client.get("/api/runs", params={"limit": 1}).json()["items"][0]
        assert first["run_id"] == results[-1]["run_id"]
        assert client.get("/api/runs", params={"offset": 10}).json()["items"] == []


def test_document_browser_is_paginated_metadata_and_detail_is_authoritative(corpus):
    with client_for(corpus) as client:
        first = client.get("/api/documents", params={"limit": 2}).json()
        second = client.get("/api/documents", params={"limit": 2, "offset": 2}).json()
        assert (first["total"], first["limit"], first["offset"]) == (3, 2, 0)
        assert len(first["items"]) == 2
        assert len(second["items"]) == 1
        items = first["items"] + second["items"]
        assert {item["doc_id"] for item in items} == set(corpus[1])
        for item in items:
            assert "raw_text" not in item
            assert "text" not in item
            document = client.get(f"/api/documents/{item['doc_id']}").json()
            for key, value in corpus[1][item["doc_id"]].model_dump(mode="json").items():
                assert document[key] == value


@pytest.mark.parametrize("query,expected", [
    ("infrastructure", {"doc-a"}),
    ("%", {"doc-b"}),
    ("_", {"doc-b"}),
    ("' OR 1=1 --", set()),
])
def test_document_search_treats_sql_syntax_and_wildcards_as_literal(corpus, query, expected):
    with client_for(corpus) as client:
        response = client.get("/api/documents", params={"q": query})
        assert response.status_code == 200
        assert {item["doc_id"] for item in response.json()["items"]} == expected
        assert response.json()["total"] == len(expected)
        assert client.get("/api/documents").json()["total"] == 3


@pytest.mark.parametrize("payload", [
    {}, {"query": ""}, {"query": " \n\t "}, {"query": "x" * 4001},
    {"query": None}, {"query": 123}, {"query": []},
    {"query": "valid", "n_results": 0}, {"query": "valid", "n_results": 21},
    {"query": "valid", "n_results": True}, {"query": "valid", "n_results": "5"},
    {"query": "valid", "n_results": 1.5}, {"query": "valid", "unexpected": 1},
])
def test_malformed_submission_is_rejected_before_execution_or_audit(corpus, payload):
    def executor(**kwargs):
        pytest.fail("Malformed input must never invoke ARGUS")

    with client_for(corpus, executor) as client:
        assert client.post("/api/runs", json=payload).status_code == 422
        assert client.get("/api/runs").json()["total"] == 0


@pytest.mark.parametrize("path", ["/api/runs", "/api/documents"])
@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "bad"}])
def test_invalid_pagination_is_rejected(corpus, path, params):
    with client_for(corpus) as client:
        assert client.get(path, params=params).status_code == 422


def test_missing_run_document_and_run_local_alias_return_404(corpus):
    with client_for(corpus) as client:
        result = client.post("/api/runs", json={"query": "A question"}).json()
        for path in (
            "/api/runs/missing",
            "/api/runs/missing/sources",
            "/api/runs/missing/sources/S1",
            f"/api/runs/{result['run_id']}/sources/S999",
            "/api/documents/missing",
            "/api/documents/doc-a?run_id=missing",
        ):
            assert client.get(path).status_code == 404


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://localhost:3000.evil.example"])
def test_cross_origin_websites_cannot_trigger_local_model_runs(corpus, origin):
    with client_for(corpus) as client:
        response = client.post("/api/runs", json={"query": "A question"}, headers={"Origin": origin})
        assert response.status_code == 403
        assert client.get("/api/runs").json()["total"] == 0


@pytest.mark.parametrize("origin", ["http://localhost:3000", "http://127.0.0.1:3000"])
def test_expected_local_frontend_can_generate_briefs(corpus, origin):
    with client_for(corpus) as client:
        response = client.post("/api/runs", json={"query": "A question"}, headers={"Origin": origin})
        assert response.status_code == 201
        assert response.json()["status"] == "success"


def test_storage_failure_preserves_health_and_prevents_unrecorded_execution(tmp_path):
    def executor(*args):
        pytest.fail("Unavailable audit storage must prevent model execution")

    app = create_app(
        db_path=tmp_path / "missing-directory" / "argus.sqlite",
        run_executor=executor,
        health_reader=lambda: deepcopy(HEALTH),
    )
    with TestClient(app) as client:
        assert client.get("/api/health").json() == HEALTH
        for response in (
            client.post("/api/runs", json={"query": "A question"}),
            client.get("/api/runs"),
            client.get("/api/documents"),
        ):
            assert response.status_code == 503
            assert response.json()["detail"]
            assert "Traceback" not in response.text


def test_simultaneous_submission_is_rejected_without_duplicate_execution(corpus):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def executor(query, n_results, run_id, created_at):
        calls.append(run_id)
        entered.set()
        assert release.wait(timeout=5), "Test failed to release the waiting executor"
        return run_result(query, n_results, run_id, created_at)

    with client_for(corpus, executor) as client, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/api/runs", json={"query": "First question"})
        try:
            assert entered.wait(timeout=5)
            duplicate = client.post("/api/runs", json={"query": "Second question"})
            assert duplicate.status_code == 409
            history = client.get("/api/runs").json()
            assert history["total"] == 1
            assert history["items"][0]["query"] == "First question"
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 201
        # Completion releases the application lock for the next deliberate run.
        assert client.post("/api/runs", json={"query": "Third question"}).status_code == 201
        assert len(calls) == 2


def test_transport_owns_audit_identity_even_if_executor_returns_other_metadata(corpus):
    def executor(query, n_results, run_id, created_at):
        result = run_result(query, n_results, run_id, created_at)
        result.update(run_id="incorrect-id", query="wrong", created_at="invalid", n_results=99)
        return result

    with client_for(corpus, executor) as client:
        response = client.post("/api/runs", json={"query": "  Actual question  ", "n_results": 3})
        assert response.status_code == 201
        result = response.json()
        assert result["run_id"] != "incorrect-id"
        assert result["query"] == "Actual question"
        assert result["n_results"] == 3
        assert datetime.fromisoformat(result["created_at"]).tzinfo is not None
        assert client.get(f"/api/runs/{result['run_id']}").json() == result
        assert client.get("/api/runs/incorrect-id").status_code == 404


def test_invalid_json_and_overlong_document_search_return_validation_errors(corpus):
    with client_for(corpus) as client:
        assert client.post(
            "/api/runs", content="{", headers={"Content-Type": "application/json"}
        ).status_code == 422
        assert client.get("/api/documents", params={"q": "x" * 201}).status_code == 422
        assert client.get("/api/runs").json()["total"] == 0

"""Lightweight readiness checks use no inference or real Chroma client."""

import builtins
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import requests

from osint_agent.config import settings
from osint_agent.workspace import health


@pytest.fixture
def local_runtime(tmp_path, monkeypatch):
    db_path = tmp_path / "corpus.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY)")
        con.executemany("INSERT INTO documents VALUES (?)", [("a",), ("b",), ("c",)])
        con.execute("CREATE TABLE semantic_index_state (state_key TEXT PRIMARY KEY, status TEXT, error TEXT)")
        con.execute("INSERT INTO semantic_index_state VALUES ('semantic_index', 'current', NULL)")

    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    (chroma_path / "chroma.sqlite3").touch()
    monkeypatch.setattr(settings, "CHROMA_PATH", chroma_path)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "REASONING_MODEL", "reasoner")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "embedder:v1")
    calls = {"count": 0, "get": []}

    def count():
        calls["count"] += 1
        return 8

    monkeypatch.setitem(sys.modules, "osint_agent.storage.chroma", SimpleNamespace(
        get_document_collection=lambda: SimpleNamespace(count=count),
    ))

    def get(url, **kwargs):
        calls["get"].append((url, kwargs))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"models": [{"name": "reasoner:latest"}, {"name": "embedder:v1"}]},
        )

    def forbidden(*args, **kwargs):
        pytest.fail("Readiness must never invoke inference or an embedding request")

    monkeypatch.setattr(health.requests, "get", get)
    monkeypatch.setattr(health.requests, "post", forbidden)
    return db_path, calls


def test_health_reads_only_counts_recorded_state_and_installed_model_names(local_runtime, monkeypatch):
    db_path, calls = local_runtime
    statements = []
    connect = sqlite3.connect

    def observed_connect(*args, **kwargs):
        assert kwargs == {"uri": True}
        assert args[0].endswith("?mode=ro")
        con = connect(*args, **kwargs)
        con.set_trace_callback(statements.append)
        return con

    monkeypatch.setattr(health.sqlite3, "connect", observed_connect)
    result = health.read_health(db_path)

    assert result["status"] == "ready"
    assert result["sqlite"] == {"status": "ready", "document_count": 3}
    assert result["chroma"] == {"status": "ready", "record_count": 8}
    assert result["ollama"]["status"] == "ready"
    assert result["ollama"]["reasoning_model"] == "reasoner"
    assert result["ollama"]["embedding_model"] == "embedder:v1"
    assert result["index"]["status"] == "current"
    assert "retrieval time" in result["index"]["detail"]
    assert calls == {
        "count": 1,
        "get": [("http://127.0.0.1:11434/api/tags", {"timeout": 2, "allow_redirects": False})],
    }
    assert len(statements) == 3
    assert "COUNT(*)" in statements[0]
    assert all("raw_text" not in sql and "cleaned_text" not in sql for sql in statements)


@pytest.mark.parametrize("missing", ["reasoner", "embedder:v1"])
def test_missing_configured_model_is_distinct_from_disconnected_runtime(local_runtime, monkeypatch, missing):
    remaining = "embedder:v1" if missing == "reasoner" else "reasoner:latest"
    monkeypatch.setattr(health.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"models": [{"name": remaining}]},
    ))
    result = health.read_health(local_runtime[0])
    assert result["status"] == "degraded"
    assert result["ollama"]["status"] == "missing_models"
    assert result["ollama"]["detail"] == f"Configured models not installed: {missing}"


@pytest.mark.parametrize("failure", [requests.ConnectionError("not running"), requests.Timeout("slow runtime")])
def test_ollama_connection_failure_does_not_hide_corpus_counts(local_runtime, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(health.requests, "get", fail)
    result = health.read_health(local_runtime[0])
    assert result["status"] == "degraded"
    assert result["ollama"]["status"] == "unavailable"
    assert result["sqlite"]["document_count"] == 3
    assert result["chroma"]["record_count"] == 8


@pytest.mark.parametrize("payload", [{}, {"models": None}, {"models": [{}]}, {"models": ["not a model object"]}])
def test_malformed_model_listing_degrades_without_crashing(local_runtime, monkeypatch, payload):
    monkeypatch.setattr(health.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: payload,
    ))
    result = health.read_health(local_runtime[0])
    assert result["status"] == "degraded"
    assert result["ollama"]["status"] == "unavailable"
    assert result["ollama"]["detail"]


@pytest.mark.parametrize("host", [
    "https://remote.example", "http://192.0.2.1:11434", "http://localhost.evil.example",
    "http://user:password@localhost:11434", "http://[malformed", "http://localhost:bad",
    "http://localhost:0", "http://localhost:99999",
])
def test_nonlocal_or_malformed_ollama_configuration_never_sends_request(local_runtime, monkeypatch, host):
    monkeypatch.setattr(settings, "OLLAMA_HOST", host)
    assert not health.local_ollama_configured()
    result = health.read_health(local_runtime[0])
    assert result["ollama"]["status"] == "unavailable"
    assert local_runtime[1]["get"] == []


@pytest.mark.parametrize("host", ["http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"])
def test_loopback_ollama_addresses_are_supported(monkeypatch, host):
    monkeypatch.setattr(settings, "OLLAMA_HOST", host)
    assert health.local_ollama_configured()


def test_missing_chroma_is_not_initialized_during_health(local_runtime, monkeypatch, tmp_path):
    absent = tmp_path / "uninitialized-index"
    monkeypatch.setattr(settings, "CHROMA_PATH", absent)
    result = health.read_health(local_runtime[0])
    assert result["chroma"]["status"] == "unavailable"
    assert result["chroma"]["record_count"] is None
    assert local_runtime[1]["count"] == 0
    assert not absent.exists()


def test_missing_optional_chroma_dependency_keeps_health_available(local_runtime, monkeypatch):
    original_import = builtins.__import__

    def import_without_chroma(name, *args, **kwargs):
        if name == "osint_agent.storage.chroma":
            raise ImportError("optional Chroma dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_chroma)
    result = health.read_health(local_runtime[0])
    assert result["chroma"]["status"] == "unavailable"
    assert "ImportError" in result["chroma"]["detail"]
    assert result["sqlite"]["document_count"] == 3
    assert result["ollama"]["status"] == "ready"


@pytest.mark.parametrize("kind", ["missing", "corrupt", "uninitialized"])
def test_unreadable_sqlite_degrades_without_creating_or_repairing_corpus(local_runtime, tmp_path, kind):
    db_path = tmp_path / f"{kind}.sqlite"
    if kind == "corrupt":
        db_path.write_bytes(b"not a SQLite database")
    elif kind == "uninitialized":
        with sqlite3.connect(db_path):
            pass
    before = db_path.read_bytes() if db_path.exists() else None
    result = health.read_health(db_path)

    assert result["status"] == "degraded"
    assert result["sqlite"]["status"] == "unavailable"
    assert result["sqlite"]["document_count"] is None
    assert result["chroma"]["record_count"] == 8
    assert result["ollama"]["status"] == "ready"
    assert (db_path.read_bytes() if db_path.exists() else None) == before


def test_malformed_index_record_is_separate_from_readable_documents(local_runtime):
    db_path, _ = local_runtime
    with sqlite3.connect(db_path) as con:
        con.execute("DROP TABLE semantic_index_state")
        con.execute("CREATE TABLE semantic_index_state (state_key TEXT)")
    result = health.read_health(db_path)
    assert result["sqlite"] == {"status": "ready", "document_count": 3}
    assert result["index"]["status"] == "unavailable"
    assert "could not be read" in result["index"]["detail"]
    assert result["status"] == "degraded"


@pytest.mark.parametrize("status", ["stale", "incomplete", "invalid"])
def test_recorded_index_failures_remain_degraded(local_runtime, status):
    db_path, _ = local_runtime
    with sqlite3.connect(db_path) as con:
        con.execute("UPDATE semantic_index_state SET status=?, error='Recorded failure'", (status,))
    result = health.read_health(db_path)
    assert result["index"] == {"status": status, "detail": "Recorded failure"}
    assert result["status"] == "degraded"


def test_cache_reuses_one_health_observation_until_expiry(local_runtime, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(health.time, "monotonic", lambda: now[0])
    cached = health.CachedHealth(local_runtime[0])
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: cached(), range(16)))
    assert all(result == results[0] for result in results)
    assert local_runtime[1]["count"] == 1
    assert len(local_runtime[1]["get"]) == 1
    now[0] = 114.99
    assert cached() == results[0]
    assert local_runtime[1]["count"] == 1
    now[0] = 115.0
    assert cached()["status"] == "ready"
    assert local_runtime[1]["count"] == 2
    assert len(local_runtime[1]["get"]) == 2

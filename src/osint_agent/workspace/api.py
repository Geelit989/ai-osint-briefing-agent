"""Loopback FastAPI transport for the existing ARGUS application workflow."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from osint_agent.config import settings
from osint_agent.workspace.health import CachedHealth, local_ollama_configured
from osint_agent.workspace.storage import WorkspaceStore

LOGGER = logging.getLogger(__name__)


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000, strict=True)
    n_results: int = Field(default=5, ge=1, le=20, strict=True)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter an intelligence question.")
        return value.strip()


def _execute(query: str, n_results: int, run_id: str, created_at: str) -> dict:
    if not local_ollama_configured():
        raise ValueError("The analyst workspace requires local Ollama configuration.")
    from osint_agent.workspace.runs import execute_run
    return execute_run(query, n_results, run_id, created_at)


def create_app(db_path: Path | None = None, run_executor=None, health_reader=None) -> FastAPI:
    app = FastAPI(title="ARGUS Local Analyst Workspace", version="1.0")
    store = WorkspaceStore(Path(db_path or settings.DB_PATH))
    executor = run_executor or _execute
    health = health_reader or CachedHealth(store.db_path)
    run_lock = threading.Lock()
    try:
        store.initialize()
    except (sqlite3.Error, OSError):
        LOGGER.exception("Workspace audit storage initialization failed")

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    ports = {str(int(os.getenv("ARGUS_UI_PORT", "3000"))), str(int(os.getenv("ARGUS_API_PORT", "8000")))}
    origins = {f"http://{host}:{port}" for host in ("127.0.0.1", "localhost") for port in ports}

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin is not None and origin not in origins:
                return JSONResponse({"detail": "Only the local ARGUS workspace may submit runs."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request: Request, exc: sqlite3.Error):
        LOGGER.error("SQLite request failed: %s", exc, exc_info=exc)
        return JSONResponse({"detail": "Local SQLite storage is unavailable. Check API logs and corpus status."}, status_code=503)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        LOGGER.error("Workspace request failed: %s", exc, exc_info=exc)
        return JSONResponse({"detail": "ARGUS could not complete this request. Check the local API logs."}, status_code=500)

    def load_run(run_id: str) -> dict:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(404, "Run not found.")
        return run

    @app.get("/api/health")
    @app.get("/api/corpus/status")
    def get_health():
        return health()

    @app.post("/api/runs", status_code=201)
    def post_run(body: RunInput):
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "ARGUS is already processing a question. Wait for the current run to finish.")
        try:
            run_id = str(uuid4())
            created_at = datetime.now(timezone.utc).isoformat()
            pending = dict(
                run_id=run_id, query=body.query, created_at=created_at,
                n_results=body.n_results, status="running", retrieval=None,
                sufficiency=None, brief=None, claims=[], sources=[], evidence=[],
                grouping_manifest=[], known_contradictions=[],
                validation={"stages": [], "claim_support": None}, failure_detail=None,
                cli_equivalent="",
            )
            # Commit identity before long local inference; disconnects do not erase it.
            store.save_run(pending)
            try:
                result = executor(body.query, body.n_results, run_id, created_at)
            except Exception as exc:
                LOGGER.exception("ARGUS run %s failed before producing an application result", run_id)
                result = {**pending, "status": "error", "failure_detail": {
                    "type": type(exc).__name__,
                    "message": "ARGUS could not complete this run. Inspect diagnostics and local API logs.",
                }}
            result.update(run_id=run_id, query=body.query, created_at=created_at, n_results=body.n_results)
            # Only a successful, accepted domain product can enter the brief view.
            if result["status"] != "success":
                result["brief"] = None
                result["claims"] = []
            store.save_run(result)
            return result
        finally:
            run_lock.release()

    @app.get("/api/runs")
    def get_runs(limit: Annotated[int, Query(ge=1, le=100)] = 20,
                 offset: Annotated[int, Query(ge=0)] = 0):
        return store.list_runs(limit, offset)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return load_run(run_id)

    @app.get("/api/runs/{run_id}/sources")
    def get_run_sources(run_id: str):
        return {"items": load_run(run_id)["sources"]}

    @app.get("/api/runs/{run_id}/sources/{citation_id}")
    def get_run_source(run_id: str, citation_id: str):
        run = load_run(run_id)
        source = next((item for item in run["sources"] if item["source_id"] == citation_id), None)
        if source is None:
            raise HTTPException(404, "Citation not found in this run.")
        document_error = None
        try:
            document = store.get_document(source["doc_id"])
        except (sqlite3.Error, ValueError) as exc:
            LOGGER.warning("Stored citation document %s could not be read: %s", source["doc_id"], exc)
            document = None
            document_error = "The authoritative document could not be read. Recorded run evidence remains available."
        chunks = [item for item in run["evidence"] if item["doc_id"] == source["doc_id"] and item["chunk_id"] in source["chunk_ids"]]
        claims = [item for item in run["claims"] if citation_id in item["citations"]]
        return {"source": source, "document": document, "document_error": document_error, "chunks": chunks, "claims": claims}

    @app.get("/api/documents")
    def get_documents(limit: Annotated[int, Query(ge=1, le=100)] = 20,
                      offset: Annotated[int, Query(ge=0)] = 0,
                      q: Annotated[str, Query(max_length=200)] = ""):
        return store.list_documents(limit, offset, q.strip())

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str, run_id: str | None = None):
        document = store.get_document(document_id)
        if document is None:
            raise HTTPException(404, "Stored document not found.")
        document["relevant_chunks"] = (
            [item for item in load_run(run_id)["evidence"] if item["doc_id"] == document_id]
            if run_id else []
        )
        return document

    return app


app = create_app()

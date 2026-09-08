"""Cheap readiness observations. Semantic certification remains in retrieval."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from osint_agent.config import settings

LOGGER = logging.getLogger(__name__)


def local_ollama_configured() -> bool:
    try:
        parsed = urlparse(settings.OLLAMA_HOST)
        # Accessing port also validates malformed or out-of-range port values.
        port = parsed.port
        return (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                and not parsed.username and not parsed.password
                and (port is None or port > 0))
    except (TypeError, ValueError):
        return False


def read_health(db_path: Path) -> dict:
    result = {
        "status": "degraded",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "sqlite": {"status": "unavailable", "document_count": None},
        "chroma": {"status": "unavailable", "record_count": None},
        "ollama": {
            "status": "unavailable", "reasoning_model": settings.REASONING_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
        },
        "index": {"status": "unknown", "detail": "No recorded index certification is available."},
    }
    try:
        with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) as con:
            count = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            result["sqlite"] = {"status": "ready", "document_count": count}
            try:
                table = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_index_state'").fetchone()
                if table:
                    row = con.execute("""
                        SELECT status, error FROM semantic_index_state
                        WHERE state_key='semantic_index'
                    """).fetchone()
                    if row:
                        result["index"] = {
                            "status": row[0],
                            "detail": row[1] or "Recorded index state. Full corpus and compatibility checks run at retrieval time.",
                        }
            except sqlite3.Error as exc:
                result["index"] = {"status": "unavailable", "detail": f"Recorded index state could not be read: {exc}"}
    except (sqlite3.Error, OSError) as exc:
        result["sqlite"]["detail"] = str(exc)

    try:
        if not (settings.CHROMA_PATH / "chroma.sqlite3").is_file():
            raise FileNotFoundError("The local Chroma index has not been initialized.")
        # Lazy import allows the shell to load when Chroma/dependencies fail.
        from osint_agent.storage.chroma import get_document_collection
        result["chroma"] = {"status": "ready", "record_count": get_document_collection().count()}
    except Exception as exc:
        result["chroma"]["detail"] = f"{type(exc).__name__}: {exc}"

    try:
        if not local_ollama_configured():
            raise ValueError("This workspace requires Ollama on a local loopback HTTP address.")
        # Listing installed models invokes no inference, embeddings, or model load.
        response = requests.get(f"{settings.OLLAMA_HOST.rstrip('/')}/api/tags", timeout=2, allow_redirects=False)
        response.raise_for_status()
        models = response.json()["models"]
        names = {item["name"] for item in models}
        def installed(name: str) -> bool:
            return name in names or (":" not in name and f"{name}:latest" in names)
        missing = [name for name in (settings.REASONING_MODEL, settings.EMBEDDING_MODEL) if not installed(name)]
        result["ollama"]["status"] = "missing_models" if missing else "ready"
        if missing:
            result["ollama"]["detail"] = "Configured models not installed: " + ", ".join(missing)
    except Exception as exc:
        result["ollama"]["detail"] = f"{type(exc).__name__}: {exc}"
    if all(result[key]["status"] == "ready" for key in ("sqlite", "chroma", "ollama")) and result["index"]["status"] == "current":
        result["status"] = "ready"
    return result


class CachedHealth:
    """Cache status for 15 seconds, including across concurrent page requests."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._expires = 0.0
        self._value = None

    def __call__(self) -> dict:
        with self._lock:
            if self._value is None or time.monotonic() >= self._expires:
                self._value = read_health(self.db_path)
                self._expires = time.monotonic() + 15
            return self._value

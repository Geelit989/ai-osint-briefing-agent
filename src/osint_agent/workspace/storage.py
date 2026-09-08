"""Small, additive SQLite audit store alongside authoritative ARGUS documents."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from osint_agent.storage.sqlite import get_document


class WorkspaceStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        # No corpus initialization, migration, index writes, or article copies.
        with sqlite3.connect(self.db_path, timeout=10) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS analyst_runs (
                    run_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    n_results INTEGER NOT NULL,
                    result_json TEXT NOT NULL
                )
            """)
            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_analyst_runs_created_at
                ON analyst_runs(created_at DESC, run_id DESC)
            """)

    def save_run(self, result: dict) -> None:
        with sqlite3.connect(self.db_path, timeout=10) as con:
            con.execute("""
                INSERT INTO analyst_runs
                    (run_id, query, created_at, status, n_results, result_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status, result_json=excluded.result_json
            """, (
                result["run_id"], result["query"], result["created_at"],
                result["status"], result["n_results"],
                json.dumps(result, ensure_ascii=False, allow_nan=False),
            ))

    def _read_connection(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        return con

    def get_run(self, run_id: str) -> dict | None:
        with self._read_connection() as con:
            row = con.execute(
                "SELECT result_json FROM analyst_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_runs(self, limit: int, offset: int) -> dict:
        with self._read_connection() as con:
            total = con.execute("SELECT COUNT(*) FROM analyst_runs").fetchone()[0]
            rows = con.execute("""
                SELECT run_id, query, created_at, status, n_results
                FROM analyst_runs ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
        return dict(items=[dict(row) for row in rows], total=total, limit=limit, offset=offset)

    def list_documents(self, limit: int, offset: int, q: str = "") -> dict:
        # Search only metadata; never load corpus text for a source-list request.
        search = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where = ""
        params: tuple = ()
        if q:
            where = "WHERE title LIKE ? ESCAPE '\\' OR source LIKE ? ESCAPE '\\' OR doc_id LIKE ? ESCAPE '\\'"
            params = (f"%{search}%",) * 3
        with self._read_connection() as con:
            total = con.execute(f"SELECT COUNT(*) FROM documents {where}", params).fetchone()[0]
            rows = con.execute(f"""
                SELECT doc_id, title, source, provider, source_type,
                       published_date, retrieved_at, url
                FROM documents {where}
                ORDER BY published_date DESC, doc_id LIMIT ? OFFSET ?
            """, (*params, limit, offset)).fetchall()
        return dict(items=[dict(row) for row in rows], total=total, limit=limit, offset=offset)

    def get_document(self, doc_id: str) -> dict | None:
        if not self.db_path.is_file():
            raise sqlite3.OperationalError("Authoritative SQLite database is unavailable")
        document = get_document(doc_id, self.db_path)
        return document.model_dump(mode="json") if document else None

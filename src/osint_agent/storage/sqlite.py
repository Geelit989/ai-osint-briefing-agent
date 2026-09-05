"""SQLite access for ARGUS's authoritative document store."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from osint_agent.config import settings
from osint_agent.models.document import Document


def _create_documents_table(con: sqlite3.Connection) -> None:
    """Create the documents table."""

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            provider TEXT NOT NULL,
            source_type TEXT NOT NULL,
            published_date TEXT,
            event_time TEXT,
            retrieved_at TEXT NOT NULL,
            url TEXT,
            raw_text TEXT NOT NULL,
            cleaned_text TEXT NOT NULL,
            meta_data TEXT NOT NULL DEFAULT '{}',
            contradiction_group TEXT,
            contradiction_position TEXT
        )
        """
    )

    # Narrow additive migration for databases created before Task 5.
    columns = {
        row[1] for row in con.execute("PRAGMA table_info(documents)").fetchall()
    }
    additions = {
        "event_time": "TEXT",
        "contradiction_group": "TEXT",
        "contradiction_position": "TEXT",
    }
    for name, declaration in additions.items():
        if name not in columns:
            con.execute(f"ALTER TABLE documents ADD COLUMN {name} {declaration}")


def _create_semantic_index_state_table(con: sqlite3.Connection) -> None:
    """Create durable commit metadata for the derived semantic index."""

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS semantic_index_state (
            state_key TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            attempt_id TEXT,
            corpus_digest TEXT,
            compatibility_manifest TEXT,
            compatibility_fingerprint TEXT,
            expected_chunks_digest TEXT,
            expected_chunk_count INTEGER,
            collection_id TEXT,
            embedding_dimension INTEGER,
            error TEXT,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )


def _create_entities_table(con: sqlite3.Connection) -> None:
    """Create the entities table."""

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ent_text TEXT NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            label TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            FOREIGN KEY (doc_id)
                REFERENCES documents (doc_id)
                ON DELETE CASCADE,
            UNIQUE (
                doc_id,
                start_char,
                end_char,
                label
            )
        )
        """
    )


def _create_indexes(con: sqlite3.Connection) -> None:
    """Create supporting indexes."""

    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_provider
        ON documents(provider)
        """
    )

    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_published_date
        ON documents(published_date)
        """
    )

    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_entities_doc_id
        ON entities(doc_id)
        """
    )

    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_entities_text
        ON entities(ent_text)
        """
    )


def create_db(db_path: str | Path | None = None) -> None:
    """Create the ARGUS SQLite schema if it does not already exist."""

    db_path = Path(db_path or settings.DB_PATH)

    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON;")

        _create_documents_table(con)
        _create_entities_table(con)
        _create_semantic_index_state_table(con)
        _create_indexes(con)


def mark_semantic_index_stale(
    reason: str,
    db_path: str | Path | None = None,
) -> None:
    """Invalidate an existing success marker before out-of-band index writes."""

    active_path = Path(db_path or settings.DB_PATH)
    if not active_path.is_file():
        return
    with sqlite3.connect(active_path) as con:
        exists = con.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'semantic_index_state'
            """
        ).fetchone()
        if exists is None:
            return
        con.execute(
            """
            UPDATE semantic_index_state
            SET status = 'stale', error = ?, updated_at = ?
            WHERE state_key = 'semantic_index'
            """,
            (reason, datetime.now(timezone.utc).isoformat()),
        )


def _document_projection(con: sqlite3.Connection) -> str:
    """Return a projection compatible with current and legacy schemas."""

    columns = {
        row[1] for row in con.execute("PRAGMA table_info(documents)").fetchall()
    }

    def column_or_null(name: str) -> str:
        return name if name in columns else f"NULL AS {name}"

    return ",\n                ".join(
        [
            "doc_id",
            "title",
            "source",
            "provider",
            "source_type",
            "published_date",
            column_or_null("event_time"),
            "retrieved_at",
            "url",
            "raw_text",
            "cleaned_text",
            column_or_null("meta_data"),
            column_or_null("contradiction_group"),
            column_or_null("contradiction_position"),
        ]
    )


def _document_from_row(row: sqlite3.Row) -> Document:
    metadata = json.loads(row["meta_data"] or "{}")
    if not isinstance(metadata, dict):
        raise ValueError("documents.meta_data must contain a JSON object")
    return Document(
        doc_id=row["doc_id"],
        title=row["title"],
        source=row["source"],
        provider=row["provider"],
        source_type=row["source_type"],
        published_date=row["published_date"],
        event_time=row["event_time"],
        retrieved_at=row["retrieved_at"],
        url=row["url"],
        raw_text=row["raw_text"],
        text=row["cleaned_text"],
        meta_data=metadata,
        contradiction_group=row["contradiction_group"],
        contradiction_position=row["contradiction_position"],
    )


def get_document(
    doc_id: str,
    db_path: str | Path | None = None,
) -> Document | None:
    """Load one stored document from SQLite."""

    with sqlite3.connect(db_path or settings.DB_PATH) as con:
        con.row_factory = sqlite3.Row
        projection = _document_projection(con)
        row = con.execute(
            f"""
            SELECT {projection}
            FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        ).fetchone()

    if row is None:
        return None

    return _document_from_row(row)


def get_documents_by_ids(
    doc_ids: set[str],
    db_path: str | Path | None = None,
) -> dict[str, Document]:
    """Load authoritative documents for a set of retrieval document IDs.

    A missing database or uninitialized schema is treated as no resolved
    records. The evidence-identity boundary then keeps unresolved documents
    separately countable unless another deterministic signal proves identity.
    """

    active_path = Path(db_path or settings.DB_PATH)
    if not doc_ids or not active_path.is_file():
        return {}

    placeholders = ", ".join("?" for _ in doc_ids)
    try:
        with sqlite3.connect(active_path) as con:
            con.row_factory = sqlite3.Row
            projection = _document_projection(con)
            rows = con.execute(
                f"""
                SELECT {projection}
                FROM documents
                WHERE doc_id IN ({placeholders})
                """,
                tuple(sorted(doc_ids)),
            ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).lower():
            raise
        return {}

    documents = [_document_from_row(row) for row in rows]
    return {document.doc_id: document for document in documents}


def get_documents(db_path: str | Path | None = None) -> list[Document]:
    """Load all stored documents from SQLite."""

    with sqlite3.connect(db_path or settings.DB_PATH) as con:
        con.row_factory = sqlite3.Row
        projection = _document_projection(con)
        rows = con.execute(
            f"""
            SELECT {projection}
            FROM documents
            ORDER BY doc_id
            """
        ).fetchall()

    return [_document_from_row(row) for row in rows]


if __name__ == "__main__":
    create_db()

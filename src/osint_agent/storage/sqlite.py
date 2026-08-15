"""SQLite schema creation for the ARGUS application."""

from pathlib import Path
import sqlite3

from osint_agent.config import settings


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
            retrieved_at TEXT NOT NULL,
            url TEXT,
            raw_text TEXT NOT NULL,
            cleaned_text TEXT NOT NULL,
            meta_data TEXT NOT NULL DEFAULT '{}'
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


def create_db(db_path: str | Path = settings.DB_PATH) -> None:
    """Create the ARGUS SQLite schema if it does not already exist."""

    db_path = Path(db_path)

    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON;")

        _create_documents_table(con)
        _create_entities_table(con)
        _create_indexes(con)



    import sqlite3

from osint_agent.config import settings
from osint_agent.models.document import Document


def get_document(doc_id: str) -> Document | None:
    """Load one stored document from SQLite."""

    with sqlite3.connect(settings.DB_PATH) as con:
        con.row_factory = sqlite3.Row

        row = con.execute(
            """
            SELECT
                doc_id,
                title,
                source,
                provider,
                source_type,
                published_date,
                url,
                raw_text,
                cleaned_text
            FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        ).fetchone()

    if row is None:
        return None

    return Document(
        doc_id=row["doc_id"],
        title=row["title"],
        source=row["source"],
        provider=row["provider"],
        source_type=row["source_type"],
        published_date=row["published_date"],
        url=row["url"],
        raw_text=row["raw_text"],
        text=row["cleaned_text"],
    )


def get_documents() -> list[Document]:
    """Load all stored documents from SQLite."""

    with sqlite3.connect(settings.DB_PATH) as con:
        con.row_factory = sqlite3.Row

        rows = con.execute(
            """
            SELECT
                doc_id,
                title,
                source,
                provider,
                source_type,
                published_date,
                url,
                raw_text,
                cleaned_text
            FROM documents
            ORDER BY published_date
            """
        ).fetchall()

    return [
        Document(
            doc_id=row["doc_id"],
            title=row["title"],
            source=row["source"],
            provider=row["provider"],
            source_type=row["source_type"],
            published_date=row["published_date"],
            url=row["url"],
            raw_text=row["raw_text"],
            text=row["cleaned_text"],
        )
        for row in rows
    ]


if __name__ == "__main__":
    create_db()
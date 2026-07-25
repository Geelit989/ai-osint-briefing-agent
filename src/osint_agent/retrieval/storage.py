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


if __name__ == "__main__":
    create_db()
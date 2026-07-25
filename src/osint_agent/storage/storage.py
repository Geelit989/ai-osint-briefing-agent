import sqlite3

from osint_agent.config import settings


### Create SQLite datebase


def create_db() -> None:
    with sqlite3.connect(settings.DB_PATH) as con:

        con.execute("PRAGMA foreign_keys = ON;")


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


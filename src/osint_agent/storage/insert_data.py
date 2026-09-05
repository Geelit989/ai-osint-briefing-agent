import json
import sqlite3
from datetime import datetime, timezone

from osint_agent.models.document import Document, Entity
from osint_agent.storage.sqlite import _create_semantic_index_state_table


DOCUMENT_UPSERT_SQL = """
INSERT INTO documents (
    doc_id,
    title,
    source,
    provider,
    source_type,
    published_date,
    event_time,
    retrieved_at,
    url,
    raw_text,
    cleaned_text,
    meta_data,
    contradiction_group,
    contradiction_position
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(doc_id) DO UPDATE SET
    title = excluded.title,
    source = excluded.source,
    provider = excluded.provider,
    source_type = excluded.source_type,
    published_date = excluded.published_date,
    event_time = excluded.event_time,
    retrieved_at = excluded.retrieved_at,
    url = excluded.url,
    raw_text = excluded.raw_text,
    cleaned_text = excluded.cleaned_text,
    meta_data = excluded.meta_data,
    contradiction_group = excluded.contradiction_group,
    contradiction_position = excluded.contradiction_position
"""


ENTITY_INSERT_SQL = """
INSERT INTO entities (
    ent_text,
    start_char,
    end_char,
    label,
    doc_id
)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(
    doc_id,
    start_char,
    end_char,
    label
)
DO NOTHING
"""


def prepare_document_row(document: Document) -> tuple:
    return (
        document.doc_id,
        document.title,
        document.source,
        document.provider,
        document.source_type,
        (
            document.published_date.isoformat()
            if document.published_date
            else None
        ),
        document.event_time.isoformat() if document.event_time else None,
        document.retrieved_at.isoformat(),
        document.url,
        document.raw_text,
        document.text,
        json.dumps(document.meta_data),
        document.contradiction_group,
        document.contradiction_position,
    )


def prepare_entity_row(entity: Entity) -> tuple:
    return (
        entity.ent_text,
        entity.start_char,
        entity.end_char,
        entity.label,
        entity.doc_id,
    )


def upsert_document(
    con: sqlite3.Connection,
    document: Document,
) -> None:
    # Existing local databases receive the narrow Task 5 additive columns.
    columns = {
        row[1] for row in con.execute("PRAGMA table_info(documents)").fetchall()
    }
    for name in ("event_time", "contradiction_group", "contradiction_position"):
        if name not in columns:
            con.execute(f"ALTER TABLE documents ADD COLUMN {name} TEXT")
    con.execute(
        DOCUMENT_UPSERT_SQL,
        prepare_document_row(document),
    )
    _create_semantic_index_state_table(con)
    con.execute(
        """
        UPDATE semantic_index_state
        SET status = 'stale',
            error = 'authoritative document write invalidated the index',
            updated_at = ?
        WHERE state_key = 'semantic_index'
        """,
        (datetime.now(timezone.utc).isoformat(),),
    )


def insert_entities(
    con: sqlite3.Connection,
    entities: list[Entity],
) -> None:
    if not entities:
        return

    con.executemany(
        ENTITY_INSERT_SQL,
        [prepare_entity_row(entity) for entity in entities],
    )

import json
import sqlite3

from osint_agent.models.document import Document, Entity


DOCUMENT_UPSERT_SQL = """
INSERT INTO documents (
    doc_id,
    title,
    source,
    provider,
    source_type,
    published_date,
    retrieved_at,
    url,
    raw_text,
    cleaned_text,
    meta_data
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(doc_id) DO UPDATE SET
    title = excluded.title,
    source = excluded.source,
    provider = excluded.provider,
    source_type = excluded.source_type,
    published_date = excluded.published_date,
    retrieved_at = excluded.retrieved_at,
    url = excluded.url,
    raw_text = excluded.raw_text,
    cleaned_text = excluded.cleaned_text,
    meta_data = excluded.meta_data
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
        document.retrieved_at.isoformat(),
        document.url,
        document.raw_text,
        document.text,
        json.dumps(document.meta_data),
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
    con.execute(
        DOCUMENT_UPSERT_SQL,
        prepare_document_row(document),
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
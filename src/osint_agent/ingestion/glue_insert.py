import sqlite3
from dataclasses import dataclass

from spacy.language import Language

from osint_agent.config import settings
from osint_agent.models.document import Document
from osint_agent.extraction.ner import extract_entities
from osint_agent.retrieval.insert_data import upsert_document
from osint_agent.retrieval.insert_data import insert_entities


@dataclass
class IngestionResult:
    doc_id: str
    entity_count: int
    document_persisted: bool


def ingest_document(
    document: Document,
    nlp: Language,
) -> IngestionResult:
    """Persist and enrich one normalized ARGUS document."""

    entities = extract_entities(document, nlp)

    with sqlite3.connect(settings.DB_PATH) as con:
        con.execute("PRAGMA foreign_keys = ON;")

        upsert_document(con, document)
        insert_entities(con, entities)

    return IngestionResult(
        doc_id=document.doc_id,
        entity_count=len(entities),
        document_persisted=True,
    )
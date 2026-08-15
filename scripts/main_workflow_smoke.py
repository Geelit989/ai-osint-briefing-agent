"""Run the end-to-end Currents ingestion and query workflow.

Workflow:
1. Retrieve Currents articles.
2. Normalize each article into a Document.
3. Run named entity recognition.
4. Insert the Document and Entities into SQLite.
5. Query SQLite to verify the persisted records.
"""

import logging
import sqlite3
import sys

from osint_agent.config import settings
from osint_agent.extraction.ner import (
    extract_entities,
    load_ner_model,
)
from osint_agent.providers.currents import (
    search_currents,
)
from osint_agent.storage.insert_data import (
    insert_entities,
    upsert_document,
)


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure application logging for the smoke test."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def query_document(
    con: sqlite3.Connection,
    doc_id: str,
) -> sqlite3.Row | None:
    """Retrieve one ingested document from SQLite."""

    return con.execute(
        """
        SELECT
            doc_id,
            title,
            source,
            provider,
            source_type,
            published_date,
            retrieved_at,
            url
        FROM documents
        WHERE doc_id = ?
        """,
        (doc_id,),
    ).fetchone()


def query_entities(
    con: sqlite3.Connection,
    doc_id: str,
) -> list[sqlite3.Row]:
    """Retrieve all entities associated with one document."""

    return con.execute(
        """
        SELECT
            entity_id,
            ent_text,
            label,
            start_char,
            end_char
        FROM entities
        WHERE doc_id = ?
        ORDER BY start_char
        """,
        (doc_id,),
    ).fetchall()


def main() -> None:
    """Execute the Currents ingestion smoke test."""

    configure_logging()

    query = "Iran"
    limit = 20

    logger.info("Loading spaCy NER model.")
    nlp = load_ner_model()

    logger.info(
        "Searching Currents for query=%r with limit=%d.",
        query,
        limit,
    )

    articles = search_currents(
        query=query,
        limit=limit,
    )

    if not articles:
        logger.warning(
            "Currents returned no articles for query=%r.",
            query,
        )
        return

    logger.info(
        "Currents returned %d article(s).",
        len(articles),
    )

    with sqlite3.connect(settings.DB_PATH) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON;")

        for normalized_doc in articles:
            try:
                logger.info(
                    "Processing document doc_id=%s title=%r.",
                    normalized_doc.doc_id,
                    normalized_doc.title,
                )

                entities = extract_entities(
                    normalized_doc,
                    nlp,
                )

                logger.info(
                    "Extracted %d entities from doc_id=%s.",
                    len(entities),
                    normalized_doc.doc_id,
                )

                upsert_document(
                    con,
                    normalized_doc,
                )

                insert_entities(
                    con,
                    entities,
                )

                document_row = query_document(
                    con,
                    normalized_doc.doc_id,
                )

                entity_rows = query_entities(
                    con,
                    normalized_doc.doc_id,
                )

                if document_row is None:
                    raise RuntimeError(
                        "Document could not be verified after insertion: "
                        f"{normalized_doc.doc_id}"
                    )

                logger.info(
                    "Verified document in SQLite: %s",
                    dict(document_row),
                )

                logger.info(
                    "Verified %d associated entity row(s) for doc_id=%s.",
                    len(entity_rows),
                    normalized_doc.doc_id,
                )

                for entity_row in entity_rows:
                    logger.info(
                        "Verified entity: %s",
                        dict(entity_row),
                    )

            except Exception:
                logger.exception(
                    "Failed to ingest document doc_id=%s.",
                    normalized_doc.doc_id,
                )
                raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.critical(
            "Currents ingestion smoke test failed.",
            exc_info=True,
        )
        sys.exit(1)

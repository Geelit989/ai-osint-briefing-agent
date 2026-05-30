"""Run a small end-to-end smoke test for the OSINT workflow."""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from osint_agent.retrieval.storage import create_db
from osint_agent.ingestion.load_samples import PROCESSED_PATH, process_sample_articles
from osint_agent.extraction.ner import ENT_PATH, extract_entities, save_entities
from osint_agent.retrieval.insert_data import (
    DB_PATH,
    DOC_TABLE,
    ENT_TABLE,
    documents_query,
    entities_query,
    insert_data_query,
    prepare_documents,
    prepare_entities,
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure simple progress logging for the workflow."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def reset_tables(connection: sqlite3.Connection) -> None:
    """Clear existing smoke-test data so the script can be rerun."""

    connection.execute(f"DELETE FROM {ENT_TABLE}")
    connection.execute(f"DELETE FROM {DOC_TABLE}")
    connection.commit()


def print_one_document(connection: sqlite3.Connection) -> None:
    """Query the database and print one stored document row."""

    connection.row_factory = sqlite3.Row
    row = connection.execute(
        f"""
        SELECT doc_id, title, source, published_date, url, cleaned_text
        FROM {DOC_TABLE}
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        LOGGER.warning("No rows found in %s.", DOC_TABLE)
        return

    print("\nSample database row:")
    print(dict(row))


def run_workflow() -> int:
    """Create the DB, process samples, extract entities, insert, and query."""

    try:
        LOGGER.info("Starting the OSINT Agent smoke-test workflow.")

        LOGGER.info("Step 1/5: Creating database tables.")
        create_db()

        LOGGER.info("Step 2/5: Loading and cleaning sample articles.")
        documents = process_sample_articles()
        LOGGER.info("Saved %s cleaned article(s) to %s.", len(documents), PROCESSED_PATH)

        LOGGER.info("Step 3/5: Running NER over cleaned articles.")
        entities = extract_entities(PROCESSED_PATH)
        save_entities(entities, ENT_PATH)
        LOGGER.info("Saved %s extracted entit(y/ies) to %s.", len(entities), ENT_PATH)

        document_rows = [prepare_documents(document) for document in documents]
        entity_rows = [prepare_entities(entity) for entity in entities]

        LOGGER.info("Step 4/5: Inserting documents and entities into SQLite.")
        with sqlite3.connect(DB_PATH) as connection:
            reset_tables(connection)
            insert_data_query(connection, documents_query, DOC_TABLE, document_rows)
            insert_data_query(connection, entities_query, ENT_TABLE, entity_rows)

            LOGGER.info("Step 5/5: Querying one stored document row.")
            print_one_document(connection)

        LOGGER.info("Workflow completed successfully.")
        return 0

    except (FileNotFoundError, sqlite3.Error, ValueError) as error:
        LOGGER.error("Workflow failed: %s", error)
        return 1
    except Exception:
        LOGGER.exception("Workflow failed unexpectedly.")
        return 1


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(run_workflow())

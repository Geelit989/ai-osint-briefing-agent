"""Manage the local ARGUS development SQLite database.

Commands:
    create      Create missing tables without deleting existing data.
    truncate    Delete all records while preserving the current schema.
    reset       Delete the database file and recreate the latest schema.
    inspect     Display the current tables, columns, and row counts.

Examples:
    python scripts/manage_dev_db.py create
    python scripts/manage_dev_db.py truncate
    python scripts/manage_dev_db.py reset
    python scripts/manage_dev_db.py inspect
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

from osint_agent.config import settings


logger = logging.getLogger(__name__)

DOCUMENTS_TABLE = "documents"
ENTITIES_TABLE = "entities"


DOCUMENTS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
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
    meta_data TEXT NOT NULL DEFAULT '{{}}'
)
"""


ENTITIES_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {ENTITIES_TABLE} (
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ent_text TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    label TEXT NOT NULL,
    doc_id TEXT NOT NULL,

    FOREIGN KEY (doc_id)
        REFERENCES {DOCUMENTS_TABLE} (doc_id)
        ON DELETE CASCADE,

    UNIQUE (
        doc_id,
        start_char,
        end_char,
        label
    )
)
"""


INDEX_STATEMENTS = [
    f"""
    CREATE INDEX IF NOT EXISTS idx_documents_provider
    ON {DOCUMENTS_TABLE}(provider)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_documents_source_type
    ON {DOCUMENTS_TABLE}(source_type)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_documents_published_date
    ON {DOCUMENTS_TABLE}(published_date)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_entities_doc_id
    ON {ENTITIES_TABLE}(doc_id)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_entities_text
    ON {ENTITIES_TABLE}(ent_text)
    """,
]


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def get_db_path() -> Path:
    """Return the configured SQLite database path."""

    db_path = Path(settings.DB_PATH).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return db_path


def open_connection(db_path: Path) -> sqlite3.Connection:
    """Open a configured SQLite connection."""

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")

    return con


def create_schema(db_path: Path) -> None:
    """Create the latest ARGUS development database schema."""

    logger.info("Creating database schema at %s.", db_path)

    with open_connection(db_path) as con:
        con.execute(DOCUMENTS_TABLE_SQL)
        con.execute(ENTITIES_TABLE_SQL)

        for statement in INDEX_STATEMENTS:
            con.execute(statement)

    logger.info("Database schema created successfully.")


def truncate_database(db_path: Path) -> None:
    """Delete all records while preserving tables and indexes."""

    if not db_path.exists():
        logger.warning(
            "Database does not exist at %s. Creating it instead.",
            db_path,
        )
        create_schema(db_path)
        return

    logger.info("Truncating development database at %s.", db_path)

    with open_connection(db_path) as con:
        # Delete child records before parent records.
        con.execute(f"DELETE FROM {ENTITIES_TABLE}")
        con.execute(f"DELETE FROM {DOCUMENTS_TABLE}")

        # Reset AUTOINCREMENT values.
        con.execute(
            """
            DELETE FROM sqlite_sequence
            WHERE name IN (?, ?)
            """,
            (
                ENTITIES_TABLE,
                DOCUMENTS_TABLE,
            ),
        )

    logger.info("Development database truncated successfully.")


def reset_database(db_path: Path) -> None:
    """Delete the database file and recreate the latest schema."""

    if db_path.exists():
        logger.warning(
            "Deleting development database at %s.",
            db_path,
        )
        db_path.unlink()
    else:
        logger.info(
            "No existing database found at %s.",
            db_path,
        )

    create_schema(db_path)

    logger.info("Development database reset successfully.")


def get_table_names(
    con: sqlite3.Connection,
) -> list[str]:
    """Return user-created SQLite table names."""

    rows = con.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()

    return [row["name"] for row in rows]


def inspect_database(db_path: Path) -> None:
    """Log tables, columns, constraints, and row counts."""

    if not db_path.exists():
        logger.warning(
            "Database does not exist at %s.",
            db_path,
        )
        return

    logger.info("Inspecting database at %s.", db_path)

    with open_connection(db_path) as con:
        tables = get_table_names(con)

        if not tables:
            logger.warning("Database contains no application tables.")
            return

        for table_name in tables:
            count_row = con.execute(
                f"SELECT COUNT(*) AS count FROM {table_name}"
            ).fetchone()

            row_count = count_row["count"]

            logger.info(
                "Table: %s | Rows: %d",
                table_name,
                row_count,
            )

            columns = con.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()

            for column in columns:
                logger.info(
                    (
                        "  Column: %-20s "
                        "Type: %-10s "
                        "Nullable: %-5s "
                        "Primary key: %s"
                    ),
                    column["name"],
                    column["type"],
                    "no" if column["notnull"] else "yes",
                    "yes" if column["pk"] else "no",
                )

            foreign_keys = con.execute(
                f"PRAGMA foreign_key_list({table_name})"
            ).fetchall()

            for foreign_key in foreign_keys:
                logger.info(
                    "  Foreign key: %s -> %s.%s | on_delete=%s",
                    foreign_key["from"],
                    foreign_key["table"],
                    foreign_key["to"],
                    foreign_key["on_delete"],
                )

            indexes = con.execute(
                f"PRAGMA index_list({table_name})"
            ).fetchall()

            for index in indexes:
                logger.info(
                    "  Index: %s | unique=%s",
                    index["name"],
                    bool(index["unique"]),
                )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Create, truncate, reset, or inspect the "
            "ARGUS development SQLite database."
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "create",
            "truncate",
            "reset",
            "inspect",
        ],
        help="Database operation to perform.",
    )

    return parser


def main() -> None:
    """Execute the requested database operation."""

    configure_logging()

    parser = build_parser()
    args = parser.parse_args()

    db_path = get_db_path()

    logger.info(
        "Using configured database path: %s",
        db_path,
    )

    if args.command == "create":
        create_schema(db_path)

    elif args.command == "truncate":
        truncate_database(db_path)

    elif args.command == "reset":
        reset_database(db_path)

    elif args.command == "inspect":
        inspect_database(db_path)

    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except (sqlite3.Error, OSError):
        logger.exception(
            "Development database operation failed."
        )
        sys.exit(1)
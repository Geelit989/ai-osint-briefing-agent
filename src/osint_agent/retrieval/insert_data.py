# This module is responsible for inserting data into the SQLite database.

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("/Users/duk3y6/Desktop/aia-osint-briefer/ai-osint-briefing-agent/osint_sys.db")
SRC_PATH = Path("/Users/duk3y6/Desktop/aia-osint-briefer/ai-osint-briefing-agent/data/processed/cleaned_articles.json")
ENT_PATH = Path("/Users/duk3y6/Desktop/aia-osint-briefer/ai-osint-briefing-agent/data/processed/extracted_entities.json")

DOC_TABLE = "documents"
ENT_TABLE = "entities"


documents_query = f"""
INSERT INTO {DOC_TABLE} (
    doc_id, title, source, published_date, url, raw_text, cleaned_text, meta_data
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

entities_query = f"""
INSERT INTO {ENT_TABLE} (
    entity_id, ent_text, start_char, end_char, label, doc_id
)
VALUES (?, ?, ?, ?, ?, ?)
"""


def get_data(src: Path | str) -> list[dict]:
    src = Path(src)

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Data type: {type(data)}, Number of articles: {len(data)}")
    return data


def prepare_documents(data: dict) -> tuple:
    return (
        data.get("doc_id"),
        data.get("title"),
        data.get("source"),
        data.get("published_date"),
        data.get("url"),
        data.get("raw_text"),
        data.get("text"),
        json.dumps(data.get("meta_data", {})),
        )

def prepare_entities(data: dict) -> tuple:
    return (
        data.get("entity_id"),
        data.get("ent_text"),
        data.get("start_char"),
        data.get("end_char"),
        data.get("label"),
        data.get("doc_id"),
    )   


def insert_data_query(
    con: sqlite3.Connection,
    sql_query: str,
    table_name: str,
    rows: list[tuple],
) -> None:
    try:
        con.executemany(sql_query, rows)
        con.commit()
        print(f"Data inserted successfully into {table_name} table.")
    except sqlite3.Error as e:
        con.rollback()
        print(f"Error inserting data into {table_name} table: {e}")


if __name__ == "__main__":
    doc_data = get_data(SRC_PATH)
    ent_data = get_data(ENT_PATH)

    document_rows = [prepare_documents(doc) for doc in doc_data]
    ent_rows = [prepare_entities(ent) for ent in ent_data]

    with sqlite3.connect(DB_PATH) as con:
        insert_data_query(con, documents_query, DOC_TABLE, document_rows)
        insert_data_query(con, entities_query, ENT_TABLE, ent_rows)

    print("Data insertion process completed.")
"""Opt-in isolated baseline for the supplied ARGUS synthetic 50-document ZIP.

Run in a dedicated process; settings overrides are not thread-safe. No production
algorithms are replaced. Article prose (including embedded instructions) remains
untrusted evidence. External answer keys are used only in the output report.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import zipfile

from osint_agent.config import settings
from osint_agent.indexing.corpus import reconcile_index
from osint_agent.indexing.state import IndexNotUsableError, assert_index_usable
from osint_agent.models.document import Document
from osint_agent.retrieval.semantic import semantic_search
from osint_agent.retrieval.sufficiency import check_retrieval_sufficiency
from osint_agent.storage.chroma import get_document_collection
from osint_agent.storage.insert_data import upsert_document
from osint_agent.storage.sqlite import create_db, get_documents
from osint_agent.workflow import generate_brief_for_query


PROVIDER = "argus_adversarial_v1"
SOURCE_TYPE = "synthetic_adversarial"
COLLECTION = "argus_adversarial_v1_chunks"
# Exact defaults from scripts/reasoning_smoke.py; not calibrated thresholds.
N_RESULTS, MAX_DISTANCE, MIN_EVIDENCE = 5, 0.5, 2
SMOKE_IDS = ("Q01", "Q02", "Q12")


def parse_jsonl(text: str) -> list[dict]:
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("record must be an object")
        except ValueError as exc:
            raise ValueError(f"invalid JSONL record at line {number}: {exc}") from exc
        rows.append(row)
    return rows


def validate_documents(rows: list[dict]) -> list[Document]:
    if len(rows) != 50:
        raise ValueError(f"expected exactly 50 records, found {len(rows)}")
    documents = []
    retrieved_at = datetime.now(timezone.utc)
    for row in rows:
        for field in (
            "corpus_id", "title", "source", "provider", "source_type",
            "published_date", "url", "text",
        ):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"missing/non-string required field: {field}")
        if row["provider"] != PROVIDER or row["source_type"] != SOURCE_TYPE:
            raise ValueError("corpus record lacks required synthetic provenance")
        documents.append(Document(
            doc_id=row["corpus_id"], title=row["title"], source=row["source"],
            provider=row["provider"], source_type=row["source_type"],
            published_date=row["published_date"], url=row["url"],
            raw_text=row["text"], text=row["text"], retrieved_at=retrieved_at,
        ))
    if {doc.doc_id for doc in documents} != {
        f"ADV-{number:03d}" for number in range(1, 51)
    }:
        raise ValueError("expected unique supplied identities ADV-001 through ADV-050")
    return documents


def load_package(archive: Path):
    # Read members without extraction; support either supplied archive layout.
    with zipfile.ZipFile(archive) as package:
        def read(name):
            matches = [item for item in package.namelist()
                       if Path(item).name == name]
            if len(matches) != 1:
                raise ValueError(f"expected one archive member named {name}")
            return package.read(matches[0]).decode("utf-8")

        manifest = json.loads(read("manifest.json"))
        if manifest.get("synthetic") is not True or manifest.get("article_count") != 50:
            raise ValueError("manifest must declare 50 synthetic articles")
        rows = parse_jsonl(read("argus_adversarial_corpus_50.jsonl"))
        documents = validate_documents(rows)
        queries = parse_jsonl(read("evaluation_queries.jsonl"))
        if not queries or any(
            not isinstance(q.get(key), str) or not q[key].strip()
            for q in queries for key in ("query_id", "query", "expected_behavior")
        ):
            raise ValueError("invalid evaluation query")
        ids = [q["query_id"] for q in queries]
        if len(set(ids)) != len(ids) or not set(SMOKE_IDS) <= set(ids):
            raise ValueError("duplicate or missing representative query IDs")
    external = {row["corpus_id"]: {
        key: row.get(key) for key in
        ("event_group", "adversarial_pattern", "lineage_group")
    } for row in rows}
    return documents, queries, external


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


@contextmanager
def isolated_settings(root: Path, archive_digest: str):
    """Protect resolved paths before any write; reject unowned existing state."""
    saved = settings.DB_PATH, settings.CHROMA_PATH, settings.CHROMA_COLLECTION
    root = root.resolve()
    db_path, chroma_path = (root / "osint_sys.db").resolve(), (root / "chroma").resolve()
    live = [Path(saved[0]).resolve(), Path(saved[1]).resolve()]
    if any(_overlap(candidate, path)
           for candidate in (root, db_path, chroma_path) for path in live):
        raise ValueError("evaluation paths overlap live persistence")
    if db_path.parent != root or chroma_path.parent != root:
        raise ValueError("evaluation persistence must remain inside its root")
    # A SQLite hard link can alias the live file even when resolved paths differ.
    if db_path.exists() and live[0].exists() and db_path.samefile(live[0]):
        raise ValueError("evaluation SQLite aliases live persistence")
    marker = root / "synthetic-corpus.json"
    identity = {"provider": PROVIDER, "archive_sha256": archive_digest}
    if root.exists() and any(root.iterdir()):
        if marker.is_symlink() or not marker.is_file() or json.loads(
            marker.read_text()
        ) != identity:
            raise ValueError("evaluation directory is not owned by this corpus")
    # Reports must not be symlinks to arbitrary user files either.
    if (root / "baseline.json").is_symlink():
        raise ValueError("evaluation report cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text(json.dumps(identity, indent=2) + "\n")
    settings.DB_PATH, settings.CHROMA_PATH, settings.CHROMA_COLLECTION = (
        db_path, chroma_path, COLLECTION,
    )
    try:
        yield root
    finally:
        settings.DB_PATH, settings.CHROMA_PATH, settings.CHROMA_COLLECTION = saved


def ingest(documents: list[Document]) -> dict:
    """Caller establishes isolation; all domain writes delegate to production."""
    intended = {doc.doc_id for doc in documents}
    if Path(settings.DB_PATH).exists():
        existing = get_documents()
        if any(doc.doc_id not in intended or doc.provider != PROVIDER
               or doc.source_type != SOURCE_TYPE for doc in existing):
            raise ValueError("evaluation database contains foreign documents")
    create_db()
    with sqlite3.connect(settings.DB_PATH) as connection:
        for document in documents:
            upsert_document(connection, document)
    result = reconcile_index()
    inspection = assert_index_usable()
    persisted = get_documents()
    if {doc.doc_id for doc in persisted} != intended:
        raise ValueError("persisted identities differ from intended corpus")
    ids = sorted(get_document_collection().get()["ids"])
    return {"documents": len(persisted), "chunks": len(ids), "chunk_ids": ids,
            "reconciliation": asdict(result), "inspection": asdict(inspection)}


def chunk_baseline() -> dict:
    assert_index_usable()
    actual = get_document_collection().get(include=["documents", "metadatas"])
    per_document = []
    for doc in get_documents():
        chunks = sorted([
            {"chunk_id": key, "characters": len(text),
             "tokens": metadata["token_count"]}
            for key, text, metadata in zip(
                actual["ids"], actual["documents"], actual["metadatas"]
            ) if metadata["doc_id"] == doc.doc_id
        ], key=lambda item: item["chunk_id"])
        per_document.append({"doc_id": doc.doc_id, "title": doc.title,
                             "characters": len(doc.text),
                             "chunk_count": len(chunks), "chunks": chunks})
    counts = [doc["chunk_count"] for doc in per_document]
    return {"documents": len(counts), "chunks": sum(counts),
            "min": min(counts), "max": max(counts),
            "mean": statistics.mean(counts), "median": statistics.median(counts),
            "one_chunk_documents": counts.count(1),
            "multi_chunk_documents": sum(count > 1 for count in counts),
            "per_document": per_document}


def retrieval_baseline(queries, external) -> list[dict]:
    assert_index_usable()
    results = []
    for query in queries:
        evidence = semantic_search(query["query"], n_results=N_RESULTS)
        assessment = check_retrieval_sufficiency(
            evidence, MAX_DISTANCE, MIN_EVIDENCE,
        )
        results.append({**query, "retrieved": [
            {"rank": rank, **chunk.model_dump(mode="json"),
             "passes_threshold": chunk.distance <= MAX_DISTANCE,
             "external_metadata": external[chunk.doc_id]}
            for rank, chunk in enumerate(evidence, 1)
        ], "assessment": assessment.model_dump(mode="json")})
        print(f"retrieval {query['query_id']}: {assessment.reason}", flush=True)
    return results


def reasoning_baseline(queries) -> list[dict]:
    assert_index_usable()
    results = []
    for query in queries:
        if query["query_id"] not in SMOKE_IDS:
            continue
        print(f"reasoning {query['query_id']}: starting", flush=True)
        try:
            result = generate_brief_for_query(
                query["query"], max_distance=MAX_DISTANCE,
                min_evidence=MIN_EVIDENCE, n_results=N_RESULTS,
            )
            outcome = {"result": result.model_dump(mode="json")}
        except IndexNotUsableError:
            raise  # Never continue after a certification failure.
        except Exception as exc:
            outcome = {"error_type": type(exc).__name__, "error": str(exc)}
            if hasattr(exc, "report"):
                outcome["report"] = exc.report.model_dump(mode="json")
        results.append({"query_id": query["query_id"], "query": query["query"],
                        **outcome})
        print(f"reasoning {query['query_id']}: {outcome}", flush=True)
    return results


def run(archive: Path, root: Path) -> dict:
    documents, queries, external = load_package(archive)  # Validate before writes.
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with isolated_settings(root, digest) as output:
        report = {"archive_sha256": digest, "validated_records": len(documents),
                  "environment": {"python": platform.python_version(),
                                  "architecture": platform.machine()},
                  "paths": {"sqlite": str(settings.DB_PATH),
                            "chroma": str(settings.CHROMA_PATH),
                            "collection": settings.CHROMA_COLLECTION},
                  "retrieval_defaults": {"n_results": N_RESULTS,
                                         "max_distance": MAX_DISTANCE,
                                         "min_evidence": MIN_EVIDENCE,
                                         "source": "scripts/reasoning_smoke.py"},
                  "reasoning_model": settings.REASONING_MODEL}
        stage = "ingestion_first"
        def save():
            (output / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
        try:
            print("ingestion: first pass", flush=True)
            report["first"] = ingest(documents)
            stage = "ingestion_second"
            print("ingestion: second pass", flush=True)
            report["second"] = ingest(documents)
            report["idempotent"] = all(
                report["first"][key] == report["second"][key]
                for key in ("documents", "chunks", "chunk_ids")
            )
            if not report["idempotent"]:
                raise ValueError("identical ingestion changed document/chunk identities")
            # Read the production commit manifest; do not resolve our own config.
            with sqlite3.connect(settings.DB_PATH) as connection:
                manifest = connection.execute(
                    "SELECT compatibility_manifest FROM semantic_index_state "
                    "WHERE state_key = 'semantic_index'"
                ).fetchone()[0]
            report["compatibility_manifest"] = json.loads(manifest)
            stage = "chunking"
            report["chunking"] = chunk_baseline()
            save()
            stage = "retrieval"
            report["retrieval"] = retrieval_baseline(queries, external)
            save()
            stage = "reasoning"
            report["reasoning"] = reasoning_baseline(queries)
            report["final_inspection"] = asdict(assert_index_usable())
        except Exception as exc:
            report["failure"] = {"stage": stage, "type": type(exc).__name__,
                                 "message": str(exc),
                                 "downstream_phases": "not attempted"}
            save()
            raise
        save()
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--root", type=Path,
                        default=settings.PROJECT_ROOT / "data/eval/adversarial")
    args = parser.parse_args()
    run(args.archive, args.root)
    print(f"Baseline saved to {args.root.resolve() / 'baseline.json'}")


if __name__ == "__main__":
    main()

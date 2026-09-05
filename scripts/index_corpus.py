"""Index the ARGUS SQLite corpus into the semantic Chroma index."""

import logging
import sys
import time

from osint_agent.indexing.corpus import index_corpus


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    start = time.perf_counter()

    configure_logging()

    logger.info("Starting ARGUS corpus indexing.")

    result = index_corpus()

    logger.info(
        "Corpus indexing complete: "
        "documents_found=%d "
        "documents_indexed=%d "
        "chunks_indexed=%d "
        "chroma_before=%d "
        "chroma_after=%d "
        "changed=%s "
        "state=%s "
        "compatibility_fingerprint=%s",
        result.documents_found,
        result.documents_indexed,
        result.chunks_indexed,
        result.chroma_count_before,
        result.chroma_count_after,
        result.changed,
        result.corpus_status,
        result.compatibility_fingerprint,
    )

    end = time.perf_counter()
    execution_time = end - start
    print(f"Indexing Elapsed Time: {execution_time:.6f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.critical(
            "ARGUS corpus indexing failed.",
            exc_info=True,
        )
        sys.exit(1)

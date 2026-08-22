from dataclasses import dataclass

from osint_agent.indexing.indexing_documents import index_documents
from osint_agent.storage.chroma import create_document_collection
from osint_agent.storage.sqlite import get_documents


@dataclass
class CorpusIndexingResult:
    documents_found: int
    documents_indexed: int
    chunks_indexed: int
    chroma_count_before: int
    chroma_count_after: int


def index_corpus() -> CorpusIndexingResult:
    """Index all authoritative SQLite documents into Chroma."""

    documents = get_documents()

    collection = create_document_collection()
    chroma_count_before = collection.count()

    results = index_documents(documents)

    chunks_indexed = sum(
        result.chunks_indexed
        for result in results
    )

    chroma_count_after = collection.count()

    return CorpusIndexingResult(
        documents_found=len(documents),
        documents_indexed=len(results),
        chunks_indexed=chunks_indexed,
        chroma_count_before=chroma_count_before,
        chroma_count_after=chroma_count_after,
    )
from dataclasses import dataclass
import hashlib

from osint_agent.models.document import Document
from osint_agent.preprocessing.chunking import chunk_document
from osint_agent.indexing.embedding import embed_documents
from osint_agent.storage.chroma import (
    delete_document_chunks,
    upsert_chunks,
)
from osint_agent.storage.sqlite import mark_semantic_index_stale


@dataclass
class IndexingResult:
    doc_id: str
    chunks_created: int
    chunks_indexed: int
    chunk_ids: list[str]


def chunk_metadata(
    document: Document,
    chunk,
    *,
    corpus_digest: str = "",
    compatibility_fingerprint: str = "",
) -> dict:
    """Build the provenance metadata persisted beside one derived chunk."""

    return {
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "chunk_content_digest": hashlib.sha256(
            chunk.text.encode("utf-8")
        ).hexdigest(),
        "title": document.title or "",
        "source": document.source or "",
        "provider": document.provider,
        "source_type": document.source_type,
        "published_date": (
            document.published_date.isoformat()
            if document.published_date
            else ""
        ),
        "event_time": (
            document.event_time.isoformat() if document.event_time else ""
        ),
        "retrieved_at": document.retrieved_at.isoformat(),
        "url": document.url or "",
        "contradiction_group": document.contradiction_group or "",
        "contradiction_position": document.contradiction_position or "",
        "corpus_digest": corpus_digest,
        "compatibility_fingerprint": compatibility_fingerprint,
    }


def index_document(
    document: Document,
    *,
    corpus_digest: str = "",
    compatibility_fingerprint: str = "",
    manage_state: bool = True,
) -> IndexingResult:
    """Chunk, embed, and replace a Document in the semantic index."""

    chunks = chunk_document(document)

    embeddings = embed_documents(
        [chunk.text for chunk in chunks]
    )

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk/embedding count mismatch: "
            f"{len(chunks)} chunks, "
            f"{len(embeddings)} embeddings"
        )

    ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    documents = [
        chunk.text
        for chunk in chunks
    ]

    metadatas = [
        chunk_metadata(
            document,
            chunk,
            corpus_digest=corpus_digest,
            compatibility_fingerprint=compatibility_fingerprint,
        )
        for chunk in chunks
    ]

    if manage_state:
        mark_semantic_index_stale(
            "direct document indexing requires corpus reconciliation"
        )

    delete_document_chunks(document.doc_id)

    upsert_chunks(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return IndexingResult(
        doc_id=document.doc_id,
        chunks_created=len(chunks),
        chunks_indexed=len(chunks),
        chunk_ids=ids,
    )


def index_documents(
    documents: list[Document],
    *,
    corpus_digest: str = "",
    compatibility_fingerprint: str = "",
    manage_state: bool = True,
) -> list[IndexingResult]:
    """Index multiple Documents into the semantic index."""

    return [
        index_document(
            document,
            corpus_digest=corpus_digest,
            compatibility_fingerprint=compatibility_fingerprint,
            manage_state=manage_state,
        )
        for document in documents
    ]

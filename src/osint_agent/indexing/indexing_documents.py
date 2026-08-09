from dataclasses import dataclass

from osint_agent.models.document import Document
from osint_agent.preprocessing.chunking import chunk_document
from osint_agent.indexing.embedding import embed_documents
from osint_agent.retrieval.chroma import upsert_chunks


@dataclass
class IndexingResult:
    doc_id: str
    chunks_created: int
    chunks_indexed: int
    chunk_ids: list[str]


def index_document(document: Document) -> IndexingResult:
    """Chunk, embed, and persist a Document in the semantic index."""

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
        {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "title": document.title or "",
            "source": document.source or "",
            "provider": document.provider,
            "source_type": document.source_type,
        }
        for chunk in chunks
    ]

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
) -> list[IndexingResult]:
    """Index multiple Documents into the semantic index."""

    return [
        index_document(document)
        for document in documents
    ]
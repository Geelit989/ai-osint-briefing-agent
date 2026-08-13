import chromadb

from osint_agent.config import settings


def get_chroma_client() -> chromadb.PersistentClient:
    """Return the persistent Chroma client."""

    return chromadb.PersistentClient(
        path=str(settings.CHROMA_PATH)
    )


def create_document_collection():
    """Create the ARGUS document chunk collection if needed."""

    client = get_chroma_client()

    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION
    )


def get_document_collection():
    """Return the existing ARGUS document chunk collection."""

    client = get_chroma_client()

    return client.get_collection(
        name=settings.CHROMA_COLLECTION
    )


def upsert_chunks(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    """Persist document chunks and their embeddings in Chroma."""

    collection = get_document_collection()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_chunks(
    query_embedding: list[float],
    n_results: int = 5,
) -> dict:
    """Retrieve semantically similar document chunks."""

    collection = get_document_collection()

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )


def delete_document_chunks(doc_id: str) -> None:
    """Delete all indexed chunks belonging to a document."""

    collection = get_document_collection()

    collection.delete(
        where={"doc_id": doc_id}
    )


def delete_document_chunks(doc_id: str) -> None:
    """Delete all indexed chunks belonging to a document."""

    collection = get_document_collection()

    collection.delete(
        where={"doc_id": doc_id}
    )
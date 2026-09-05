import ollama

from osint_agent.config import settings


class EmbeddingDimensionError(ValueError):
    """An embedding did not match the configured retrieval space."""


def _validate_dimensions(embeddings: list[list[float]]) -> None:
    invalid = [
        index
        for index, embedding in enumerate(embeddings)
        if len(embedding) != settings.EMBEDDING_DIMENSION
    ]
    if invalid:
        raise EmbeddingDimensionError(
            "Embedding dimension mismatch: expected "
            f"{settings.EMBEDDING_DIMENSION}, invalid rows={invalid}"
        )


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document or chunk text for semantic indexing."""

    prefixed_texts = [
        f"{settings.DOCUMENT_EMBEDDING_PREFIX}{text}"
        for text in texts
    ]

    response = ollama.embed(
        model=settings.EMBEDDING_MODEL,
        input=prefixed_texts,
    )

    embeddings = response["embeddings"]
    _validate_dimensions(embeddings)
    return embeddings


def embed_query(query: str) -> list[float]:
    """Embed a user query for semantic retrieval."""

    response = ollama.embed(
        model=settings.EMBEDDING_MODEL,
        input=f"{settings.QUERY_EMBEDDING_PREFIX}{query}",
    )
    embeddings = response["embeddings"]
    _validate_dimensions(embeddings)
    return embeddings[0]

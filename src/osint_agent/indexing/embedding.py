import ollama

from osint_agent.config import settings


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document or chunk text for semantic indexing."""

    prefixed_texts = [
        f"search_document: {text}"
        for text in texts
    ]

    response = ollama.embed(
        model=settings.EMBEDDING_MODEL,
        input=prefixed_texts,
    )

    return response["embeddings"]


def embed_query(query: str) -> list[float]:
    """Embed a user query for semantic retrieval."""

    response = ollama.embed(
        model=settings.EMBEDDING_MODEL,
        input=f"search_query: {query}",
    )

    return response["embeddings"][0]
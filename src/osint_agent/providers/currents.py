"""Currents API ingestion utilities."""

from __future__ import annotations

from typing import Any

import requests

from osint_agent.config import settings
from osint_agent.models.document import Document
from osint_agent.preprocessing.clean_text import clean_text
from osint_agent.config import settings 


def search_currents(
    query: str,
    limit: int = 10,
) -> list[Document]:
    """Search Currents and return normalized ARGUS documents.

    Args:
        query: Search terms submitted to the Currents API.
        limit: Maximum number of returned articles to normalize.

    Returns:
        A list of validated Document objects.

    Raises:
        ValueError: If the query is empty or limit is invalid.
        RuntimeError: If the API key is missing or Currents returns an error.
        requests.RequestException: If the HTTP request fails.
    """
    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("Currents search query cannot be empty.")

    if limit < 1:
        raise ValueError("Search result limit must be at least 1.")

    if not settings.CURRENTS_API_KEY:
        raise RuntimeError("CURRENTS_API_KEY is not configured.")

    response = requests.get(
        settings.CURRENTS_SEARCH_URL,
        params={
            "keywords": normalized_query,
            "language": "en",
            "apiKey": settings.CURRENTS_API_KEY,
        },
        timeout=settings.REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == "error":
        raise RuntimeError(
            f"Currents API error: {payload.get('message', 'Unknown error')}"
        )

    articles = payload.get("news")

    if not isinstance(articles, list):
        raise RuntimeError(
            "Currents response did not contain a valid 'news' list."
        )

    documents: list[Document] = []

    for article in articles[:limit]:
        if not isinstance(article, dict):
            continue

        try:
            documents.append(currents_to_document(article))
        except ValueError:
            # Skip unusable records while allowing valid articles through.
            continue

    return documents


def currents_to_document(article: dict[str, Any]) -> Document:
    """Normalize one Currents article into an ARGUS Document."""

    provider_id = article.get("id")
    title = article.get("title")
    description = article.get("description")

    if not provider_id:
        raise ValueError("Currents article is missing an ID.")

    raw_text_parts = [
        value.strip()
        for value in (title, description)
        if isinstance(value, str) and value.strip()
    ]

    raw_text = "\n\n".join(raw_text_parts)

    if not raw_text:
        raise ValueError(
            "Currents article has no usable title or description."
        )

    return Document(
        doc_id=f"currents-{provider_id}",
        title=title,
        source=article.get("author"),
        provider="currents",
        source_type="live_news",
        published_date=article.get("published"),
        url=article.get("url"),
        raw_text=raw_text,
        text=clean_text(raw_text),
        meta_data={
            "language": article.get("language"),
            "category": article.get("category"),
            "image": article.get("image"),
            "provider_id": provider_id,
        },
    )
"""Inspect one article returned by the Currents API."""

from __future__ import annotations

import json

import requests

from osint_agent.config import settings


def main() -> None:
    """Request one Currents article and print its raw structure."""
    if not settings.CURRENTS_API_KEY:
        raise RuntimeError("CURRENTS_API_KEY is not configured.")

    response = requests.get(
        settings.CURRENTS_SEARCH_URL,
        params={
            "keywords": "Ukraine",
            "language": "en",
            "apiKey": settings.CURRENTS_API_KEY,
        },
        timeout=settings.REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()
    payload = response.json()

    articles = payload.get("news", [])

    if not articles:
        print("No articles returned.")
        return

    first_article = articles[0]

    print(json.dumps(first_article, indent=2))


if __name__ == "__main__":
    main()
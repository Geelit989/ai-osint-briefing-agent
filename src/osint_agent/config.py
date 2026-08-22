"""Application configuration for Project ARGUS."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

load_dotenv() # TODO Use .env file to set environment variables / whats more efficient: config or .env?


class Settings:
    """Centralized application configuration."""

    # -----------------------------------------------------------------------
    # Project Paths
    # -----------------------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DB_PATH = PROJECT_ROOT / "osint_sys.db"

    CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
    CHROMA_COLLECTION = "argus_document_chunks"

    # -----------------------------------------------------------------------
    # API Keys
    # -----------------------------------------------------------------------

    CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

    # -----------------------------------------------------------------------
    # External Providers
    # -----------------------------------------------------------------------

    CURRENTS_SEARCH_URL = (
        "https://api.currentsapi.services/v1/search"
    )

    STATE_DEPARTMENT_RSS = (
        "https://www.state.gov/rss-feed/"
        "department-press-briefings/feed/"
    )

    # -----------------------------------------------------------------------
    # Runtime
    # -----------------------------------------------------------------------

    REQUEST_TIMEOUT_SECONDS = 10

    MAX_RESULTS = 25

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "nomic-embed-text",
    )

    TOKENIZER_NAME = os.getenv(
        "TOKENIZER_NAME",
        "nomic-ai/nomic-embed-text-v1.5",
    )


    REASONING_MODEL = os.getenv(
        "REASONING_MODEL",
        "llama3.2",
    )

    OLLAMA_HOST = os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434",
    )

    REASONING_TIMEOUT_SECONDS = int(
    os.getenv("REASONING_TIMEOUT_SECONDS", "120")
)

settings = Settings()
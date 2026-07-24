# AI OSINT Briefing Agent

AI-powered OSINT briefing prototype that ingests, normalizes, and stores open-source intelligence documents. The current MVP supports both local sample ingestion and the initial foundation for live news ingestion via the Currents API.

---

## Project Status

This repository is an MVP-stage local workflow. The current pipeline supports:

- Loading sample article data from JSON
- Cleaning and validating article text
- Extracting named entities with spaCy
- Saving processed JSON artifacts
- Creating SQLite tables for documents and entities
- Inserting processed documents and entities into SQLite
- Running an end-to-end smoke test from one command
- Centralized application configuration through `config.py`
- Environment variable management through `.env`
- Initial Currents API integration
- Normalization of live news articles into the common `Document` model

---

## Repository Layout

```text
.
├── .env.example
├── data/
│   ├── samples/
│   │   └── sample_articles.json
│   └── processed/
│       ├── cleaned_articles.json
│       └── extracted_entities.json
├── scripts/
│   ├── inspect_currents.py
│   └── main.py
├── src/
│   └── osint_agent/
│       ├── config.py
│       ├── ingestion/
│       │   ├── currents.py
│       │   └── load_samples.py
│       ├── processing/
│       │   └── clean_text.py
│       ├── extraction/
│       │   └── ner.py
│       ├── models/
│       │   └── document.py
│       └── retrieval/
│           ├── storage.py
│           └── insert_data.py
└── tests/
    ├── test_clean_text.py
    └── test_currents.py
```

---

## Current Architecture

```text
Sample JSON ─────────────┐
                         │
Currents API (WIP) ──────┤
                         ▼
                  Document Model
                         ▼
                   Text Cleaning
                         ▼
              Named Entity Recognition
                         ▼
                      SQLite
```

All ingestion sources are normalized into the same `Document` model before entering the downstream processing pipeline.

---

## Requirements

- Python 3.11+
- spaCy
- Pydantic
- requests
- python-dotenv
- pytest
- `en_core_web_sm` spaCy model

---

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .

python -m spacy download en_core_web_sm
```

Create a local environment configuration:

```bash
cp .env.example .env
```

Populate `.env` with the required API keys.

---

## Run the End-to-End Smoke Test

```bash
python scripts/main.py
```

The smoke workflow currently performs:

1. Creates SQLite database tables
2. Loads sample article data
3. Cleans and validates document text
4. Writes cleaned JSON artifacts
5. Runs spaCy Named Entity Recognition
6. Writes extracted entities
7. Inserts documents and entities into SQLite
8. Queries one stored document to verify persistence

---

## Current Testing

Run the full test suite:

```bash
pytest
```

Current unit tests cover:

- Text cleaning and normalization
- Currents API document normalization

---

## Development Timeline

### May 2026

- Initialized repository and project structure
- Added sample article ingestion
- Implemented text cleaning pipeline
- Added Unicode normalization
- Introduced Pydantic document validation
- Added spaCy Named Entity Recognition
- Added SQLite document and entity storage
- Created end-to-end smoke workflow

### July 2026

- Added centralized application configuration (`config.py`)
- Introduced `.env` configuration management
- Installed project using editable (`pip install -e .`) package layout
- Integrated the Currents API
- Implemented Currents-to-Document normalization
- Added unit tests for Currents document normalization

---

## Next Steps

- Complete `search_currents()` live retrieval
- Integrate U.S. State Department RSS feed
- Persist live documents into SQLite
- Introduce Chroma vector storage
- Generate embeddings for ingested documents
- Implement semantic retrieval
- Build planner and executor nodes
- Implement sufficient-context decision logic
- Add structured intelligence brief generation

---

## Long-Term Architecture

Planned evolution:

- SQLite for MVP relational storage
- Chroma for semantic vector retrieval
- PostgreSQL for production relational storage
- Multi-source ingestion (news APIs, RSS, government feeds)
- Agentic workflow orchestration using LangGraph
- Planner / Executor architecture
- Historical comparison ("What Changed?")
- Timeline generation
- Intelligence-style BLUF briefing generation
- Lightweight TypeScript dashboard
- Enterprise governance, traceability, and confidence scoring
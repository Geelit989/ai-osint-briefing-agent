# AI OSINT Briefing Agent

AI-powered OSINT briefing prototype that ingests open-source article text, cleans and validates it, extracts named entities, stores the results in SQLite, and produces a small analyst-friendly smoke-test output.

## Project Status

This repository is an MVP-stage local workflow. The current pipeline supports:

- Loading sample article data from JSON
- Cleaning and validating article text
- Extracting named entities with spaCy
- Saving processed JSON artifacts
- Creating SQLite tables for documents and entities
- Inserting processed documents and entities into SQLite
- Running an end-to-end smoke test from one command

## Repository Layout

```text
.
|-- data/
|   |-- samples/sample_articles.json
|   `-- processed/
|       |-- cleaned_articles.json
|       `-- extracted_entities.json
|-- scripts/main.py
|-- src/osint_agent/
|   |-- ingestion/load_samples.py
|   |-- processing/
|   |   |-- clean_text.py
|   |   `-- document.py
|   |-- extraction/ner.py
|   `-- retrieval/
|       |-- storage.py
|       `-- insert_data.py
`-- tests/test_clean_text.py
```

## Requirements

- Python 3.11+
- spaCy
- Pydantic
- pytest for tests
- `en_core_web_sm` spaCy model

The `requirements.txt` file is currently empty, so install dependencies manually until dependency pinning is added:

```bash
python -m pip install spacy pydantic pytest
python -m spacy download en_core_web_sm
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install spacy pydantic pytest
python -m spacy download en_core_web_sm
```

Because this project uses a `src/` layout, either run scripts that already add `src/` to `sys.path`, such as `scripts/main.py`, or set `PYTHONPATH=src` when running modules directly.

## Run The End-To-End Workflow

```bash
python scripts/main.py
```

The smoke workflow does the following:

1. Creates the SQLite database tables in `osint_sys.db`
2. Loads articles from `data/samples/sample_articles.json`
3. Cleans and validates article text
4. Writes cleaned records to `data/processed/cleaned_articles.json`
5. Runs spaCy NER over the cleaned text
6. Writes extracted entities to `data/processed/extracted_entities.json`
7. Inserts documents and entities into SQLite
8. Prints one stored document row

## Run Individual Modules

Load and clean sample articles:

```bash
PYTHONPATH=src python src/osint_agent/ingestion/load_samples.py
```

Extract named entities:

```bash
PYTHONPATH=src python src/osint_agent/extraction/ner.py
```

Create SQLite tables:

```bash
PYTHONPATH=src python src/osint_agent/retrieval/storage.py
```

Insert processed JSON data into SQLite:

```bash
PYTHONPATH=src python src/osint_agent/retrieval/insert_data.py
```

## Data Flow

```text
Sample Articles JSON
    -> Document loading
    -> Text cleaning
    -> Pydantic validation
    -> Processed article JSON
    -> spaCy named entity recognition
    -> Extracted entity JSON
    -> SQLite document/entity tables
```

## Database Schema

The SQLite database is created at `osint_sys.db`.

### `documents`

- `doc_id`
- `title`
- `source`
- `published_date`
- `url`
- `raw_text`
- `cleaned_text`
- `meta_data`

### `entities`

- `entity_id`
- `ent_text`
- `start_char`
- `end_char`
- `label`
- `doc_id`

`entities.doc_id` references `documents.doc_id`.

## Testing

Run the current test suite with:

```bash
PYTHONPATH=src pytest
```

Current tests cover the text-cleaning utility, including empty input, whitespace normalization, smart quote and dash replacement, and Unicode normalization.

## Development Timeline

### 4 May 2026

- Initialized the GitHub repository and local project structure
- Added sample article data
- Created the first ingestion script

### 5 May 2026

- Implemented `clean_text()`
- Added Unicode normalization and whitespace cleanup
- Added pytest unit tests
- Established the `src/` import workflow

### 9 May 2026

- Added spaCy-based named entity extraction
- Saved extracted entities for inspection and downstream use

### 30 May 2026

- Added SQLite table creation
- Added document and entity insertion helpers
- Added `scripts/main.py` end-to-end smoke workflow

## Next Steps

- Add pinned dependencies to `requirements.txt` or `pyproject.toml`
- Add tests for document validation, NER output shape, and SQLite insertion
- Improve storage error handling and logging consistency
- Add event extraction
- Add timeline generation
- Add change detection for "what changed" briefings
- Add structured briefing generation
- Add a CLI or lightweight dashboard interface

## Long-Term Architecture

Planned evolution:

- SQLite for the MVP
- PostgreSQL for production relational storage
- Vector search for semantic retrieval
- Graph relationships between actors, entities, events, and sources
- Agent orchestration for multi-step intelligence workflows

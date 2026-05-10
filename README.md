# AI OSINT Briefing Agent

AI-powered OSINT (Open-Source Intelligence) briefing agent that ingests unstructured open-source text, extracts structured intelligence signals, and generates analyst-style briefings.

---

## Setup

### Requirements

- Python 3.11+

### Environment Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Local Modules

Because the project uses a `src/` layout:

```bash
PYTHONPATH=src python src/osint_agent/ingestion/load_samples.py
```

---

## Current MVP Scope

Current prototype pipeline:

```text
Sample Articles JSON
    ↓
Load into Document objects
    ↓
Text cleaning / validation
    ↓
Named Entity Recognition (spaCy)
    ↓
Structured entity extraction
    ↓
SQLite persistence (in progress)
```

---

## Current Functionality

### Data Ingestion
- Load sample article JSON data
- Transform raw articles into structured document objects

### Text Processing
- Unicode normalization
- Whitespace cleanup
- Basic document validation

### NLP / Entity Extraction
- spaCy-based named entity recognition (NER)
- Extract entity mentions from cleaned text
- Persist extracted entities as structured dictionaries

### Testing
- pytest unit tests for text cleaning utilities

### Tooling / Dev Environment
- GitHub repository connected to local development environment
- `.gitignore` configured for Python artifacts (`__pycache__`, virtualenv files, etc.)
- VS Code + virtual environment workflow

---

## Development Timeline

### 4 May 2026
Project initialization:
- GitHub repo setup
- Local environment configuration
- Initial project scaffolding
- Sample dummy data creation
- First ingestion script

### 5 May 2026
Text preprocessing milestone:
- Implemented `clean_text()` utility
- Added Unicode normalization
- Added whitespace cleanup
- Added pytest unit tests
- Configured `.gitignore`
- Established `src/` module import workflow

### 9 May 2026
NLP milestone:
- Built named entity extraction pipeline
- Extracted entities from cleaned document text
- Serialized extracted entities for inspection/testing

### Current Work
Database persistence layer:
- SQLite schema design
- Document/entity relational model
- Ingestion pipeline for document + entity persistence

---

## Next Steps

Planned MVP milestones:

- [ ] SQLite database initialization
- [ ] Document + entity ingestion pipeline
- [ ] Event extraction layer
- [ ] Timeline generation
- [ ] Change detection ("what changed")
- [ ] Structured briefing generation
- [ ] CLI or lightweight dashboard interface

---

## Long-Term Architecture

Planned evolution:

- SQLite (MVP)
- PostgreSQL / production relational storage
- vector search / semantic retrieval
- graph relationships between actors/events
- agent orchestration for multi-step intelligence workflows
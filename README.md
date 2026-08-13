# Project ARGUS — AI OSINT Briefing Agent

Project ARGUS is an AI-powered OSINT briefing prototype designed to ingest, normalize, store, retrieve, and eventually synthesize open-source reporting into structured, intelligence-style products.

Rather than functioning as a generic chatbot or news summarizer, ARGUS is being developed as a **bounded agentic OSINT reasoning system** that combines deterministic data processing, semantic retrieval, and controlled AI reasoning while maintaining source traceability and auditability.

The current MVP has progressed from basic document ingestion and structured storage into a working **semantic retrieval pipeline** using SQLite, local embeddings, and ChromaDB.

---

## Project Status

ARGUS currently supports an end-to-end local data path approximately equivalent to:

```text
Currents API / Sample Data
        ↓
Provider / Ingestion Layer
        ↓
Document Normalization
        ↓
Text Preprocessing
        ↓
Named Entity Recognition
        ↓
SQLite Persistence
        ↓
Deterministic Chunking
        ↓
Local Embeddings
        ↓
ChromaDB Vector Index
        ↓
Semantic Retrieval
```

Current capabilities include:

* Loading local sample article data
* Retrieving live news through the Currents API
* Normalizing multiple ingestion paths into a common `Document` model
* Cleaning and validating article text
* Extracting named entities with spaCy
* Persisting documents and entities in SQLite
* Deterministically chunking documents for semantic indexing
* Generating document and query embeddings locally with Ollama
* Using `nomic-embed-text` with retrieval-specific task prefixes
* Persisting embeddings and chunk metadata in ChromaDB
* Performing semantic top-k retrieval against stored reporting
* Deterministic/idempotent Chroma upserts
* Persistent Chroma collections across runtime restarts
* Centralized application configuration through `config.py`
* Environment variable management through `.env`
* Unit testing of ingestion, preprocessing, and chunking behavior
* End-to-end smoke testing of the ingestion and persistence workflow

SQLite currently serves as ARGUS's **authoritative structured data store**, while ChromaDB serves as its **semantic retrieval index**.

---

## Repository Layout

The project is evolving toward the following package structure:

```text
.
├── data/
│   ├── processed/
│   ├── raw/
│   └── samples/
│
├── docs/
│   └── decisions/
│
├── notebooks/
│   ├── chroma_db/
│   ├── agent.ipynb
│   ├── chunking_embedding.ipynb
│   ├── osint_sys.db
│   └── test.ipynb
│
├── scripts/
│   ├── main_workflow_smoke.py
│   ├── main.py
│   ├── manage_dev_db.py
│   └── test_search_currents.py
│
├── src/
│   └── osint_agent/
│       ├── extraction/
│       ├── indexing/
│       ├── ingestion/
│       ├── models/
│       ├── preprocessing/
│       ├── providers/
│       ├── retrieval/
│       ├── __init__.py
│       └── config.py
│
├── tests/
│   ├── test_chunking.py
│   ├── test_clean_text.py
│   ├── test_currents.py
│   └── test_indexing_documents.py
│
├── .env
├── .env.example
├── .gitignore
├── osint_sys.db
├── pyproject.toml
└── README.md
```

The emerging responsibility boundaries are:

```text
ingestion / providers
        ↓
acquire and normalize information

preprocessing
        ↓
prepare information

extraction
        ↓
derive structured information

indexing
        ↓
create semantic representations

storage
        ↓
persist structured and vector records

retrieval
        ↓
find relevant evidence

agentic reasoning
        ↓
decide what additional information or actions are required

brief generation
        ↓
produce structured intelligence products
```

---

## Current Architecture

```text
Sample JSON ─────────────┐
                         │
Currents API ────────────┤
                         ▼
                  Document Model
                         ▼
                Text Preprocessing
                         ▼
              Named Entity Recognition
                         ▼
                SQLite Storage
                         │
                         ▼
              Deterministic Chunking
                         ▼
                 Local Embeddings
                         ▼
                    ChromaDB
                         ▼
                Semantic Retrieval
```

All ingestion sources are normalized into the same `Document` model before entering the downstream pipeline.

### SQLite

SQLite acts as the authoritative structured store for records such as:

* Documents
* Source metadata
* Publication dates
* URLs
* Cleaned text
* Extracted entities
* Provider metadata

Future schema extensions may include:

* Events
* Briefs
* Audit records
* Retrieval provenance

### ChromaDB

ChromaDB acts as ARGUS's semantic retrieval index.

Each indexed chunk can contain:

* Stable `chunk_id`
* Chunk text
* Embedding vector
* `doc_id`
* Title
* Provider/source metadata

This allows ARGUS to retrieve conceptually relevant reporting even when the user's query does not exactly match stored article text.

---

## Semantic Retrieval

ARGUS currently generates embeddings locally using:

```text
nomic-embed-text
```

through Ollama.

Nomic retrieval task prefixes are used separately for indexed evidence and analyst queries:

```text
search_document:
```

and:

```text
search_query:
```

The basic retrieval path is:

```text
User Query
    ↓
Query Embedding
    ↓
ChromaDB
    ↓
Top-K Semantic Matches
    ↓
Structured Retrieval Results
```

Early testing against a small Currents corpus successfully grouped semantically related reporting and demonstrated meaningful separation between highly relevant results and weaker corpus neighbors.

No fixed similarity or Chroma distance threshold has been selected yet. Retrieval currently favors a **top-k-first approach**, with relevance and sufficient-context evaluation planned as the corpus grows.

---

## Requirements

Core project dependencies currently include:

* Python 3.11+
* spaCy
* Pydantic
* requests
* python-dotenv
* pytest
* ChromaDB
* Ollama
* `nomic-embed-text`
* `en_core_web_sm` spaCy model

Additional dependencies will be introduced as the LangGraph reasoning layer and user interface are implemented.

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

ARGUS currently uses Ollama for local embedding generation. Ensure the required embedding model is available locally:

```bash
ollama pull nomic-embed-text
```

Runtime database and vector-index artifacts should remain outside source control.

---

## Run the End-to-End Smoke Test

```bash
python scripts/main_workflow_smoke.py
```

The smoke workflow validates the core ingestion and structured persistence path.

Current workflow capabilities include:

1. Initialize SQLite tables
2. Retrieve or load source documents
3. Normalize data into the common `Document` model
4. Clean and validate document text
5. Run spaCy Named Entity Recognition
6. Persist documents and entities into SQLite
7. Retrieve stored records to verify persistence

The semantic indexing workflow extends this path with:

```text
Stored Document
      ↓
Chunk Document
      ↓
Generate Embeddings
      ↓
Upsert into Chroma
      ↓
Semantic Query
```

---

## Current Testing

Run the full test suite:

```bash
pytest
```

Current automated testing covers areas including:

* Text cleaning and normalization
* Currents API document normalization
* Short-document chunking
* Long-document chunking
* Deterministic chunk IDs
* Title handling
* Prevention of unnecessary title duplication

Additional tests will be added as embedding, Chroma persistence, retrieval, and agentic orchestration move fully into application modules.

---

## Development Timeline

### May 2026

* Initialized repository and package structure
* Added sample article ingestion
* Implemented text cleaning pipeline
* Added Unicode normalization
* Introduced Pydantic document validation
* Added spaCy Named Entity Recognition
* Added SQLite document and entity storage
* Created initial end-to-end smoke workflow

### July 2026

* Added centralized application configuration through `config.py`
* Introduced `.env` configuration management
* Installed the project using editable package layout
* Integrated the Currents API
* Implemented Currents-to-`Document` normalization
* Added unit tests for Currents normalization
* Expanded SQLite document metadata
* Validated live document ingestion and persistence
* Continued restructuring the application into clearer provider, model, preprocessing, extraction, and retrieval boundaries

### August 2026

* Implemented deterministic document chunking
* Added chunking unit tests
* Validated chunking against live Currents documents
* Added local embedding generation using Ollama and `nomic-embed-text`
* Validated semantic similarity independently of the vector database
* Added persistent ChromaDB vector storage
* Validated semantic top-k retrieval
* Confirmed deterministic/idempotent Chroma upserts
* Confirmed Chroma persistence across runtime restarts
* Began migrating notebook embedding/indexing logic into application modules
* Established separate `indexing` and vector-storage responsibilities
* Centralized embedding model configuration
* Established canonical project-root-based Chroma persistence
* Synchronized the current development architecture into the Enterprise development environment

---

## Immediate Next Steps

The semantic retrieval foundation is now functional. Near-term development is focused on turning the proven retrieval components into the evidence layer for ARGUS's bounded reasoning workflow.

### Retrieval Layer

* Complete production-oriented embedding utilities
* Complete Chroma storage/query utilities
* Finalize indexing orchestration:

```text
Document
    ↓
Chunks
    ↓
Embeddings
    ↓
Chroma Upsert
```

* Finalize semantic retrieval:

```text
User Query
    ↓
Query Embedding
    ↓
Chroma Top-K
    ↓
Structured Retrieval Results
```

* Expand the indexed corpus with additional live reporting
* Evaluate retrieval relevance as corpus size increases

### Agentic Reasoning Layer

* Introduce LangGraph orchestration
* Define bounded planner behavior
* Implement executor/tool routing
* Implement a sufficient-context decision
* Allow the system to determine whether retrieved evidence is adequate before generating an answer
* Preserve deterministic boundaries around ingestion, preprocessing, indexing, and storage

### Intelligence Product Layer

* Generate structured intelligence briefs
* Add source traceability
* Add BLUF-style output
* Surface confidence and intelligence gaps
* Begin historical comparison / "What Changed?" capability

### Interface

* Build a lightweight demonstration interface
* Support analyst queries
* Display generated briefs
* Expose supporting source material and retrieval evidence

---

## Target MVP Architecture

The target MVP moves ARGUS beyond a fixed deterministic workflow without making every component autonomous.

```text
                    Analyst Query
                          │
                          ▼
                    ARGUS State
                          │
                          ▼
                       Planner
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
        Semantic Retrieval    Structured Retrieval
             Chroma                 SQLite
                │                   │
                └─────────┬─────────┘
                          ▼
                 Retrieved Evidence
                          │
                          ▼
                 Sufficient Context?
                     │          │
                    No         Yes
                     │          │
                     ▼          ▼
             Additional Tool   Brief
                Execution    Generation
                     │          │
                     └────┬─────┘
                          ▼
                Traceable Output
```

The objective is **bounded autonomy**: AI reasoning determines what information or tools are needed, while deterministic components remain responsible for data acquisition, transformation, storage, and retrieval.

---

## Long-Term Architecture

ARGUS is intended to evolve from a document-centric semantic retrieval system toward an **AI-assisted intelligence production platform**.

Planned capabilities include:

* Multi-source ingestion from news APIs, RSS, government feeds, and curated sources
* Structured event extraction
* Event classification
* Cross-source event clustering
* Entity and event relationship modeling
* Historical comparison
* "What Changed?" analysis
* Timeline generation
* Intelligence-style BLUF briefing generation
* Confidence and intelligence-gap reporting
* Source-to-claim traceability
* Analyst feedback and human-in-the-loop validation
* Lightweight TypeScript analyst dashboard
* PostgreSQL or enterprise relational storage
* Enterprise vector search
* Governance and audit logging
* Role-based access control
* Rust optimization for deterministic high-volume preprocessing

A future event-centric architecture may represent intelligence as:

```text
Source Documents
       ↓
Semantic Retrieval
       ↓
Event Extraction
       ↓
Event Classification
       ↓
Event Clustering
       ↓
Historical Event State
       ↓
Change Detection
       ↓
Timeline / Assessment
       ↓
Structured Intelligence Brief
```

This enables ARGUS to reason about **developments over time** rather than simply comparing article text or generating isolated summaries.

---

## Design Principles

ARGUS development follows several core principles:

### Bounded Agentic Reasoning

Reasoning is introduced where dynamic decision-making adds value. Deterministic tasks such as cleaning, chunking, persistence, and embedding remain explicit application components.

### Source Traceability

Generated intelligence products should remain traceable to the source evidence that informed them.

### Auditability

The system should preserve enough structured metadata to reconstruct how information entered the system and contributed to an output.

### Local-First Development

The MVP favors lightweight local infrastructure where practical:

* SQLite for relational storage
* ChromaDB for semantic indexing
* Ollama for local model execution

These components provide a practical development environment while preserving migration paths toward enterprise infrastructure.

### Human-in-the-Loop

ARGUS is intended to accelerate analyst workflows rather than replace analytical judgment. Analysts should be able to inspect sources, understand system confidence, and challenge generated assessments.

---

## Project Direction

ARGUS should not be viewed simply as:

```text
documents → embeddings → chatbot
```

The intended architecture is:

```text
open-source reporting
        ↓
structured ingestion
        ↓
semantic memory
        ↓
evidence retrieval
        ↓
bounded agentic reasoning
        ↓
historical/event context
        ↓
traceable intelligence synthesis
```

The current MVP has established the ingestion, structured-storage, chunking, embedding, and semantic-retrieval foundation required to begin implementing that reasoning layer.

The long-term goal is an **AI-assisted OSINT intelligence production system** capable of transforming fragmented open-source reporting into structured, traceable, and evidence-backed intelligence products.

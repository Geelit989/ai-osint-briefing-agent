# Adversarial corpus integration and baseline — 2026-09-06

Observation only. No production code was changed for this task. Existing dirty
production files and earlier hardening work were preserved.

## Artifacts and usage

- `scripts/seed_adversarial_corpus.py`: validates the supplied ZIP's JSONL, isolates
  persistence, performs two normal ingestion/reconciliation passes, and captures
  chunking, all supplied retrieval queries, and three normal reasoning smokes.
- `tests/test_adversarial_corpus.py`: 22 integration/infrastructure tests with real
  temporary SQLite/Chroma and production chunking; Ollama operations are replaced
  only in tests.
- This report: usage, measured results, and limitations.

The original archive was not extracted, rewritten, or added to the repository.
Its SHA-256 is
`96bb505eab9bfeca1e53871e991c98b415982682257fa36cfd0475a41fbd04a2`.

The completed live-local invocation was:

```sh
.venv/bin/python scripts/seed_adversarial_corpus.py /path/to/argus_adversarial_corpus_50.zip --root /tmp/argus-adversarial-v1
```

Full machine-readable results, including every document's actual chunk IDs and
lengths, all ranked retrieval records/distances/external labels, and exact smoke
outcomes: `/private/tmp/argus-adversarial-v1/baseline.json`.

Running the command again repeats the baseline, including embeddings and smokes.
Use a separate root for a fresh run. Without `--root`, the default is
`data/eval/adversarial` under the repository. Evaluation data is disposable; the
original ZIP is the reconstruction source. Temporary-directory retention is not
guaranteed by the OS.

## Production path reused

`Document` → `storage.insert_data.upsert_document` → authoritative
`storage.sqlite.get_documents` → `indexing.corpus.reconcile_index` →
`indexing.state.expected_chunks` / `preprocessing.chunking.chunk_document` →
`indexing.embedding.embed_documents` → `storage.chroma.recreate_document_collection`
and `upsert_chunks` → production verification and SQLite commit →
`indexing.state.assert_index_usable` → `retrieval.semantic.semantic_search`
(`embed_query` / `query_chunks`) → `workflow.generate_brief_for_query`.

`check_retrieval_sufficiency` supplies relevance/independence counts. Production
reconciliation retains its existing batching, stale-chunk replacement, corpus
and configuration rechecks, dimensionality validation, and fail-closed guards.
The utility does not construct indexed records or chunk boundaries.

## Isolation and adaptation

The dedicated CLI process temporarily overrides only the existing settings'
SQLite path, Chroma path, and collection, restoring them in `finally`:

- SQLite and its `semantic_index_state`: `/private/tmp/argus-adversarial-v1/osint_sys.db`
- Chroma: `/private/tmp/argus-adversarial-v1/chroma`
- Collection: `argus_adversarial_v1_chunks`

Resolved live-path overlap, SQLite hard-link aliases, escaping persistence paths,
unowned nonempty directories, and report symlinks are rejected before writes.
An ownership marker ties the directory to this archive's digest. The utility is
for a dedicated process, not concurrent threads sharing mutable settings.

All 50 records are validated before establishing the write context. Supplied
`corpus_id` maps directly to `doc_id`; full text maps to both `raw_text` and `text`.
Normal Document whitespace handling applies. Title, source, publication time,
URL, provider `argus_adversarial_v1`, and source type `synthetic_adversarial` are
preserved. No event times were supplied or inferred. Retrieval time is the actual
import time, shared across the two passes; a later invocation gets a new import
time and may rebuild, while record counts and chunk identities remain stable.

External event, pattern, lineage, and expected-behavior labels are joined only
in diagnostics. `meta_data` remains empty. Embedded evaluation instructions in
the supplied article prose remain untrusted evidence, not trusted controls.

## Ingestion, chunking, and certification

Python 3.14.4, arm64; real local Ollama document and query embeddings were used.
Exactly 50 records were discovered, validated, persisted, and indexed per pass.

| Measurement | First pass | Second pass |
| --- | ---: | ---: |
| SQLite documents | 50 | 50 |
| Chroma chunks/vector records | 50 | 50 |
| Chroma count before reconciliation | 0 | 50 |
| Chroma count after reconciliation | 50 | 50 |

Chunk ID sets were identical. The normal upsert invalidates certification, so
the second pass rebuilt rather than being a no-op. Both passed normal
certification; final state is `current`, `compatible`, `usable=true`, with no
reasons or compatibility differences and 50 expected chunks. Exact production
record verification found no stale/orphaned chunks.

All documents have one chunk: min/max/mean/median = 1; one-chunk documents = 50;
multi-chunk documents = 0. Bodies are 1,575–1,788 characters and 261–300 tokens.
They fit below the unchanged 600-token size; overlap remains 100. Token counts
are the production metadata counts before the existing title-prepend behavior.

| Document | Body characters | Body tokens | Chunks |
| --- | ---: | ---: | ---: |
| ADV-001, Harbor Point initial | 1,752 | 294 | 1 |
| ADV-010, Bellweather initial | 1,661 | 272 | 1 |
| ADV-041, Harbor Point derivative | 1,788 | 300 | 1 |
| ADV-050, Bellweather derivative | 1,697 | 278 | 1 |

Unchanged semantics: `nomic-embed-text`, 768 dimensions,
`nomic-ai/nomic-embed-text-v1.5` tokenizer, `search_document: ` / `search_query: `
prefixes, `l2` distance. Full model/tokenizer digests and manifest are in the JSON.
Compatibility fingerprint:
`2ab85e108ee446e85bd1c75470a233ddae5c623051fe25c9339d4f90525d58fb`.

## Retrieval, independence, and reasoning

Used the exact existing `scripts/reasoning_smoke.py` defaults: top 5,
inclusive raw-distance threshold 0.5, minimum independent evidence 2.
The 0.5 value is a smoke default, not a calibrated similarity guarantee.

All 12 queries returned five chunks. Q01–Q10 each returned all five reports in
their externally identified event family. Nevertheless, all 60 retrieved hits
failed the threshold (overall distances 0.505388–0.712185). Every query therefore
had retrieved/usable/independent counts **5/0/0**.

Representative observations:

- Q01 Harbor Point: correct family; distances 0.654205–0.669416.
- Q02 West Fenwick: correct family; distances 0.505825–0.539817, still outside gate.
- Q10 Bellweather: correct family; distances 0.505388–0.530826; sensational headline ranked first.
- Q11 independence query: all five results have the external "Independent follow-up"
  pattern. This is retrieval behavior, not proof of independent corroboration.
- Q12 coordinated-campaign premise: retrieved Aurora-6 and West Fenwick reporting,
  distances 0.553341–0.581618. No coordination inference was generated.

Natural same-document multi-chunk accounting cannot be observed on this supplied
one-chunk corpus. A separate integration test uses genuinely longer fixture
documents with the unchanged production chunker and grouping function: multiple
usable chunks of one document count as one independent unit. This does not claim
multi-chunk coverage for the supplied corpus.

A read-only identity-only observation of Q01's original retrieved chunks used
production `group_usable_evidence` with the isolated authoritative documents.
ADV-001/011/021/031/041 remained five singleton groups, including ADV-041, labeled
externally as a syndicated derivative (`lineage-00`). This observation does not
apply a different relevance threshold or feed reasoning: actual sufficiency
remained zero usable/independent units. No lineage resolution was added.

After chunking and retrieval baseline capture, normal workflow smokes were
attempted for Q01 (attribution), Q02 (preliminary/final cause), and Q12 (leading
premise). All three returned exactly `status=insufficient_evidence`,
`reason=no evidence met relevance threshold`, with counts 5/0/0. None invoked
synthesis or claim-support models. This baseline therefore cannot establish
semantic reasoning, qualification preservation, or premise-resistance quality.
Configured reasoning model remains `llama3.2`; no retries or tuning occurred.

## Verification and pre-existing test-isolation incident

Focused: `.venv/bin/python -m pytest -q tests/test_adversarial_corpus.py`
→ **22 passed, 1 warning in 3.41s**.

Initial full suite: `.venv/bin/python -m pytest -q`
→ **176 passed, 1 warning in 3.66s**, but the checksum audit exposed an existing
test-isolation defect. `test_index_document_embeds_and_upserts_chunks` mocks
embedding/upsert, not `mark_semantic_index_stale` or `delete_document_chunks`.
It called those paths against default live state. Live certification was marked
`stale` with reason `direct document indexing requires corpus reconciliation`;
the Chroma deletion path also ran for `test-doc-001`.

This is a real side effect of that test run, not evidence that live storage was
untouched. Live SQLite still has 2,012 documents; read-only checks found **zero**
adversarial documents and **zero** adversarial Chroma metadata records. No live
repair or certification override was attempted. Live semantic retrieval remains
fail-closed until separately authorized normal reconciliation succeeds.

The complete suite was then rerun with temporary default persistence, without
skips or production/test-code changes:

```sh
.venv/bin/python -c 'import tempfile; from pathlib import Path; import pytest; from osint_agent.config import settings; from osint_agent.storage.sqlite import create_db; from osint_agent.storage.chroma import create_document_collection; tmp=tempfile.TemporaryDirectory(prefix="argus-suite-isolated-"); root=Path(tmp.name); settings.DB_PATH=root/"osint_sys.db"; settings.CHROMA_PATH=root/"chroma"; settings.CHROMA_COLLECTION="argus_suite_isolated"; create_db(); create_document_collection(); raise SystemExit(pytest.main(["-q"]))'
```

Result: **176 passed in 3.43s**, no exclusions. The ordinary run's warning was
Chroma's `asyncio.iscoroutinefunction` deprecation; the isolated command imports
Chroma before pytest warning capture. Live checksums were unchanged across this
isolated rerun. `git diff --check` passed.

## Scope closure

No production tuning, analytic remediation, model/configuration changes, or
reconciliation redesign was performed. The supplied corpus remains unchanged.
All reachable baseline phases were captured; the lack of multi-chunk documents
and threshold-qualified evidence is a measured limitation, not repaired here.

NO PRODUCTION TUNING PERFORMED.

SYNTHETIC CORPUS REMAINS ISOLATED FROM LIVE OSINT DATA.

TASK 6 NOT PERFORMED.

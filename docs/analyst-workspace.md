# ARGUS local analyst workspace

The workspace adds a Next.js/TypeScript analyst interface and a FastAPI adapter to the existing Python ARGUS application. It provides New brief, Brief history, Sources, Diagnostics, and Corpus status within the page route `/`. A completed or selected run can be reopened at `/?run=<run_id>`. Test Lab and Guided Runner are outside this implementation.

Requests follow this path:

```text
Browser → Next.js local API proxy → FastAPI adapter
        → workspace.runs.execute_run
        → workflow.generate_brief_for_query
        → retrieval → evidence assessment → sufficiency
        → synthesis → existing validation → accepted brief
```

The adapter imports Python application functions directly. It does not execute CLI scripts or parse their output. Existing CLI, smoke scripts, Python callers, and domain tests remain independent of the web application. The core changes add optional per-invocation tracing; they preserve existing return values, exceptions, gate decisions, and validation order.

## Local setup and launch

Run commands from the repository root. Use the existing Python 3.14 `.venv`, existing ARGUS configuration, SQLite corpus, Chroma index, and locally configured Ollama models.

```sh
.venv/bin/python -m pip install -r requirements-ui.txt
export PATH="$PWD/.local/node/bin:$PATH"
npm --prefix web ci
./scripts/start_argus_ui.sh
```

Node.js 24 is supported. On the inspected workstation it was installed in the ignored `.local/node` directory, with the archive checked against the official SHA256 list. The PATH line selects that local installation when present; a system Node.js installation can also be used. Frontend dependency versions are recorded in `web/package-lock.json`; the adapter dependencies are pinned in `requirements-ui.txt`.

Open **http://127.0.0.1:3000**. FastAPI documentation is available at **http://127.0.0.1:8000/docs**. The helper starts one API process and one frontend process, builds the frontend if its production build is missing, detects occupied ports, and stops both services on Ctrl-C. It does not start or duplicate Ollama. Start the existing local Ollama runtime separately when needed; the workspace can load with degraded readiness.

After changing frontend code, run `npm --prefix web run build` to update an existing production build. For frontend development:

```sh
ARGUS_UI_DEV=1 ./scripts/start_argus_ui.sh
```

The helper accepts `ARGUS_PYTHON`, `ARGUS_API_PORT`, and `ARGUS_UI_PORT`. It uses Python 3.14 and defaults both servers to loopback. When setting custom ports, export their values before launching so the proxy and API origin checks agree.

For separate terminals, start FastAPI with:

```sh
PYTHONPATH="$PWD/src" .venv/bin/python -m uvicorn osint_agent.workspace.api:app --host 127.0.0.1 --port 8000
```

Then start the frontend in another terminal:

```sh
export PATH="$PWD/.local/node/bin:$PATH"
npm --prefix web run build
npm --prefix web run start -- --port 3000
```

Use `npm --prefix web run dev -- --port 3000` for development. Keep the supported API deployment to one worker; the current execution lock is per API process. The frontend proxy targets `127.0.0.1` and the configured API port. The adapter restricts browser submission origins to configured local origins, validates host headers, and rejects remote Ollama configuration.

## Analyst behavior and domain contracts

The intelligence question is required, trimmed, and limited to 4,000 characters. The evidence-limit control sets the existing retrieval `n_results` parameter, bounded to 1–20. It counts retrieved **chunks**, not documents or sources. The adapter keeps the existing reasoning smoke defaults: `max_distance=0.5` and `min_evidence=2`. There is no supported time filter, so none is presented.

The collapsed CLI equivalent uses the real smoke entry point and safely quotes the question:

```sh
python scripts/reasoning_smoke.py 'A focused intelligence question' --max-distance 0.5 --min-evidence 2 --results 5
```

Retrieved, usable, and independent-evidence counts come directly from the core assessment. Independent evidence units are deterministic document-identity/duplicate groups; they do not establish independently sourced corroboration. The browser does not recompute grouping, sufficiency, confidence, or claim support.

Accepted products retain the current `IntelligenceBrief` schema: cited title and BLUF, reported developments, analytic assessments, intelligence gaps, sources, and known contradictions. Confidence is shown where the backend supplies it on analytic assessments. The interface does not invent a global confidence score or a separate limitations field.

| Outcome | Workspace behavior |
| --- | --- |
| `success` | Render only the accepted brief, existing claim-support judgments, citations, and exact supporting spans. |
| `insufficient_evidence` | Show the core reason, counts, retrieved evidence, and grouping; no synthesis or replacement summary. |
| `validation_failure` | Show a rejected-draft notice and diagnostics; no accepted brief or claims. |
| `error` | Show an application/runtime failure and preserve available trace information. |
| `running` | Show the last recorded audit state while no completed result has been stored. |

Citation validation, structured-output validation (including contradiction references), provenance rejection, and unsupported-claim reports produce `validation_failure`. Model invocation failures and `ClaimSupportValidatorFailure` produce `error`: an unavailable or malformed validator response is not a supported/unsupported determination. Existing rejected claim text can appear only in the labeled diagnostic support report. The raw generated draft is not retained as a brief.

Each recorded pipeline check is `pass`, `fail`, or `not_run` according to the operation actually observed. The evidence-assessment stage observes the existing combined relevance/grouping assessment. A closed sufficiency gate leaves reasoning and all subsequent validation checks `not_run`. No detailed intermediate progress or private model reasoning is invented.

## Sources, persistence, and API

Citation aliases such as `S1` belong to one run. They resolve through that run's source mapping to persistent SQLite document IDs; there is no global `/api/sources/S1` route. Source aliases use the same usable-evidence mapping as synthesis. All retrieved chunks, including those outside the relevance threshold, remain separately inspectable.

The source drawer reads authoritative metadata and full stored document text from SQLite, and presents retrieved evidence and supporting spans from the run snapshot. External article availability is not required. A missing or unreadable source document is identified explicitly; available recorded citation evidence remains visible.

Initialization adds only the idempotent `analyst_runs` table and its `idx_analyst_runs_created_at` index. Records contain run ID, query, creation time, status, requested retrieval limit, and a JSON result snapshot. The snapshot retains retrieved chunks, assessment/grouping, run-local source references, accepted product, and available diagnostics so a run can be reopened. It does not duplicate the full authoritative article corpus or rewrite document schemas/text.

| Method and route | Purpose |
| --- | --- |
| `GET /api/health` | Cached local readiness and counts. |
| `POST /api/runs` | Submit `{"query":"…","n_results":5}` and return the completed audit result. |
| `GET /api/runs?limit=20&offset=0` | Bounded recent-run metadata, newest first. |
| `GET /api/runs/{run_id}` | Reopen the saved result. |
| `GET /api/runs/{run_id}/sources` | List run-local source mappings. |
| `GET /api/runs/{run_id}/sources/{citation_id}` | Source mapping, authoritative document when available, chunks, and accepted claim links. |
| `GET /api/documents?limit=20&offset=0&q=…` | Bounded metadata browser with optional literal metadata search. |
| `GET /api/documents/{document_id}?run_id=…` | Authoritative document and optional associated run chunks. |
| `GET /api/corpus/status` | Same cached readiness information as health. |

List limits are bounded to 1–100 and offsets must be nonnegative. Missing runs, documents, or run-local citation aliases return 404. Malformed input returns 422. A concurrent submission returns 409. Expected domain refusals/rejections return a saved run with their explicit outcome, rather than a generic HTTP crash.

Generation is synchronous and permits one active execution per API process. The run identity is committed before execution. Closing the browser does not imply model cancellation. If the API process is killed before completion, its last record can remain `running`; that means the last observed state, not proof that a worker still exists. There is no background queue, automatic replay, recovery inference, or silent retry. Reopen history to inspect the recorded state before submitting another question.

Health is cached for 15 seconds, and the frontend refreshes it every 60 seconds. Checks read the SQLite document count and recorded index status, the Chroma collection count, and Ollama's installed-model listing with a short timeout. They do not infer, embed, query vectors, or scan full document contents. Recorded index state is not a fresh corpus/compatibility certification: authoritative index checks still run during retrieval. Missing services or optional Chroma dependencies degrade readiness without preventing the application shell from loading.

## Verification and current environment limits

The pre-implementation baseline passed **185 tests**. The final isolated implementation suite passed **278 tests** with one third-party Starlette/httpx deprecation warning. API/application/health tests use isolated storage and mocked model boundaries, and cover accepted products, refusal, validation rejection, system errors, persistence, citation identity, document lookup, pagination, input validation, degraded readiness, and corrupt-document citation fallback.

Use an isolated default corpus when running the complete existing test suite. A pre-existing test in `tests/test_indexing_documents.py` calls code that marks default index state stale even though embedding/upsert calls are mocked. The initial ordinary baseline invocation triggered that existing stale-state behavior; the live index is not claimed to be byte-for-byte unchanged. Subsequent full-suite execution used:

```sh
.venv/bin/python - <<'PY'
import tempfile
from pathlib import Path
import pytest
from osint_agent.config import settings
from osint_agent.storage.sqlite import create_db
from osint_agent.storage.chroma import create_document_collection
with tempfile.TemporaryDirectory(prefix='argus-ui-suite-') as directory:
    root = Path(directory)
    settings.DB_PATH = root / 'osint_sys.db'
    settings.CHROMA_PATH = root / 'chroma'
    settings.CHROMA_COLLECTION = 'argus_ui_suite'
    create_db()
    create_document_collection()
    raise SystemExit(pytest.main(['-q']))
PY
```

Frontend checks are:

```sh
export PATH="$PWD/.local/node/bin:$PATH"
npm --prefix web run typecheck
npm --prefix web run lint
npm --prefix web run build
```

The live default index reported **stale** with compatible configuration. Real API run `9f35ec57-a7e6-41ec-a683-d9af8a77e3ed` correctly returned `error` with `IndexNotUsableError`. No index reconciliation or threshold tuning was performed for the UI milestone.

A separate real API smoke against the existing isolated adversarial baseline produced run `5fa12f77-5f2b-4eab-8774-d8ce125d1eb7`: `insufficient_evidence`, with **5 retrieved / 0 usable / 0 independent**. Saved-run reopening and authoritative document details were verified. The gate prevented subsequent model calls. This establishes real API-to-retrieval/refusal behavior; it does not establish accepted live synthesis. Accepted brief rendering and rejection behavior are exercised deterministically with mocked model responses.

### Validation completed on this workstation

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` before implementation | 185 passed; one existing Chroma deprecation warning. |
| Isolated full-suite command above, after implementation | 278 passed; one Starlette/httpx deprecation warning. |
| `PYTHONPATH=src .venv/bin/pytest tests/test_workspace_api.py -q` | 49 passed. |
| `npm --prefix web run typecheck` | Passed; generates Next route types and runs `tsc --noEmit`. |
| `npm --prefix web run lint` | Passed. |
| `npm --prefix web run build` | Passed, Next.js 16.3.4 production build. |
| `git diff --check` and `bash -n scripts/start_argus_ui.sh` | Passed. |
| Chrome browser smoke | 12 checks passed, zero page errors. |
| `./scripts/start_argus_ui.sh` | Both loopback services started; frontend health proxy returned live counts. Ctrl-C stopped both processes; helper restarted for handoff. |

The browser smoke used the installed Chrome with a temporary Playwright driver in ignored `.local/browser-test`; no browser testing framework was added to application dependencies. Live checks covered the empty workspace, health, document pagination, stored-document drawer, persisted run reopening, and stale-index diagnostics. Intercepted responses built from the real domain test models covered loading, duplicate submission/history exclusion, accepted structured products, exact supporting spans, all failed/refused states, lost API connectivity, modal focus/Escape, and a narrow viewport. Those intercepted results were never persisted to the live corpus or shown as live model outputs.

Exact session smoke commands (temporary harnesses and captures remain under `/private/tmp`):

```sh
curl -fsSL http://127.0.0.1:8000/api/health -o /private/tmp/argus-api-health.json
curl -fsSL http://127.0.0.1:8000/api/runs -H 'Content-Type: application/json' -d '{"query":"Iran-linked cyber activity affecting critical infrastructure","n_results":5}' -o /private/tmp/argus-api-live-run.json
.venv/bin/python /private/tmp/argus-isolated-api-smoke.py
.venv/bin/python /private/tmp/argus-browser-fixtures.py
.local/node/bin/node /private/tmp/argus-browser-smoke.cjs
```

The proxy accepted the `localhost` origin (malformed input reached FastAPI and returned 422) and rejected an external origin with 403. Actual installed tooling was Python 3.14.4, Node 24.20.0, Next 16.3.4, React 19.2.8, TypeScript 5.9.3, and FastAPI 0.141.1. Existing ARGUS model and core dependency versions were retained. The new frontend install audit reported zero vulnerabilities. Next setup was checked against the [official installation documentation](https://nextjs.org/docs/app/getting-started/installation), and FastAPI's Python compatibility against its [published package metadata](https://pypi.org/project/fastapi/0.141.1/).

### File inventory

Added backend, startup, documentation, and test files:

```text
requirements-ui.txt
scripts/start_argus_ui.sh
docs/analyst-workspace.md
src/osint_agent/models/workflow.py
src/osint_agent/workspace/__init__.py
src/osint_agent/workspace/api.py
src/osint_agent/workspace/health.py
src/osint_agent/workspace/runs.py
src/osint_agent/workspace/storage.py
tests/test_workspace_api.py
tests/test_workspace_health.py
tests/test_workspace_runs.py
```

Added frontend files:

```text
web/.gitignore
web/package.json
web/package-lock.json
web/next.config.ts
web/next-env.d.ts
web/tsconfig.json
web/eslint.config.mjs
web/app/layout.tsx
web/app/page.tsx
web/app/globals.css
web/app/api/[...path]/route.ts
web/components/workspace.tsx
web/components/run-result.tsx
web/components/evidence.tsx
web/lib/api.ts
web/lib/types.ts
```

Modified existing files: `.gitignore`, `README.md`, `src/osint_agent/workflow.py`, and `src/osint_agent/reasoning/synthesis.py`. Existing tests were preserved. No SQLite document-schema migration was introduced; only the additive audit table/index described above was created.

### Scope and assumptions

This milestone assumes the existing local ARGUS environment and corpus, one analyst and one API worker, the current configured models, and the screenshot's desktop workspace direction. It does not install or migrate the core application, reconcile the corpus automatically, or claim that the existing distance threshold admits useful evidence for every query. The missing time filter, global confidence, and separate limitations field are omitted because the current backend does not expose those capabilities. Accepted live generation still requires a valid current index and evidence that passes the existing gate.

ARGUS retrieval, evidence independence, sufficiency, citation validation, claim support, contradiction references, provenance, and fail-closed semantics were **not weakened**. **Test Lab and Guided Runner were not implemented.**

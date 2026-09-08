"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { api, displayDate, errorMessage } from "@/lib/api";
import type { AnalystRun, DocumentSummary, Health, Page, RunSummary } from "@/lib/types";
import { SourceDrawer, type Inspect, type Inspection } from "./evidence";
import { Diagnostics, Outcome, RunResult } from "./run-result";

type View = "workspace" | "history" | "sources" | "diagnostics" | "corpus";

function EyeMark() {
  return <svg width="31" height="23" viewBox="0 0 31 23" aria-hidden="true"><path d="M2 11.5C5.1 5.6 10 3 15.5 3S25.9 5.6 29 11.5C25.9 17.4 21 20 15.5 20S5.1 17.4 2 11.5Z" fill="none" stroke="currentColor" strokeWidth="2.5" /><circle cx="15.5" cy="11.5" r="3.5" fill="currentColor" /></svg>;
}

function NavigationIcon({ name }: { name: View }) {
  const paths: Record<View, React.ReactNode> = {
    workspace: <><path d="M12 4v16M4 12h16" /></>,
    history: <><path d="M3 10a9 9 0 1 1 2 8M3 4v6h6M12 7v5l3 2" /></>,
    sources: <><path d="M5 3h10l4 4v14H5zM14 3v5h5M8 12h8M8 16h8" /></>,
    diagnostics: <><path d="M3 12h4l3-7 4 14 3-7h4" /></>,
    corpus: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v7c0 4 16 4 16 0V5M4 12v7c0 4 16 4 16 0v-7" /></>,
  };
  return <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

export function Workspace({ initialRunId }: { initialRunId: string | null }) {
  const [view, setView] = useState<View>("workspace");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [healthAttempt, setHealthAttempt] = useState(0);
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(5);
  const [run, setRun] = useState<AnalystRun | null>(null);
  const [pending, setPending] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [runError, setRunError] = useState<string | null>(null);
  const [openingRun, setOpeningRun] = useState<string | null>(initialRunId);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [historyVersion, setHistoryVersion] = useState(0);
  const requestGeneration = useRef(0);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const check = () => api<Health>("/health", { signal: controller.signal }).then((value) => {
      if (active) { setHealth(value); setHealthError(null); }
    }).catch((cause: unknown) => { if (active) { setHealthError(errorMessage(cause)); setHealth(null); } });
    void check();
    const interval = setInterval(() => { void check(); }, 60_000);
    return () => { active = false; controller.abort(); clearInterval(interval); };
  }, [healthAttempt]);

  useEffect(() => {
    if (!pending) return;
    const timer = setInterval(() => setElapsed((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, [pending]);

  useEffect(() => {
    const runId = initialRunId;
    if (!runId) return;
    const generation = requestGeneration.current;
    const controller = new AbortController();
    api<AnalystRun>(`/runs/${encodeURIComponent(runId)}`, { signal: controller.signal }).then((value) => {
      if (requestGeneration.current !== generation) return;
      setRun(value); setQuery(value.query); setLimit(value.n_results);
    }).catch((cause: unknown) => { if (!controller.signal.aborted && requestGeneration.current === generation) setRunError(errorMessage(cause)); })
      .finally(() => { if (!controller.signal.aborted && requestGeneration.current === generation) setOpeningRun(null); });
    return () => controller.abort();
  }, [initialRunId]);

  const loadRun = useCallback(async (id: string) => {
    if (pending) return;
    const generation = ++requestGeneration.current;
    setOpeningRun(id); setRunError(null);
    try {
      const value = await api<AnalystRun>(`/runs/${encodeURIComponent(id)}`);
      if (requestGeneration.current !== generation) return;
      setRun(value); setQuery(value.query); setLimit(value.n_results); setView("workspace");
      window.history.replaceState(null, "", `?run=${encodeURIComponent(value.run_id)}`);
    } catch (cause) {
      if (requestGeneration.current === generation) setRunError(errorMessage(cause));
    } finally { if (requestGeneration.current === generation) setOpeningRun(null); }
  }, [pending]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending || !query.trim()) return;
    const generation = ++requestGeneration.current;
    setPending(true); setElapsed(0); setRunError(null); setRun(null);
    window.history.replaceState(null, "", window.location.pathname);
    try {
      const result = await api<AnalystRun>("/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: query.trim(), n_results: limit }) });
      if (requestGeneration.current !== generation) return;
      setRun(result); setHistoryVersion((value) => value + 1);
      window.history.replaceState(null, "", `?run=${encodeURIComponent(result.run_id)}`);
    } catch (cause) {
      setRunError(`${errorMessage(cause)} If the connection ended during processing, check Brief history before starting another run.`);
      setHistoryVersion((value) => value + 1);
    } finally { setPending(false); }
  }

  function navigate(next: View) {
    if (next === "workspace" && !pending) {
      requestGeneration.current += 1;
      setOpeningRun(null);
      setRun(null); setQuery(""); setLimit(5); setRunError(null);
      window.history.replaceState(null, "", window.location.pathname);
    }
    setView(next);
  }

  const navItems: { id: View; label: string }[] = [{ id: "workspace", label: "New brief" }, { id: "history", label: "Brief history" }, { id: "sources", label: "Sources" }];
  const developerItems: { id: View; label: string }[] = [{ id: "diagnostics", label: "Diagnostics" }, { id: "corpus", label: "Corpus status" }];
  const readiness = health?.ollama.status === "ready" ? "Ollama ready" : health?.ollama.status === "missing_models" ? "Ollama models missing" : health?.ollama.status === "unavailable" ? "Ollama unavailable" : healthError ? "Local API unavailable" : "Checking local services";
  const inspect: Inspect = setInspection;

  return <>
    <div className="app-shell" inert={!!inspection}>
      <a className="skip-link" href="#main-content">Skip to workspace</a>
      <header className="topbar"><button className="brand" onClick={() => navigate("workspace")} aria-label="ARGUS home"><EyeMark /><span>ARGUS</span></button><span className="workspace-label">Analyst workspace</span><button className="health-summary" onClick={() => setView("corpus")} aria-label={`Corpus status: ${readiness}`}><span className={`status-dot ${health?.status === "ready" ? "ready" : healthError || health ? "degraded" : "checking"}`} /><span>{readiness}{health?.sqlite.document_count != null && <><span className="status-divider"> · </span>{health.sqlite.document_count.toLocaleString()} documents</>}</span></button></header>
      <div className="workspace-layout">
        <aside className="sidebar"><nav aria-label="Main navigation">{navItems.map((item) => <button key={item.id} className={`nav-button ${view === item.id ? "active" : ""}`} aria-current={view === item.id ? "page" : undefined} onClick={() => navigate(item.id)}><NavigationIcon name={item.id} />{item.label}</button>)}<p className="nav-group-label">Developer</p>{developerItems.map((item) => <button key={item.id} className={`nav-button ${view === item.id ? "active" : ""}`} aria-current={view === item.id ? "page" : undefined} onClick={() => setView(item.id)}><NavigationIcon name={item.id} />{item.label}</button>)}</nav><div className="local-note"><span className="status-dot ready" /><span>Local workspace<br /><small>Local storage · Local models</small></span></div></aside>
        <main id="main-content" tabIndex={-1}>
          {pending && view !== "workspace" && <button className="processing-banner" onClick={() => setView("workspace")}><span className="spinner" />Your intelligence question is processing. Return to the workspace →</button>}
          {view !== "workspace" && runError && <div className="notice error" role="alert">{runError}</div>}
          {view === "workspace" && <>
            <div className="page-heading"><h1>Generate an intelligence brief</h1><p>Ask a focused question. ARGUS retrieves, checks, and cites available evidence.</p></div>
            {healthError && <div className="notice warning" role="status">{healthError}</div>}
            {health?.status === "degraded" && <div className="notice warning health-warning" role="status"><span>Some local services need attention. Brief generation may be unavailable.</span><button className="text-button" onClick={() => setView("corpus")}>View status</button></div>}
            <form className="query-panel" onSubmit={submit} aria-busy={pending}>
              <label htmlFor="intelligence-question">Intelligence question</label><textarea id="intelligence-question" placeholder="What do the available sources establish about…" value={query} maxLength={4000} required disabled={pending || !!openingRun} onChange={(event) => setQuery(event.target.value)} rows={4} aria-describedby="question-help" />
              <div className="query-controls"><div className="query-hint" id="question-help">Grounded in your local corpus.<br /><span>{query.length.toLocaleString()} / 4,000 characters</span></div><div className="limit-control"><label htmlFor="evidence-limit">Evidence limit</label><select id="evidence-limit" value={limit} disabled={pending || !!openingRun} onChange={(event) => setLimit(Number(event.target.value))}>{![5, 10, 20].includes(limit) && <option value={limit}>{limit} chunks</option>}{[5, 10, 20].map((value) => <option key={value} value={value}>{value} chunks</option>)}</select></div><button className="primary-button" disabled={pending || !!openingRun || !query.trim()} type="submit">{pending ? <><span className="spinner" />Processing…</> : "Generate brief"}</button></div>
            </form>
            {pending && <div className="processing-state" role="status" aria-live="polite"><span className="spinner" /><div><strong>ARGUS is processing your question</strong><p>Local retrieval and model checks may take several minutes. Your result will appear when the workflow completes.</p><span className="elapsed">Elapsed: {Math.floor(elapsed / 60)}m {elapsed % 60}s</span></div></div>}
            {openingRun && <p className="loading-label" role="status"><span className="spinner" />Opening saved run…</p>}
            {runError && <div className="notice error" role="alert"><h2>Request could not complete</h2><p>{runError}</p></div>}
            {run && !pending && <RunResult run={run} inspect={inspect} diagnostics={() => setView("diagnostics")} refresh={() => { void loadRun(run.run_id); }} />}
            {!run && !pending && !runError && !openingRun && <div className="workspace-empty"><span className="eyebrow">From question to traceable evidence</span><h2>A clear question is the starting point.</h2><p>ARGUS assesses the retrieved evidence before generating a brief, then checks its citations and claim support. If evidence is insufficient or validation fails, no accepted brief is presented.</p><div className="workflow-path" aria-label="ARGUS workflow"><span>Question</span><b aria-hidden="true">→</b><span>Retrieval</span><b aria-hidden="true">→</b><span>Evidence gate</span><b aria-hidden="true">→</b><span>Reasoning & validation</span><b aria-hidden="true">→</b><span>Brief</span></div></div>}
          </>}
          {view === "history" && <History key={historyVersion} openRun={loadRun} openingRun={pending ? "active-run" : openingRun} />}
          {view === "sources" && <Documents inspect={inspect} />}
          {view === "diagnostics" && <Diagnostics run={run} returnToBrief={() => setView("workspace")} />}
          {view === "corpus" && <Corpus health={health} error={healthError} refresh={() => setHealthAttempt((value) => value + 1)} />}
        </main>
      </div>
    </div>
    {inspection && <SourceDrawer key={JSON.stringify(inspection)} inspection={inspection} close={() => setInspection(null)} />}
  </>;
}

function usePage<T>(endpoint: string) {
  const [offset, setOffset] = useState(0);
  const [attempt, setAttempt] = useState(0);
  const [page, setPage] = useState<Page<T> | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    api<Page<T>>(`${endpoint}?limit=20&offset=${offset}`, { signal: controller.signal }).then(setPage).catch((cause: unknown) => { if (!controller.signal.aborted) setError(errorMessage(cause)); });
    return () => controller.abort();
  }, [endpoint, offset, attempt]);
  return { page, error, changePage: (next: number) => { setPage(null); setError(null); setOffset(next); }, retry: () => { setError(null); setAttempt((value) => value + 1); } };
}

function Pagination({ page, changePage }: { page: Page<unknown>; changePage: (offset: number) => void }) {
  return <div className="pagination"><span>{page.total === 0 ? "0 results" : `${page.offset + 1}–${page.offset + page.items.length} of ${page.total.toLocaleString()}`}</span><div><button className="secondary-button" disabled={page.offset === 0} onClick={() => changePage(Math.max(0, page.offset - page.limit))}>Previous</button><button className="secondary-button" disabled={page.offset + page.items.length >= page.total} onClick={() => changePage(page.offset + page.limit)}>Next</button></div></div>;
}

function LoadingOrError({ error, retry, label }: { error: string | null; retry: () => void; label: string }) {
  return error ? <div className="notice error" role="alert"><p>{error}</p><button className="secondary-button" onClick={retry}>Retry</button></div> : <p className="loading-label" role="status"><span className="spinner" />{label}</p>;
}

function History({ openRun, openingRun }: { openRun: (id: string) => Promise<void>; openingRun: string | null }) {
  const { page, error, changePage, retry } = usePage<RunSummary>("/runs");
  return <><div className="page-heading"><h1>Brief history</h1><p>Reopen saved runs and inspect their evidence, outcomes, and validation records.</p></div>{!page || error ? <LoadingOrError error={error} retry={retry} label="Loading run history…" /> : page.items.length === 0 ? <div className="empty-state"><h2>No runs yet</h2><p>Your intelligence questions and their recorded outcomes will appear here.</p></div> : <><div className="history-list">{page.items.map((item) => <button key={item.run_id} className="history-row" disabled={!!openingRun} onClick={() => { void openRun(item.run_id); }}><div className="history-row-main"><h2>{item.query}</h2><p>{displayDate(item.created_at)}<span> · {item.n_results} requested chunks</span></p><code>{item.run_id}</code></div><div className="history-outcome"><Outcome status={item.status} />{openingRun === item.run_id ? <span className="spinner" /> : <span aria-hidden="true">→</span>}</div></button>)}</div><Pagination page={page} changePage={changePage} /></>}</>;
}

function Documents({ inspect }: { inspect: Inspect }) {
  const { page, error, changePage, retry } = usePage<DocumentSummary>("/documents");
  return <><div className="page-heading"><h1>Sources</h1><p>Browse authoritative documents stored in the local ARGUS corpus.</p></div>{!page || error ? <LoadingOrError error={error} retry={retry} label="Loading stored documents…" /> : page.items.length === 0 ? <div className="empty-state"><h2>No stored documents</h2><p>Documents ingested through the existing ARGUS workflow will appear here.</p></div> : <><div className="panel document-browser table-scroll"><table><thead><tr><th>Document</th><th>Provider</th><th>Published</th><th><span className="sr-only">Action</span></th></tr></thead><tbody>{page.items.map((document) => <tr key={document.doc_id}><td><button className="document-link" onClick={() => inspect({ kind: "document", docId: document.doc_id })}>{document.title || "Untitled document"}</button><code className="document-id">{document.doc_id}</code></td><td>{document.provider || document.source || "Not recorded"}</td><td>{displayDate(document.published_date)}</td><td><button className="icon-button" aria-label={`Open document ${document.title || document.doc_id}`} onClick={() => inspect({ kind: "document", docId: document.doc_id })}>↗</button></td></tr>)}</tbody></table></div><Pagination page={page} changePage={changePage} /></>}</>;
}

function Corpus({ health, error, refresh }: { health: Health | null; error: string | null; refresh: () => void }) {
  return <><div className="page-heading heading-with-action"><div><h1>Corpus status</h1><p>Local storage and model readiness.</p></div><button className="secondary-button" onClick={refresh}>Refresh status</button></div>{!health ? <LoadingOrError error={error} retry={refresh} label="Checking local services…" /> : <>
    <p className="status-checked"><span className={`status-dot ${health.status === "ready" ? "ready" : "degraded"}`} />{health.status === "ready" ? "Local services are ready" : "Local services are degraded"}<span className="muted">Checked {displayDate(health.checked_at)}</span></p>
    <div className="corpus-grid"><section className="panel"><div className="section-heading"><h2>SQLite</h2><span className={`badge ${health.sqlite.status === "ready" ? "success" : "error"}`}>{health.sqlite.status}</span></div><strong className="corpus-count">{health.sqlite.document_count?.toLocaleString() ?? "—"}</strong><p className="muted">Stored documents</p>{health.sqlite.detail && <p className="small">{health.sqlite.detail}</p>}</section><section className="panel"><div className="section-heading"><h2>Chroma</h2><span className={`badge ${health.chroma.status === "ready" ? "success" : "error"}`}>{health.chroma.status}</span></div><strong className="corpus-count">{health.chroma.record_count?.toLocaleString() ?? "—"}</strong><p className="muted">Index records</p>{health.chroma.detail && <p className="small">{health.chroma.detail}</p>}</section></div>
    <section className="panel"><div className="section-heading"><h2>Ollama</h2><span className={`badge ${health.ollama.status === "ready" ? "success" : "warning"}`}>{health.ollama.status.replaceAll("_", " ")}</span></div><dl className="metadata"><dt>Reasoning model</dt><dd><code>{health.ollama.reasoning_model}</code></dd><dt>Embedding model</dt><dd><code>{health.ollama.embedding_model}</code></dd></dl>{health.ollama.detail && <p className="small">{health.ollama.detail}</p>}<p className="muted small">Readiness checks connectivity and configured model availability without running inference.</p></section>
    <section className="panel"><h2>Recorded index status</h2><p><span className="badge neutral">{health.index.status.replaceAll("_", " ")}</span></p><p>{health.index.detail}</p><p className="muted small">This is the stored index marker. It is not a real-time certification that every document is indexed.</p></section>
  </>}</>;
}

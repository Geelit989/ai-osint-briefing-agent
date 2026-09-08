"use client";

import { useEffect, useRef, useState } from "react";
import { api, displayDate, errorMessage, safeUrl } from "@/lib/api";
import type { AnalystRun, CitedStatement, EvidenceChunk, Metadata, SourceDetail, StoredDocument } from "@/lib/types";

export type Inspection = { kind: "source"; runId: string; sourceId: string } | { kind: "document"; docId: string; runId?: string };
export type Inspect = (inspection: Inspection) => void;

export function CitationLinks({ ids, run, inspect }: { ids: string[]; run: AnalystRun; inspect: Inspect }) {
  return <span className="citations">{ids.map((id) => run.sources.some((source) => source.source_id === id)
    ? <button key={id} className="citation" aria-label={`Inspect source ${id}`} onClick={() => inspect({ kind: "source", runId: run.run_id, sourceId: id })}>{id}</button>
    : <span key={id} className="citation unavailable" title="No source relationship is available for this run">{id}</span>)}</span>;
}

export function Statement({ statement, run, inspect }: { statement: CitedStatement; run: AnalystRun; inspect: Inspect }) {
  return <div className="statement">
    <p>{statement.text} <CitationLinks ids={statement.citations} run={run} inspect={inspect} /></p>
    {statement.confidence && <span className="confidence">Confidence: {statement.confidence}</span>}
    {statement.acknowledged_contradictions.length > 0 && <p className="muted small">Acknowledges: {statement.acknowledged_contradictions.join(", ")}</p>}
  </div>;
}

export function Chunk({ chunk, inspect, runId }: { chunk: EvidenceChunk; inspect?: Inspect; runId?: string }) {
  return <article className="evidence-chunk">
    <div className="row-between"><strong>{chunk.title || "Retrieved passage"}</strong>
      {inspect && <button className="text-button" onClick={() => inspect({ kind: "document", docId: chunk.doc_id, runId })}>View stored document ↗</button>}
    </div>
    <p className="muted small">{[chunk.provider, chunk.source].filter(Boolean).join(" · ") || "Provider not recorded"}</p>
    <blockquote>{chunk.text}</blockquote>
    <details className="compact-details"><summary>Evidence identifiers</summary>
      <dl className="metadata"><dt>Chunk ID</dt><dd><code>{chunk.chunk_id}</code></dd><dt>Document ID</dt><dd><code>{chunk.doc_id}</code></dd>
        <dt>Retrieval distance</dt><dd>{chunk.distance} <span className="muted">(lower is more similar)</span></dd></dl>
    </details>
  </article>;
}

export function RetrievedEvidence({ run, inspect }: { run: AnalystRun; inspect: Inspect }) {
  return <section className="result-section" id="retrieved-evidence">
    <div className="section-heading"><h2>Retrieved evidence</h2><span className="muted small">{run.evidence.length} chunks</span></div>
    <p className="muted small">Retrieved passages remain available for review even when the evidence gate closes. Retrieval alone does not establish support for a claim.</p>
    {run.evidence.length === 0 ? <p className="empty-inline">No retrieved passages were recorded for this run.</p>
      : <div className="evidence-list">{run.evidence.map((chunk) => <Chunk key={chunk.chunk_id} chunk={chunk} inspect={inspect} runId={run.run_id} />)}</div>}
    <details className="grouping-details"><summary>Evidence grouping · {run.grouping_manifest.length} recorded units</summary>
      <p className="muted small">Independent evidence units group document identities and duplicates. They do not establish independently sourced corroboration.</p>
      {run.grouping_manifest.map((group) => <div className="group" key={group.unit_id}>
        <strong>{group.unit_id}</strong>
        <dl className="metadata"><dt>Grouping reasons</dt><dd>{group.grouping_reasons.map((reason) => reason.replaceAll("_", " ")).join(", ")}</dd>
          <dt>Document IDs</dt><dd>{group.member_document_ids.map((id) => <button className="id-link" key={id} onClick={() => inspect({ kind: "document", docId: id, runId: run.run_id })}>{id}</button>)}</dd>
          <dt>Chunk IDs</dt><dd>{group.member_chunk_ids.map((id) => <code className="block-code" key={id}>{id}</code>)}</dd></dl>
      </div>)}
      {run.grouping_manifest.length === 0 && <p className="muted">No evidence groups were recorded.</p>}
    </details>
  </section>;
}

export function SourceMetadata({ metadata, docId }: { metadata: Metadata; docId: string }) {
  const url = safeUrl(metadata.url);
  return <dl className="metadata">
    <dt>Document ID</dt><dd><code>{docId}</code></dd>
    <dt>Provider</dt><dd>{metadata.provider || "Not recorded"}</dd>
    <dt>Source</dt><dd>{metadata.source || "Not recorded"}</dd>
    <dt>Published</dt><dd>{displayDate(metadata.published_date)}</dd>
    <dt>Retrieved</dt><dd>{displayDate(metadata.retrieved_at)}</dd>
    {metadata.event_time && <><dt>Event time</dt><dd>{displayDate(metadata.event_time)}</dd></>}
    {metadata.source_type && <><dt>Source type</dt><dd>{metadata.source_type}</dd></>}
    <dt>Original URL</dt><dd>{url ? <a href={url} target="_blank" rel="noopener noreferrer">{url} ↗</a> : "Not available"}</dd>
  </dl>;
}

export function SourceDrawer({ inspection, close }: { inspection: Inspection; close: () => void }) {
  const [data, setData] = useState<SourceDetail | StoredDocument | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDocument, setShowDocument] = useState(inspection.kind === "document");
  const [attempt, setAttempt] = useState(0);
  const dialog = useRef<HTMLDivElement>(null);
  const closeRef = useRef(close);
  useEffect(() => { closeRef.current = close; }, [close]);
  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialog.current?.focus();
    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape") { event.preventDefault(); closeRef.current(); }
      if (event.key !== "Tab" || !dialog.current) return;
      const elements = Array.from(dialog.current.querySelectorAll<HTMLElement>("button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex='0']"))
        .filter((element) => element.getClientRects().length > 0);
      const first = elements[0];
      const last = elements[elements.length - 1];
      if (!first) { event.preventDefault(); dialog.current.focus(); return; }
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) {
        event.preventDefault(); last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === dialog.current)) {
        event.preventDefault(); first.focus();
      }
    }
    document.addEventListener("keydown", keydown);
    return () => { document.body.style.overflow = overflow; document.removeEventListener("keydown", keydown); previous?.focus(); };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const path = inspection.kind === "source"
      ? `/runs/${encodeURIComponent(inspection.runId)}/sources/${encodeURIComponent(inspection.sourceId)}`
      : `/documents/${encodeURIComponent(inspection.docId)}${inspection.runId ? `?run_id=${encodeURIComponent(inspection.runId)}` : ""}`;
    api<SourceDetail | StoredDocument>(path, { signal: controller.signal })
      .then(setData).catch((cause: unknown) => { if (!controller.signal.aborted) setError(errorMessage(cause)); });
    return () => controller.abort();
  }, [inspection, attempt]);

  const sourceData = data && "chunks" in data ? data : null;
  const stored = sourceData ? sourceData.document : data as StoredDocument | null;
  const metadata = stored || sourceData?.source;
  const chunks = sourceData?.chunks || stored?.relevant_chunks || [];

  return <div className="drawer-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}>
    <div className="source-drawer" ref={dialog} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="source-drawer-title">
      <header className="drawer-header"><div><span className="eyebrow">Evidence inspection</span><h2 id="source-drawer-title">{inspection.kind === "source" ? `Source ${inspection.sourceId}` : "Stored document"}</h2></div><button className="icon-button" aria-label="Close source drawer" onClick={close}>×</button></header>
      <div className="drawer-body">
        {error ? <div className="notice error" role="alert"><p>{error}</p><button className="secondary-button" onClick={() => { setError(null); setAttempt(attempt + 1); }}>Retry</button></div>
          : !data ? <p className="loading-label" role="status"><span className="spinner" />Loading stored evidence…</p>
          : <>
            <h3 className="document-title">{metadata?.title || "Untitled source"}</h3>
            {metadata && <SourceMetadata metadata={metadata} docId={sourceData?.source.doc_id || stored?.doc_id || ""} />}
            {inspection.kind === "source" && <p className="muted small">{inspection.sourceId} is a citation alias within run <code>{inspection.runId}</code>.</p>}
            {sourceData && stored && <button className="secondary-button wide-button" onClick={() => setShowDocument(!showDocument)}>{showDocument ? "Return to relevant evidence" : "View stored document"}</button>}
            {sourceData && !stored && <p className="notice warning">{sourceData.document_error || "The underlying document is unavailable in SQLite."} Recorded retrieved passages are shown below.</p>}
            {showDocument && stored && <section className="drawer-section"><h3>Authoritative stored text</h3><p className="muted small">Cleaned text stored by ARGUS. Reading it does not require access to the original website.</p><div className="document-text">{stored.text}</div>
              <details className="grouping-details"><summary>Raw stored text</summary><div className="document-text">{stored.raw_text}</div></details>
            </section>}
            <section className="drawer-section"><h3>Relevant retrieved evidence</h3>
              {chunks.length > 0 ? chunks.map((chunk) => <Chunk key={chunk.chunk_id} chunk={chunk} />) : <p className="muted">No run-local retrieved passages are associated with this view.</p>}
            </section>
            {sourceData && sourceData.claims.length > 0 && <section className="drawer-section"><h3>Claim-support links</h3>{sourceData.claims.map((claim) => <article className="source-claim" key={claim.claim_id}>
              <div className="row-between"><strong>{claim.claim_id}</strong><span className={`badge ${claim.status === "supported" ? "success" : "error"}`}>{claim.status}</span></div><p>{claim.text}</p>
              {claim.supporting_spans.filter((span) => span.source_id === sourceData.source.source_id).map((span, index) => <div className="exact-span" key={`${span.chunk_id}-${index}`}><blockquote>{span.text}</blockquote><p className="muted small">Exact supporting span · <code>{span.chunk_id}</code> · characters {span.start}–{span.end}</p></div>)}
            </article>)}</section>}
          </>}
      </div>
    </div>
  </div>;
}

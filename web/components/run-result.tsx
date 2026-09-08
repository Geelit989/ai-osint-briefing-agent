"use client";

import { displayDate } from "@/lib/api";
import type { AnalystRun, RunStatus } from "@/lib/types";
import { CitationLinks, RetrievedEvidence, Statement, type Inspect } from "./evidence";

export const outcomeLabels: Record<RunStatus, string> = {
  success: "Accepted brief",
  insufficient_evidence: "Insufficient evidence",
  validation_failure: "Validation failure",
  error: "System error",
  running: "Running",
};

export function Outcome({ status }: { status: RunStatus }) {
  const tone = status === "success" ? "success" : status === "insufficient_evidence" ? "warning" : status === "running" ? "neutral" : "error";
  return <span className={`badge ${tone}`}>{outcomeLabels[status]}</span>;
}

export function Metrics({ run }: { run: AnalystRun }) {
  return <>
    <div className="metrics" aria-label="Evidence assessment">
      <div className="metric"><strong>{run.retrieval?.retrieved_chunk_count ?? "—"}</strong><span>Retrieved chunks</span></div>
      <div className="metric"><strong>{run.retrieval?.usable_chunk_count ?? "—"}</strong><span>Usable chunks</span></div>
      <div className="metric"><strong>{run.retrieval?.independent_evidence_count ?? "—"}</strong><span>Independent evidence</span></div>
      <div className={`metric gate ${run.sufficiency?.status === "SUFFICIENT" ? "sufficient" : ""}`}><strong>{run.sufficiency?.status === "SUFFICIENT" ? "Sufficient" : run.sufficiency?.status === "INSUFFICIENT" ? "Insufficient" : "Not assessed"}</strong><span>Evidence gate</span></div>
    </div>
    {run.retrieval && <p className="metrics-note">Independent evidence counts distinct evidence identities; it does not establish independent corroboration.</p>}
  </>;
}

export function RunResult({ run, inspect, diagnostics, refresh }: { run: AnalystRun; inspect: Inspect; diagnostics: () => void; refresh: () => void }) {
  const brief = run.status === "success" ? run.brief : null;
  return <div className="run-result">
    <div className="recorded-question"><span className="eyebrow">Question for this run</span><p>{run.query}</p></div>
    <div className="run-meta"><Outcome status={run.status} /><span>{displayDate(run.created_at)}</span><button className="text-button" onClick={diagnostics}>View diagnostics</button></div>
    <Metrics run={run} />
    {run.status === "running" && <div className="notice neutral" role="status"><h2>Run recorded as in progress</h2><p>No final result has been recorded yet. Refresh to check whether this run has finished.</p><button className="secondary-button" onClick={refresh}>Refresh run</button></div>}
    {run.status === "insufficient_evidence" && <div className="notice warning outcome-panel"><span className="eyebrow">ARGUS did not generate a brief</span><h2>Insufficient independent evidence</h2><p>{run.sufficiency?.reason || run.failure_detail?.message || "The retrieved evidence did not satisfy the ARGUS evidence gate."}</p><a className="secondary-button" href="#retrieved-evidence">Review retrieved evidence ↓</a></div>}
    {run.status === "validation_failure" && <div className="notice error outcome-panel"><span className="eyebrow">No accepted brief</span><h2>Validation rejected the generated draft</h2><p>{run.failure_detail?.message || "An existing ARGUS validation contract failed. The draft is withheld."}</p><button className="secondary-button" onClick={diagnostics}>Inspect validation results</button></div>}
    {run.status === "error" && <div className="notice error outcome-panel"><span className="eyebrow">Run could not complete</span><h2>ARGUS encountered a system error</h2><p>{run.failure_detail?.message || "The local workflow could not complete."}</p><button className="secondary-button" onClick={diagnostics}>View technical details</button></div>}
    {brief && <article className="accepted-brief" aria-label="Accepted intelligence brief">
      <div className="brief-heading"><span className="eyebrow">Intelligence brief</span><h2>{brief.title.text}</h2><CitationLinks ids={brief.title.citations} run={run} inspect={inspect} /></div>
      <section className="bluf"><h2>BLUF</h2><Statement statement={brief.bluf} run={run} inspect={inspect} /></section>
      {brief.reported_developments.length > 0 && <section className="result-section"><h2>Reported developments</h2>{brief.reported_developments.map((statement, index) => <Statement key={index} statement={statement} run={run} inspect={inspect} />)}</section>}
      {brief.analytic_assessments.length > 0 && <section className="result-section"><h2>Analytic assessments</h2>{brief.analytic_assessments.map((statement, index) => <Statement key={index} statement={statement} run={run} inspect={inspect} />)}</section>}
      {brief.intelligence_gaps.length > 0 && <section className="result-section"><h2>Intelligence gaps</h2>{brief.intelligence_gaps.map((statement, index) => <Statement key={index} statement={statement} run={run} inspect={inspect} />)}</section>}
      {run.claims.length > 0 && <section className="result-section"><h2>Claim support</h2><div className="table-scroll"><table className="claim-table"><thead><tr><th>Claim</th><th>Support</th><th>Sources</th></tr></thead><tbody>{run.claims.map((claim) => <tr key={claim.claim_id}><td><p>{claim.text}</p><details className="claim-details"><summary>Support detail</summary><p>{claim.rationale}</p>{claim.issues.length > 0 && <p>{claim.issues.map((issue) => issue.replaceAll("_", " ")).join(", ")}</p>}{claim.supporting_spans.map((span, index) => <div className="exact-span" key={index}><blockquote>{span.text}</blockquote><p className="muted small"><CitationLinks ids={[span.source_id]} run={run} inspect={inspect} /> <code>{span.chunk_id}</code> · characters {span.start}–{span.end}</p></div>)}</details></td><td><span className={`support-label ${claim.status}`}>{claim.status === "supported" ? "Supported" : "Unsupported"}</span></td><td><CitationLinks ids={claim.citations} run={run} inspect={inspect} /></td></tr>)}</tbody></table></div></section>}
      {run.known_contradictions.length > 0 && <section className="result-section"><h2>Known contradictions</h2><p className="muted small">Conflicting positions explicitly recorded in the selected evidence.</p>{run.known_contradictions.map((contradiction) => <div className="contradiction" key={contradiction.contradiction_id}><strong>{contradiction.contradiction_id}</strong><div><CitationLinks ids={contradiction.source_ids} run={run} inspect={inspect} /><p className="muted small">Recorded positions: {contradiction.positions.join(", ")}</p></div></div>)}</section>}
      {run.sources.length > 0 && <section className="result-section"><h2>Supporting sources</h2><div className="source-list">{run.sources.map((source) => <button className="source-row" key={source.source_id} onClick={() => inspect({ kind: "source", runId: run.run_id, sourceId: source.source_id })}><span className="citation static">{source.source_id}</span><span><strong>{source.title || source.doc_id}</strong><small>{source.provider || source.source || "Provider not recorded"}</small></span><span aria-hidden="true">↗</span></button>)}</div></section>}
    </article>}
    {run.status === "success" && !brief && <div className="notice error">The stored run has no accepted brief attached. Review its diagnostics.</div>}
    {run.sufficiency && run.status !== "insufficient_evidence" && <details className="grouping-details"><summary>Evidence assessment reason</summary><p>{run.sufficiency.reason}</p></details>}
    <RetrievedEvidence run={run} inspect={inspect} />
    {run.cli_equivalent && <details className="cli-equivalent"><summary>Show CLI equivalent</summary><p className="muted small">Reproduce the query with the existing ARGUS command line workflow.</p><pre><code>{run.cli_equivalent}</code></pre></details>}
    <p className="audit-id">Run ID <code>{run.run_id}</code></p>
  </div>;
}

export function Diagnostics({ run, returnToBrief }: { run: AnalystRun | null; returnToBrief: () => void }) {
  return <>
    <div className="page-heading"><h1>Diagnostics</h1><p>Recorded checks and audit details for the selected run.</p></div>
    {!run ? <div className="empty-state"><h2>No run selected</h2><p>Generate a brief or open a run from Brief history to inspect its recorded checks.</p></div> : <>
      <div className="selected-run"><div><Outcome status={run.status} /><h2>{run.query}</h2><p className="muted small">{displayDate(run.created_at)} · <code>{run.run_id}</code></p></div><button className="secondary-button" onClick={returnToBrief}>Return to result</button></div>
      <div className="panel"><h2>Pipeline checks</h2><p className="muted small">These are the stages reported by ARGUS. A check marked “Not run” has no passing result.</p>
        <div className="stage-list">{run.validation.stages.map((stage) => <div className="stage-row" key={stage.name}><div><strong>{stage.name.replaceAll("_", " ")}</strong>{stage.detail && <p className="muted small">{stage.detail}</p>}</div><span className={`badge ${stage.status === "pass" ? "success" : stage.status === "fail" ? "error" : "neutral"}`}>{stage.status === "pass" ? "Pass" : stage.status === "fail" ? "Fail" : "Not run"}</span></div>)}</div>
        {run.validation.stages.length === 0 && <p>No pipeline stage results were recorded.</p>}
      </div>
      {run.failure_detail && <section className="panel"><h2>Failure detail</h2><p>{run.failure_detail.message}</p><dl className="metadata"><dt>Failure type</dt><dd>{run.failure_detail.type}</dd></dl>{run.failure_detail.technical_detail && <details className="grouping-details"><summary>Technical detail</summary><pre>{run.failure_detail.technical_detail}</pre></details>}</section>}
      {run.validation.claim_support && <section className="panel"><h2>Claim-support validation</h2><p className="muted small">{run.status === "validation_failure" ? "Diagnostic judgments below refer to a rejected draft. They are not an accepted intelligence product." : "Recorded judgments from the ARGUS support validator."}</p>{run.validation.claim_support.judgments.map((judgment) => <div className="source-claim" key={judgment.claim_id}><div className="row-between"><strong>{judgment.claim_id}</strong><span className={`badge ${judgment.status === "supported" ? "success" : "error"}`}>{judgment.status}</span></div><p>{judgment.claim_text}</p><p className="muted small">{judgment.rationale}</p>{judgment.issues.length > 0 && <p className="small">Issues: {judgment.issues.map((issue) => issue.replaceAll("_", " ")).join(", ")}</p>}</div>)}</section>}
    </>}
  </>;
}

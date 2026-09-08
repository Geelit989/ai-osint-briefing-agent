// Transport fields mirror the adapter and existing ARGUS domain models.
// The browser displays these decisions; it never recomputes evidence support.
export type Metadata = {
  title?: string | null;
  source?: string | null;
  provider?: string | null;
  source_type?: string | null;
  published_date?: string | null;
  event_time?: string | null;
  retrieved_at?: string | null;
  url?: string | null;
};

export type SourceReference = Metadata & { source_id: string; doc_id: string; chunk_ids: string[] };
export type EvidenceChunk = Metadata & { chunk_id: string; doc_id: string; text: string; distance: number };
export type StoredDocument = Metadata & {
  doc_id: string;
  text: string;
  raw_text: string;
  meta_data?: Record<string, unknown>;
  relevant_chunks?: EvidenceChunk[];
};
export type DocumentSummary = Metadata & { doc_id: string };
export type SupportingSpan = { source_id: string; chunk_id: string; start: number; end: number; text: string };
export type CitedStatement = {
  text: string;
  citations: string[];
  supporting_spans: SupportingSpan[];
  acknowledged_contradictions: string[];
  confidence?: "low" | "moderate" | "high";
};
export type RunClaim = CitedStatement & {
  claim_id: string;
  status: "supported" | "unsupported";
  issues: string[];
  rationale: string;
};
export type KnownContradiction = {
  contradiction_id: string;
  source_ids: string[];
  chunk_ids: string[];
  positions: string[];
};
export type Brief = {
  query: string;
  generated_date: string;
  title: CitedStatement;
  bluf: CitedStatement;
  reported_developments: CitedStatement[];
  analytic_assessments: CitedStatement[];
  intelligence_gaps: CitedStatement[];
  sources: SourceReference[];
  known_contradictions: KnownContradiction[];
};
export type EvidenceGroup = {
  unit_id: string;
  member_chunk_ids: string[];
  member_document_ids: string[];
  grouping_reasons: string[];
};
export type RunStatus = "running" | "success" | "insufficient_evidence" | "validation_failure" | "error";
export type RunSummary = {
  run_id: string;
  query: string;
  created_at: string;
  status: RunStatus;
  n_results: number;
};
export type AnalystRun = RunSummary & {
  retrieval: { retrieved_chunk_count: number; usable_chunk_count: number; independent_evidence_count: number } | null;
  sufficiency: { status: "SUFFICIENT" | "INSUFFICIENT"; reason: string } | null;
  brief: Brief | null;
  claims: RunClaim[];
  sources: SourceReference[];
  evidence: EvidenceChunk[];
  grouping_manifest: EvidenceGroup[];
  known_contradictions: KnownContradiction[];
  validation: {
    stages: { name: string; status: "pass" | "fail" | "not_run"; detail?: string | null }[];
    claim_support: {
      status: "supported" | "unsupported";
      judgments: { claim_id: string; claim_text: string; status: string; issues: string[]; rationale: string }[];
    } | null;
  };
  failure_detail: { type: string; message: string; technical_detail?: string | null } | null;
  cli_equivalent: string;
};
export type SourceDetail = { source: SourceReference; document: StoredDocument | null; document_error?: string | null; chunks: EvidenceChunk[]; claims: RunClaim[] };
export type Page<T> = { items: T[]; total: number; limit: number; offset: number };
export type Health = {
  status: "ready" | "degraded";
  checked_at: string;
  sqlite: { status: "ready" | "unavailable"; document_count: number | null; detail?: string | null };
  chroma: { status: "ready" | "unavailable"; record_count: number | null; detail?: string | null };
  ollama: { status: "ready" | "unavailable" | "missing_models"; reasoning_model: string; embedding_model: string; detail?: string | null };
  index: { status: string; detail: string };
};

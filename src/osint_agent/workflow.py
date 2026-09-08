"""Thin orchestration layer for retrieval, gating, and bounded synthesis."""

from datetime import datetime

from osint_agent.models.brief import BriefResult, InsufficientEvidenceResult
from osint_agent.models.document import EvidenceChunk
from osint_agent.models.workflow import WorkflowTrace, trace_stage
from osint_agent.reasoning.claim_support import ClaimSupportModel
from osint_agent.reasoning.synthesis import StructuredReasoningModel, synthesize_brief
from osint_agent.retrieval.semantic import semantic_search
from osint_agent.retrieval.sufficiency import check_retrieval_sufficiency


def reason_over_evidence(
    query: str,
    evidence: list[EvidenceChunk],
    max_distance: float,
    min_evidence: int,
    model: StructuredReasoningModel | None = None,
    support_model: ClaimSupportModel | None = None,
    reference_time: datetime | None = None,
    trace: WorkflowTrace | None = None,
) -> BriefResult:
    """Enforce the sufficiency gate before any model can be invoked."""

    if trace is not None:
        trace.evidence = list(evidence)
    with trace_stage(trace, "evidence_assessment"):
        assessment = check_retrieval_sufficiency(
            evidence, max_distance=max_distance, min_evidence=min_evidence
        )
    if trace is not None:
        trace.assessment = assessment
        trace.record(
            "sufficiency", "pass" if assessment.sufficient else "fail",
            assessment.reason,
        )
    if not assessment.sufficient:
        return InsufficientEvidenceResult(
            reason=assessment.reason,
            evidence_count=assessment.retrieved_chunk_count,
            usable_evidence_count=assessment.usable_chunk_count,
            retrieved_chunk_count=assessment.retrieved_chunk_count,
            usable_chunk_count=assessment.usable_chunk_count,
            independent_evidence_count=assessment.independent_evidence_count,
            grouping_manifest=assessment.grouping_manifest,
        )
    synthesis_kwargs = {
        "model": model,
        "support_model": support_model,
    }
    if reference_time is not None:
        synthesis_kwargs["reference_time"] = reference_time
    if trace is not None:
        synthesis_kwargs["trace"] = trace
    return synthesize_brief(
        query,
        assessment.usable_evidence,
        **synthesis_kwargs,
    )


def generate_brief_for_query(
    query: str,
    max_distance: float,
    min_evidence: int,
    n_results: int = 5,
    model: StructuredReasoningModel | None = None,
    support_model: ClaimSupportModel | None = None,
    reference_time: datetime | None = None,
    trace: WorkflowTrace | None = None,
) -> BriefResult:
    """Run the existing retrieval interface, then gate and synthesize."""

    with trace_stage(trace, "retrieval"):
        evidence = semantic_search(query, n_results=n_results)
    trace_kwargs = {"trace": trace} if trace is not None else {}
    return reason_over_evidence(
        query,
        evidence,
        max_distance,
        min_evidence,
        model=model,
        support_model=support_model,
        reference_time=reference_time,
        **trace_kwargs,
    )

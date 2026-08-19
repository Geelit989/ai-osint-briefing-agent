"""Thin orchestration layer for retrieval, gating, and bounded synthesis."""

from osint_agent.models.brief import BriefResult, InsufficientEvidenceResult
from osint_agent.models.document import EvidenceChunk
from osint_agent.reasoning.synthesis import StructuredReasoningModel, synthesize_brief
from osint_agent.retrieval.semantic import semantic_search
from osint_agent.retrieval.sufficiency import check_retrieval_sufficiency


def reason_over_evidence(
    query: str,
    evidence: list[EvidenceChunk],
    max_distance: float,
    min_evidence: int,
    model: StructuredReasoningModel | None = None,
) -> BriefResult:
    """Enforce the sufficiency gate before any model can be invoked."""

    assessment = check_retrieval_sufficiency(
        evidence, max_distance=max_distance, min_evidence=min_evidence
    )
    if not assessment.sufficient:
        return InsufficientEvidenceResult(
            reason=assessment.reason,
            evidence_count=assessment.evidence_count,
            usable_evidence_count=len(assessment.usable_evidence),
        )
    return synthesize_brief(query, assessment.usable_evidence, model=model)


def generate_brief_for_query(
    query: str,
    max_distance: float,
    min_evidence: int,
    n_results: int = 5,
    model: StructuredReasoningModel | None = None,
) -> BriefResult:
    """Run the existing retrieval interface, then gate and synthesize."""

    evidence = semantic_search(query, n_results=n_results)
    return reason_over_evidence(
        query, evidence, max_distance, min_evidence, model=model
    )

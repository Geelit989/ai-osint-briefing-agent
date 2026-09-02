"""Thin orchestration layer for retrieval, gating, and bounded synthesis."""

from osint_agent.models.brief import BriefResult, InsufficientEvidenceResult
from osint_agent.models.document import EvidenceChunk
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
) -> BriefResult:
    """Enforce the sufficiency gate before any model can be invoked."""

    assessment = check_retrieval_sufficiency(
        evidence, max_distance=max_distance, min_evidence=min_evidence
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
    return synthesize_brief(
        query,
        assessment.usable_evidence,
        model=model,
        support_model=support_model,
    )


def generate_brief_for_query(
    query: str,
    max_distance: float,
    min_evidence: int,
    n_results: int = 5,
    model: StructuredReasoningModel | None = None,
    support_model: ClaimSupportModel | None = None,
) -> BriefResult:
    """Run the existing retrieval interface, then gate and synthesize."""

    evidence = semantic_search(query, n_results=n_results)
    return reason_over_evidence(
        query,
        evidence,
        max_distance,
        min_evidence,
        model=model,
        support_model=support_model,
    )

from collections.abc import Mapping

from osint_agent.models.document import Document, EvidenceChunk
from osint_agent.models.retrieval import RetrievalAssessment
from osint_agent.retrieval.evidence_identity import group_usable_evidence


def check_retrieval_sufficiency(
    evidence: list[EvidenceChunk],
    max_distance: float,
    min_evidence: int,
    *,
    authoritative_documents: Mapping[str, Document] | None = None,
) -> RetrievalAssessment:
    """Check whether retrieval returned enough distinct relevant evidence.

    ``max_distance`` is an inclusive raw-distance threshold, and
    ``min_evidence`` is the minimum number of deterministic independent
    evidence units required. This accounting does not prove corroboration.
    """

    if min_evidence < 1:
        raise ValueError("min_evidence must be at least 1")

    retrieved_chunk_count = len(evidence)
    best_distance = min(
        (chunk.distance for chunk in evidence),
        default=None,
    )
    usable_evidence = [
        chunk for chunk in evidence if chunk.distance <= max_distance
    ]
    usable_count = len(usable_evidence)
    grouping_manifest = group_usable_evidence(
        usable_evidence,
        authoritative_documents=authoritative_documents,
    )
    independent_count = len(grouping_manifest)

    if not evidence:
        reason = "no evidence retrieved"
    elif not usable_evidence:
        reason = "no evidence met relevance threshold"
    elif independent_count < min_evidence:
        reason = (
            "insufficient independent evidence: "
            f"{independent_count} < {min_evidence} "
            f"from {usable_count} usable chunks"
        )
    else:
        reason = (
            f"sufficient evidence: {independent_count} independent evidence "
            f"units from {usable_count} usable chunks"
        )

    return RetrievalAssessment(
        sufficient=independent_count >= min_evidence,
        reason=reason,
        evidence_count=retrieved_chunk_count,
        retrieved_chunk_count=retrieved_chunk_count,
        usable_chunk_count=usable_count,
        independent_evidence_count=independent_count,
        grouping_manifest=grouping_manifest,
        best_distance=best_distance,
        usable_evidence=usable_evidence,
    )

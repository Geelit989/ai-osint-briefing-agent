from osint_agent.models.document import EvidenceChunk
from osint_agent.models.retrieval import RetrievalAssessment


def check_retrieval_sufficiency(
    evidence: list[EvidenceChunk],
    max_distance: float,
    min_evidence: int,
) -> RetrievalAssessment:
    """Check whether retrieval returned enough relevant evidence.

    ``max_distance`` is an inclusive raw-distance threshold, and
    ``min_evidence`` is the minimum number of chunks that must meet it.
    """

    if min_evidence < 1:
        raise ValueError("min_evidence must be at least 1")

    evidence_count = len(evidence)
    best_distance = min(
        (chunk.distance for chunk in evidence),
        default=None,
    )
    usable_evidence = [
        chunk for chunk in evidence if chunk.distance <= max_distance
    ]
    usable_count = len(usable_evidence)

    if not evidence:
        reason = "no evidence retrieved"
    elif not usable_evidence:
        reason = "no evidence met relevance threshold"
    elif usable_count < min_evidence:
        reason = (
            f"insufficient usable evidence: {usable_count} < {min_evidence}"
        )
    else:
        reason = f"sufficient evidence: {usable_count} usable chunks"

    return RetrievalAssessment(
        sufficient=usable_count >= min_evidence,
        reason=reason,
        evidence_count=evidence_count,
        best_distance=best_distance,
        usable_evidence=usable_evidence,
    )

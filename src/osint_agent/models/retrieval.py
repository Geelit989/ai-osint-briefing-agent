from pydantic import BaseModel

from osint_agent.models.document import EvidenceChunk


class RetrievalAssessment(BaseModel):
    """Deterministic assessment of evidence returned by semantic retrieval."""

    sufficient: bool
    reason: str
    evidence_count: int
    best_distance: float | None
    usable_evidence: list[EvidenceChunk]

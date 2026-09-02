from typing import Literal

from pydantic import BaseModel, Field, model_validator

from osint_agent.models.document import EvidenceChunk


GroupingReason = Literal[
    "same_doc_id",
    "canonical_url",
    "content_fingerprint",
    "singleton",
]


class EvidenceGroup(BaseModel):
    """One deterministic evidence-identity unit for sufficiency accounting."""

    unit_id: str
    member_chunk_ids: list[str] = Field(min_length=1)
    member_document_ids: list[str] = Field(min_length=1)
    grouping_reasons: list[GroupingReason] = Field(min_length=1)


class RetrievalAssessment(BaseModel):
    """Separated retrieval, relevance, and evidence-identity accounting.

    Independent evidence units are deterministic accounting constructs. They
    do not establish independently sourced corroboration.
    """

    sufficient: bool
    reason: str
    evidence_count: int = Field(
        ge=0,
        description="Deprecated alias of retrieved_chunk_count.",
    )
    retrieved_chunk_count: int = Field(ge=0)
    usable_chunk_count: int = Field(ge=0)
    independent_evidence_count: int = Field(ge=0)
    grouping_manifest: list[EvidenceGroup]
    best_distance: float | None
    usable_evidence: list[EvidenceChunk]

    @model_validator(mode="after")
    def counts_must_be_consistent(self) -> "RetrievalAssessment":
        if self.evidence_count != self.retrieved_chunk_count:
            raise ValueError("evidence_count must equal retrieved_chunk_count")
        if self.usable_chunk_count != len(self.usable_evidence):
            raise ValueError("usable_chunk_count must match usable_evidence")
        if self.independent_evidence_count != len(self.grouping_manifest):
            raise ValueError(
                "independent_evidence_count must match grouping_manifest"
            )
        if not (
            self.retrieved_chunk_count
            >= self.usable_chunk_count
            >= self.independent_evidence_count
        ):
            raise ValueError(
                "retrieval counts must be monotonically nonincreasing"
            )
        return self

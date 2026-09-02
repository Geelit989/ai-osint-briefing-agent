"""Typed intelligence-product contracts for bounded ARGUS synthesis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from osint_agent.models.retrieval import EvidenceGroup


Confidence = Literal["low", "moderate", "high"]
ClaimSupportIssue = Literal[
    "unsupported_claim",
    "modality_strengthening",
    "attribution_loss",
    "unsupported_inference",
    "contradiction",
    "partial_support",
    "uncited_claim",
    "missing_support_span",
    "invalid_span_provenance",
    "invalid_claim_evidence_association",
]
SemanticSupportIssue = Literal[
    "unsupported_claim",
    "modality_strengthening",
    "attribution_loss",
    "unsupported_inference",
    "contradiction",
    "partial_support",
]


class SupportingSpan(BaseModel):
    """A verbatim, character-addressed span from one retrieved chunk."""

    source_id: str
    chunk_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> "SupportingSpan":
        if self.end <= self.start:
            raise ValueError("supporting span end must be greater than start")
        return self


class CitedStatement(BaseModel):
    """One claim validated in full against its cited, verbatim evidence."""

    text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    supporting_spans: list[SupportingSpan] = Field(default_factory=list)


class ReportedDevelopment(CitedStatement):
    """A reported factual development and its supplied source citations."""


class AnalyticAssessment(CitedStatement):
    """An evidence-bounded analytic judgment, kept separate from reporting."""

    confidence: Confidence


class IntelligenceGap(CitedStatement):
    """A claim about what the supplied evidence does or does not establish."""


class GeneratedBrief(BaseModel):
    title: CitedStatement
    bluf: CitedStatement
    reported_developments: list[ReportedDevelopment]
    analytic_assessments: list[AnalyticAssessment]
    intelligence_gaps: list[IntelligenceGap]


class SemanticSupportDecision(BaseModel):
    """Strict structured output returned by the bounded semantic validator."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    status: Literal["supported", "unsupported"]
    issues: list[SemanticSupportIssue]

    @model_validator(mode="after")
    def status_and_issues_must_agree(self) -> "SemanticSupportDecision":
        if self.status == "supported" and self.issues:
            raise ValueError("supported decision cannot contain support issues")
        if self.status == "unsupported" and not self.issues:
            raise ValueError("unsupported decision must contain a support issue")
        return self


class ClaimSupportJudgment(BaseModel):
    """Auditable support result for one generated claim."""

    claim_id: str
    claim_text: str
    status: Literal["supported", "unsupported"]
    issues: list[ClaimSupportIssue]
    rationale: str


class ClaimSupportReport(BaseModel):
    """Post-citation claim-support validation result."""

    status: Literal["supported", "unsupported"]
    judgments: list[ClaimSupportJudgment]


class SourceReference(BaseModel):
    """Deterministic document-level source identity."""

    source_id: str
    doc_id: str
    chunk_ids: list[str]
    title: str | None = None
    source: str | None = None
    provider: str | None = None
    source_type: str | None = None
    published_date: str | None = None
    url: str | None = None


class IntelligenceBrief(GeneratedBrief):
    """Final traceable ARGUS intelligence product."""

    query: str
    generated_date: str
    sources: list[SourceReference]


class BriefSuccess(BaseModel):
    status: Literal["success"] = "success"
    brief: IntelligenceBrief
    claim_support: ClaimSupportReport


class InsufficientEvidenceResult(BaseModel):
    status: Literal["insufficient_evidence"] = "insufficient_evidence"
    reason: str
    evidence_count: int = Field(
        ge=0,
        description="Deprecated alias of retrieved_chunk_count.",
    )
    usable_evidence_count: int = Field(
        ge=0,
        description="Deprecated alias of usable_chunk_count.",
    )
    retrieved_chunk_count: int = Field(ge=0)
    usable_chunk_count: int = Field(ge=0)
    independent_evidence_count: int = Field(ge=0)
    grouping_manifest: list[EvidenceGroup]

    @model_validator(mode="after")
    def retrieval_counts_must_be_consistent(
        self,
    ) -> "InsufficientEvidenceResult":
        if self.evidence_count != self.retrieved_chunk_count:
            raise ValueError("evidence_count must equal retrieved_chunk_count")
        if self.usable_evidence_count != self.usable_chunk_count:
            raise ValueError(
                "usable_evidence_count must equal usable_chunk_count"
            )
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


BriefResult = BriefSuccess | InsufficientEvidenceResult

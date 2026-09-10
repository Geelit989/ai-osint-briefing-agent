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
    "polarity_change",
    "forecast_as_occurrence",
    "temporal_strengthening",
    "stale_currentness",
    "denial_as_fact",
    "unknown_contradiction_reference",
]
SemanticSupportIssue = Literal[
    "unsupported_claim",
    "modality_strengthening",
    "attribution_loss",
    "unsupported_inference",
    "contradiction",
    "partial_support",
    "polarity_change",
    "forecast_as_occurrence",
    "temporal_strengthening",
    "stale_currentness",
    "denial_as_fact",
]


class SupportingSpan(BaseModel):
    """A verbatim, character-addressed span from one retrieved chunk."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    supporting_spans: list[SupportingSpan] = Field(default_factory=list)
    acknowledged_contradictions: list[str] = Field(default_factory=list)


class ReportedDevelopment(CitedStatement):
    """A reported factual development and its supplied source citations."""


class AnalyticAssessment(CitedStatement):
    """An evidence-bounded analytic judgment, kept separate from reporting."""

    confidence: Confidence


class IntelligenceGap(CitedStatement):
    """A claim about what the supplied evidence does or does not establish."""


class GeneratedBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: CitedStatement
    bluf: CitedStatement
    reported_developments: list[ReportedDevelopment]
    analytic_assessments: list[AnalyticAssessment]
    intelligence_gaps: list[IntelligenceGap]


class EvidenceQuote(BaseModel):
    """Model-selected verbatim evidence, without model-authored offsets."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    text: str = Field(min_length=1)
    chunk_id: str | None = None


class SynthesisStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1)
    supporting_quotes: list[EvidenceQuote] = Field(min_length=1)
    acknowledged_contradictions: list[str] = Field(
        default_factory=list,
        description=(
            "Only exact contradiction IDs from the current request's "
            "allowed_contradiction_ids; never source IDs. Must be empty when "
            "allowed_contradiction_ids is empty."
        ),
    )


class SynthesisAssessment(SynthesisStatement):
    confidence: Confidence


class SynthesisDraft(BaseModel):
    """Generation-only contract; intelligence gaps remain substantive claims."""

    model_config = ConfigDict(extra="forbid")
    title: SynthesisStatement
    bluf: SynthesisStatement
    reported_developments: list[SynthesisStatement]
    analytic_assessments: list[SynthesisAssessment]
    intelligence_gaps: list[SynthesisStatement]


class SemanticSupportDecision(BaseModel):
    """Strict structured output returned by the bounded semantic validator."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    status: Literal["supported", "unsupported"]
    issues: list[SemanticSupportIssue] = Field(
        max_length=11,
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def status_and_issues_must_agree(self) -> "SemanticSupportDecision":
        if len(self.issues) != len(set(self.issues)):
            raise ValueError("semantic support issue codes must be unique")
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
    event_time: str | None = None
    retrieved_at: str | None = None
    url: str | None = None
    contradiction_group: str | None = None
    contradiction_position: Literal["affirmation", "denial"] | None = None


class KnownContradiction(BaseModel):
    """A conflict explicitly represented in selected evidence metadata."""

    contradiction_id: str
    source_ids: list[str] = Field(min_length=2)
    chunk_ids: list[str] = Field(min_length=2)
    positions: list[Literal["affirmation", "denial"]] = Field(min_length=2)


class IntelligenceBrief(GeneratedBrief):
    """Final traceable ARGUS intelligence product."""

    query: str
    generated_date: str
    sources: list[SourceReference]
    known_contradictions: list[KnownContradiction] = Field(default_factory=list)


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

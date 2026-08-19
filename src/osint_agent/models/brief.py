"""Typed intelligence-product contracts for bounded ARGUS synthesis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Confidence = Literal["low", "moderate", "high"]


class ReportedDevelopment(BaseModel):
    """A reported factual development and its supplied source citations."""

    text: str
    citations: list[str] = Field(min_length=1)


class AnalyticAssessment(BaseModel):
    """An evidence-bounded analytic judgment, kept separate from reporting."""

    text: str
    confidence: Confidence
    citations: list[str] = Field(min_length=1)


class CitedStatement(BaseModel):
    text: str
    citations: list[str] = Field(min_length=1)


class GeneratedBrief(BaseModel):
    title: str
    bluf: CitedStatement
    reported_developments: list[ReportedDevelopment]
    analytic_assessments: list[AnalyticAssessment]
    intelligence_gaps: list[str]


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


class InsufficientEvidenceResult(BaseModel):
    status: Literal["insufficient_evidence"] = "insufficient_evidence"
    reason: str
    evidence_count: int
    usable_evidence_count: int


BriefResult = BriefSuccess | InsufficientEvidenceResult

"""Optional execution observations for Python callers of the ARGUS workflow.

The trace records existing domain operations; it never decides whether evidence
or a generated claim is acceptable and never retains rejected model drafts.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from pydantic import BaseModel, Field

from osint_agent.models.brief import KnownContradiction, SourceReference
from osint_agent.models.document import EvidenceChunk
from osint_agent.models.retrieval import RetrievalAssessment


StageName = Literal[
    "retrieval", "evidence_assessment", "sufficiency", "reasoning",
    "structured_output", "citation_validation", "contradiction_references",
    "provenance", "claim_support",
]
STAGE_NAMES: tuple[StageName, ...] = (
    "retrieval", "evidence_assessment", "sufficiency", "reasoning",
    "structured_output", "citation_validation", "contradiction_references",
    "provenance", "claim_support",
)


class WorkflowStage(BaseModel):
    name: StageName
    status: Literal["pass", "fail", "not_run"] = "not_run"
    detail: str | None = None


class WorkflowTrace(BaseModel):
    """A per-invocation trace, supplied and owned by an interested caller."""

    evidence: list[EvidenceChunk] = Field(default_factory=list)
    assessment: RetrievalAssessment | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    known_contradictions: list[KnownContradiction] = Field(default_factory=list)
    stages: list[WorkflowStage] = Field(
        default_factory=lambda: [WorkflowStage(name=name) for name in STAGE_NAMES]
    )

    def record(
        self,
        name: StageName,
        status: Literal["pass", "fail", "not_run"],
        detail: str | None = None,
    ) -> None:
        stage = next(item for item in self.stages if item.name == name)
        stage.status = status
        stage.detail = detail


@contextmanager
def trace_stage(trace: WorkflowTrace | None, name: StageName) -> Iterator[None]:
    """Observe completion without handling, replacing, or retrying failures."""

    try:
        yield
    except Exception as exc:
        if trace is not None:
            trace.record(name, "fail", f"{type(exc).__name__}: {exc}")
        raise
    else:
        if trace is not None:
            trace.record(name, "pass")

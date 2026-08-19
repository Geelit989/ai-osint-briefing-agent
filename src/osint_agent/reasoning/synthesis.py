"""Evidence-only structured synthesis using the configured local Ollama model."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from typing import Any, Protocol

import requests
from pydantic import ValidationError

from osint_agent.config import settings
from osint_agent.models.brief import (
    BriefSuccess,
    GeneratedBrief,
    IntelligenceBrief,
    SourceReference,
)
from osint_agent.models.document import EvidenceChunk


SYSTEM_PROMPT = """You are the analytic synthesis component of Project ARGUS.
Use ONLY the evidence supplied in the current request. Do not introduce external
factual knowledge. Separate reported facts from analytic assessments. Every
substantive factual claim must cite one or more supplied source identifiers.
When evidence conflicts, identify the disagreement rather than resolving it
without support. When the supplied evidence does not support a requested
conclusion, explicitly state the intelligence gap. Do not invent source
identifiers. Use only low, moderate, or high qualitative confidence; retrieval
distance is not analytic confidence. Produce output strictly conforming to the
provided ARGUS intelligence brief schema. Do not reveal chain-of-thought."""


class ReasoningFailure(RuntimeError):
    """Base controlled failure for the reasoning boundary."""


class ModelInvocationFailure(ReasoningFailure):
    """The configured model provider could not return a response."""


class StructuredOutputFailure(ReasoningFailure):
    """The model response did not conform to the required schema."""


class CitationValidationFailure(ReasoningFailure):
    """The model cited a source identifier that was not supplied."""


class StructuredReasoningModel(Protocol):
    def generate(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any]
    ) -> Any: ...


class OllamaReasoningModel:
    """Minimal structured-output adapter for the configured local runtime."""

    def __init__(self, model: str, host: str, timeout: int) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any]
    ) -> Any:
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": schema,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "options": {"temperature": 0},
                },
                timeout=settings.REASONING_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return json.loads(response.json()["message"]["content"])
        except (requests.RequestException, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ModelInvocationFailure("Ollama model invocation failed") from exc


def build_source_mapping(evidence: list[EvidenceChunk]) -> list[SourceReference]:
    """Map unique documents to stable source IDs, independent of chunk order."""

    grouped: dict[str, list[EvidenceChunk]] = defaultdict(list)
    for chunk in evidence:
        grouped[chunk.doc_id].append(chunk)

    sources = []
    for index, doc_id in enumerate(sorted(grouped), start=1):
        chunks = sorted(grouped[doc_id], key=lambda item: item.chunk_id)
        first = chunks[0]
        sources.append(
            SourceReference(
                source_id=f"S{index}",
                doc_id=doc_id,
                chunk_ids=[chunk.chunk_id for chunk in chunks],
                title=first.title,
                source=first.source,
                provider=first.provider,
                source_type=first.source_type,
                published_date=first.published_date,
                url=first.url,
            )
        )
    return sources


def _build_user_prompt(
    query: str,
    evidence: list[EvidenceChunk],
    sources: list[SourceReference],
) -> str:
    source_by_doc = {source.doc_id: source.source_id for source in sources}
    payload = [
        {
            "source_id": source_by_doc[chunk.doc_id],
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "title": chunk.title,
            "published_date": chunk.published_date,
        }
        for chunk in evidence
    ]
    return (
        f"Analyst query: {query}\n\n"
        "Write a concise unclassified brief ordered as title, BLUF, reported "
        "developments, analytic assessments, and intelligence gaps. Prefix "
        "judgments naturally with 'ARGUS assesses' where appropriate. "
        "Citations are bare identifiers such as S1 (not invented labels).\n\n"
        f"Supplied evidence:\n{json.dumps(payload, indent=2)}"
    )


def _validate_citations(generated: GeneratedBrief, valid_ids: set[str]) -> None:
    cited = {
        *generated.bluf.citations,
        *[
            citation
            for item in [
                *generated.reported_developments,
                *generated.analytic_assessments,
            ]
            for citation in item.citations
        ],
    }

    invalid = cited - valid_ids
    if invalid:
        raise CitationValidationFailure(
            f"Generated brief contained unknown source IDs: {sorted(invalid)}"
        )


def synthesize_brief(
    query: str,
    evidence: list[EvidenceChunk],
    model: StructuredReasoningModel | None = None,
    generated_date: date | None = None,
) -> BriefSuccess:
    """Generate and validate a brief from evidence already approved by the gate."""

    sources = build_source_mapping(evidence)
    if not sources:
        raise ValueError("synthesize_brief requires supplied evidence")
    active_model = model or OllamaReasoningModel(
        settings.REASONING_MODEL,
        settings.OLLAMA_HOST,
        settings.REQUEST_TIMEOUT_SECONDS,
    )
    try:
        raw = active_model.generate(
            SYSTEM_PROMPT,
            _build_user_prompt(query, evidence, sources),
            GeneratedBrief.model_json_schema(),
        )
    except ReasoningFailure:
        raise
    except Exception as exc:
        raise ModelInvocationFailure("Reasoning model invocation failed") from exc

    try:
        generated = GeneratedBrief.model_validate(raw)
    except (ValidationError, TypeError, ValueError) as exc:
        raise StructuredOutputFailure(
            "Reasoning model returned invalid structured output"
        ) from exc

    _validate_citations(generated, {source.source_id for source in sources})
    brief = IntelligenceBrief(
        **generated.model_dump(),
        query=query,
        generated_date=(generated_date or date.today()).isoformat(),
        sources=sources,
    )
    return BriefSuccess(brief=brief)

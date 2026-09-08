from __future__ import annotations

import json

from osint_agent.models.document import EvidenceChunk
from osint_agent.reasoning.synthesis import (
    CitationValidationFailure,
    ModelInvocationFailure,
    StructuredOutputFailure,
    build_source_mapping,
)
from osint_agent.workflow import reason_over_evidence


def evidence(doc_id: str, chunk: int, distance: float = 0.2) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=f"{doc_id}::chunk-{chunk:03}",
        doc_id=doc_id,
        text="The supplied reporting documents a development.",
        title=f"Title {doc_id}",
        source="Fixture News",
        provider="fixture",
        source_type="news",
        published_date="2026-08-15",
        url=f"https://example.test/{doc_id}",
        distance=distance,
    )


def valid_output(
    citations: list[str] | None = None,
    chunk_id: str = "doc-a::chunk-000",
    text: str = "The supplied reporting documents a development.",
) -> dict:
    citations = citations or ["S1"]
    span = {
        "source_id": citations[0],
        "chunk_id": chunk_id,
        "text": text,
    }
    statement = {
        "text": text,
        "citations": citations,
        "supporting_quotes": [span],
    }
    return {
        "title": statement,
        "bluf": statement,
        "reported_developments": [
            statement
        ],
        "analytic_assessments": [
            {
                **statement,
                "confidence": "moderate",
            }
        ],
        "intelligence_gaps": [statement],
    }


class FakeModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0

    def generate(self, system_prompt, user_prompt, schema):
        self.calls += 1
        return self.response


class SupportingModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt, user_prompt, schema):
        self.calls += 1
        payload = json.loads(user_prompt.split("\n\n", 1)[1])
        return {
            "claim_id": payload["claim_id"],
            "status": "supported",
            "issues": [],
        }


def test_insufficient_gate_returns_typed_result_without_model_call():
    model = FakeModel(valid_output())
    result = reason_over_evidence(
        "query", [evidence("doc-a", 0, 0.8)], 0.5, 1, model=model
    )

    assert result.status == "insufficient_evidence"
    assert result.evidence_count == 1
    assert model.calls == 0


def test_zero_results_skip_model_call():
    model = FakeModel(valid_output())
    result = reason_over_evidence("query", [], 0.5, 1, model=model)

    assert result.status == "insufficient_evidence"
    assert result.reason == "no evidence retrieved"
    assert model.calls == 0


def test_sufficient_evidence_generates_typed_traceable_brief():
    model = FakeModel(valid_output())
    result = reason_over_evidence(
        "query",
        [evidence("doc-a", 0)],
        0.5,
        1,
        model=model,
        support_model=SupportingModel(),
    )

    assert result.status == "success"
    assert model.calls == 1
    assert result.brief.sources[0].source_id == "S1"
    assert result.brief.sources[0].doc_id == "doc-a"
    assert result.brief.sources[0].url == "https://example.test/doc-a"
    assert result.brief.bluf.citations == ["S1"]
    assert result.brief.reported_developments[0].citations == ["S1"]
    assert result.claim_support.status == "supported"
    assert len(result.claim_support.judgments) == 5


def test_duplicate_document_chunks_map_to_one_logical_source():
    mapping = build_source_mapping(
        [evidence("doc-b", 0), evidence("doc-a", 1), evidence("doc-a", 0)]
    )

    assert [item.source_id for item in mapping] == ["S1", "S2"]
    assert [item.doc_id for item in mapping] == ["doc-a", "doc-b"]
    assert mapping[0].chunk_ids == [
        "doc-a::chunk-000",
        "doc-a::chunk-001",
    ]


def test_fabricated_citation_is_rejected():
    model = FakeModel(valid_output(["S99"]))

    try:
        reason_over_evidence(
            "query", [evidence("doc-a", 0)], 0.5, 1, model=model
        )
    except CitationValidationFailure as exc:
        assert "S99" in str(exc)
    else:
        raise AssertionError("fabricated citation should fail validation")


def test_invalid_structured_output_is_controlled_failure():
    model = FakeModel({"title": "missing required fields"})

    try:
        reason_over_evidence(
            "query", [evidence("doc-a", 0)], 0.5, 1, model=model
        )
    except StructuredOutputFailure:
        pass
    else:
        raise AssertionError("invalid output should fail validation")


def test_provider_exception_is_controlled_failure():
    class BrokenModel:
        def generate(self, system_prompt, user_prompt, schema):
            raise OSError("provider unavailable")

    try:
        reason_over_evidence(
            "query", [evidence("doc-a", 0)], 0.5, 1, model=BrokenModel()
        )
    except ModelInvocationFailure as exc:
        assert isinstance(exc.__cause__, OSError)
    else:
        raise AssertionError("provider exception should be controlled")


def test_missing_optional_source_metadata_does_not_crash():
    chunk = EvidenceChunk(
        chunk_id="doc-a::chunk-000",
        doc_id="doc-a",
        text="Evidence with sparse metadata.",
        distance=0.2,
    )
    result = reason_over_evidence(
        "query",
        [chunk],
        0.5,
        1,
        model=FakeModel(
            valid_output(chunk_id=chunk.chunk_id, text=chunk.text)
        ),
        support_model=SupportingModel(),
    )

    assert result.status == "success"
    assert result.brief.sources[0].url is None
    assert result.brief.sources[0].title is None


def test_fabricated_bluf_citation_is_rejected():
    output = valid_output()

    output["bluf"]["citations"] = ["S99"]
    model = FakeModel(output)

    try:
        reason_over_evidence(
            "query",
            [evidence("doc-a", 0)],
            0.5,
            1,
            model=model,
        )
    except CitationValidationFailure as exc:
        assert "S99" in str(exc)
    else:
        raise AssertionError(
            "fabricated BLUF citation should fail validation"
        )

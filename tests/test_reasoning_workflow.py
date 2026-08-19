from __future__ import annotations

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
        text=f"Reported evidence from {doc_id}, chunk {chunk}.",
        title=f"Title {doc_id}",
        source="Fixture News",
        provider="fixture",
        source_type="news",
        published_date="2026-08-15",
        url=f"https://example.test/{doc_id}",
        distance=distance,
    )


def valid_output(citations: list[str] | None = None) -> dict:
    citations = citations or ["S1"]
    return {
        "title": "PROJECT ARGUS — FIXTURE INTELLIGENCE BRIEF",
        "bluf": {
            "text": "The supplied reporting indicates a documented development.",
            "citations": citations,
        },
        "reported_developments": [
            {"text": "A development was reported.", "citations": citations}
        ],
        "analytic_assessments": [
            {
                "text": "ARGUS assesses the reporting is consistent.",
                "confidence": "moderate",
                "citations": citations,
            }
        ],
        "intelligence_gaps": [
            "Intent is not established by supplied evidence."
        ],
    }


class FakeModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0

    def generate(self, system_prompt, user_prompt, schema):
        self.calls += 1
        return self.response


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
        "query", [evidence("doc-a", 0)], 0.5, 1, model=model
    )

    assert result.status == "success"
    assert model.calls == 1
    assert result.brief.sources[0].source_id == "S1"
    assert result.brief.sources[0].doc_id == "doc-a"
    assert result.brief.sources[0].url == "https://example.test/doc-a"
    assert result.brief.bluf.citations == ["S1"]
    assert result.brief.reported_developments[0].citations == ["S1"]


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
        "query", [chunk], 0.5, 1, model=FakeModel(valid_output())
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

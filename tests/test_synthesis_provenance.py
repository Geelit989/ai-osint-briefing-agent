from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from osint_agent.models.brief import SynthesisDraft
from osint_agent.models.document import EvidenceChunk
from osint_agent.reasoning.provenance import resolve_provenance
from osint_agent.reasoning.claim_support import ClaimSupportValidationFailure, validate_claim_support
from osint_agent.reasoning.synthesis import (
    synthesize_brief, build_source_mapping, StructuredOutputFailure,
    CitationValidationFailure,
)


def chunk(text="Before. Officials suspected an attack. After.", **changes):
    return EvidenceChunk(**{"chunk_id": "c1", "doc_id": "d1", "text": text,
                            "distance": 0.2, **changes})


def draft():
    claim = {"text": "Officials suspected an attack.", "citations": ["S1"],
             "supporting_quotes": [{"source_id": "S1", "text": "Officials suspected an attack."}]}
    return {"title": deepcopy(claim), "bluf": deepcopy(claim),
            "reported_developments": [deepcopy(claim)],
            "analytic_assessments": [{**deepcopy(claim), "confidence": "low"}],
            "intelligence_gaps": [deepcopy(claim)]}


def execute(raw, chunks):
    support = Mock()
    support.generate.side_effect = AssertionError("semantic validator must not run")
    return synthesize_brief("query", chunks, model=Mock(generate=Mock(return_value=raw)),
                            support_model=support)


def test_unique_exact_quote_resolves_application_offsets():
    item = chunk()
    resolved = resolve_provenance(SynthesisDraft.model_validate(draft()), [item], build_source_mapping([item]))
    span = resolved.bluf.supporting_spans[0]
    assert (span.source_id, span.chunk_id, span.start, span.end) == ("S1", "c1", 8, 38)
    assert item.text[span.start:span.end] == span.text == "Officials suspected an attack."
    assert resolved.bluf.citations == ["S1"]


@pytest.mark.parametrize("field", ["title", "bluf", "reported_developments", "analytic_assessments", "intelligence_gaps"])
@pytest.mark.parametrize("missing", [True, False])
def test_all_substantive_fields_require_nonempty_citations(field, missing):
    raw = draft()
    claim = raw[field] if isinstance(raw[field], dict) else raw[field][0]
    if missing:
        del claim["citations"]
    else:
        claim["citations"] = []
    with pytest.raises(StructuredOutputFailure):
        execute(raw, [chunk()])


@pytest.mark.parametrize("quote", [
    {"source_id": "S99", "text": "Officials suspected an attack."},
    {"source_id": "S1", "chunk_id": "unknown", "text": "Officials suspected an attack."},
    {"source_id": "S1", "text": "Fabricated evidence"},
    {"source_id": "S1", "text": "officials suspected an attack."},
    {"source_id": "S1", "text": "Officials  suspected an attack."},
])
def test_unresolvable_quotes_fail_before_semantics(quote):
    raw = draft()
    raw["bluf"]["supporting_quotes"] = [quote]
    with pytest.raises(ClaimSupportValidationFailure):
        execute(raw, [chunk()])


def test_unknown_citation_fails_before_semantics():
    raw = draft()
    raw["bluf"]["citations"] = ["S99"]
    with pytest.raises(CitationValidationFailure):
        execute(raw, [chunk()])


@pytest.mark.parametrize("missing", [True, False])
def test_support_quotes_are_required_and_nonempty(missing):
    raw = draft()
    if missing:
        del raw["bluf"]["supporting_quotes"]
    else:
        raw["bluf"]["supporting_quotes"] = []
    with pytest.raises(StructuredOutputFailure):
        execute(raw, [chunk()])


def test_inconsistent_source_chunk_pair_cannot_borrow_other_source_text():
    raw = draft()
    raw["bluf"]["supporting_quotes"][0]["chunk_id"] = "c2"
    with pytest.raises(ClaimSupportValidationFailure):
        execute(raw, [chunk(), chunk(doc_id="d2", chunk_id="c2")])


@pytest.mark.parametrize("chunks", [
    [chunk("Officials suspected an attack. Officials suspected an attack.")],
    [chunk(), chunk(chunk_id="c2")],
    [chunk(), chunk()],
])
def test_repeated_or_ambiguous_quotes_fail(chunks):
    with pytest.raises(ClaimSupportValidationFailure):
        execute(draft(), chunks)


def test_chunk_disambiguator_selects_only_declared_chunk():
    items = [chunk(), chunk(chunk_id="c2")]
    raw = draft()
    for claim in [raw["title"], raw["bluf"], *raw["reported_developments"],
                  *raw["analytic_assessments"], *raw["intelligence_gaps"]]:
        claim["supporting_quotes"][0]["chunk_id"] = "c2"
    result = resolve_provenance(SynthesisDraft.model_validate(raw), items, build_source_mapping(items))
    assert result.bluf.supporting_spans[0].chunk_id == "c2"


def test_every_citation_must_contribute_and_quotes_must_be_cited():
    items = [chunk(), chunk("Other evidence.", doc_id="d2", chunk_id="c2")]
    for citations in (["S1", "S2"], ["S2"]):
        raw = draft()
        raw["bluf"]["citations"] = citations
        with pytest.raises(ClaimSupportValidationFailure):
            execute(raw, items)


def test_model_offsets_are_forbidden_and_schema_contains_none():
    schema = SynthesisDraft.model_json_schema()
    assert "start" not in schema["$defs"]["EvidenceQuote"]["properties"]
    assert "end" not in schema["$defs"]["EvidenceQuote"]["properties"]
    raw = draft()
    raw["bluf"]["supporting_quotes"][0].update(start=0, end=74)
    with pytest.raises(StructuredOutputFailure):
        execute(raw, [chunk()])


def test_final_validator_still_rejects_tampered_offsets():
    item = chunk()
    final = resolve_provenance(SynthesisDraft.model_validate(draft()), [item], build_source_mapping([item]))
    final.bluf.supporting_spans[0].start = 0
    support = Mock()
    with pytest.raises(ClaimSupportValidationFailure):
        validate_claim_support(final, [item], build_source_mapping([item]), support,
                               reference_time=datetime.now(timezone.utc))
    support.generate.assert_not_called()


def test_query_repetition_with_lost_qualification_still_requires_semantic_support():
    raw = draft()
    raw["bluf"]["text"] = "Confirmed attacks"
    support = Mock()
    import json
    def decide(system, user, schema):
        claim_id = json.loads(user.split("\n\n", 1)[1])["claim_id"]
        return {"claim_id": claim_id, "status": "unsupported" if claim_id == "bluf" else "supported",
                "issues": ["modality_strengthening"] if claim_id == "bluf" else []}
    support.generate.side_effect = decide
    with pytest.raises(ClaimSupportValidationFailure) as exc:
        synthesize_brief("Confirmed attacks", [chunk()], model=Mock(generate=Mock(return_value=raw)), support_model=support)
    assert any(j.claim_id == "bluf" and j.issues == ["modality_strengthening"] for j in exc.value.report.judgments)

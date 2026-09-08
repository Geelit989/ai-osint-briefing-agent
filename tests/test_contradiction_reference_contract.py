from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import Mock
import json

import pytest

from osint_agent.models.brief import SynthesisDraft
from osint_agent.models.document import EvidenceChunk
from osint_agent.reasoning import synthesis
from osint_agent.reasoning.claim_support import (
    ClaimSupportValidationFailure, validate_claim_support,
)
from osint_agent.reasoning.provenance import resolve_provenance


def inputs(groups=()):
    evidence = [EvidenceChunk(chunk_id="neutral", doc_id="a", text="Reporting exists.", distance=0.2)]
    for index, group in enumerate(groups):
        for position in ("affirmation", "denial"):
            identifier = f"b-{index}-{position}"
            evidence.append(EvidenceChunk(
                chunk_id=identifier, doc_id=identifier, text=f"{group}: {position}.",
                distance=0.2, contradiction_group=group, contradiction_position=position,
            ))
    claim = {"text": "Reporting exists.", "citations": ["S1"],
             "supporting_quotes": [{"source_id": "S1", "text": "Reporting exists."}],
             "acknowledged_contradictions": []}
    raw = {"title": deepcopy(claim), "bluf": deepcopy(claim),
           "reported_developments": [deepcopy(claim)],
           "analytic_assessments": [{**deepcopy(claim), "confidence": "low"}],
           "intelligence_gaps": [deepcopy(claim)]}
    return evidence, raw


def supporting_model():
    model = Mock()
    def generate(system, user, schema):
        payload = json.loads(user.split("\n\n", 1)[1])
        return {"claim_id": payload["claim_id"], "status": "supported", "issues": []}
    model.generate.side_effect = generate
    return model


@pytest.mark.parametrize("groups,acknowledged", [
    ([], []), (["C1"], ["C1"]),
    (["C1", "C2"], ["C2"]),
    (["C1", "C2"], ["C1", "C2"]),
    (["C1"], ["C1", "C1"]),
    (["C1", "C2"], []),
])
def test_valid_runtime_references_survive_to_final_brief(groups, acknowledged):
    evidence, raw = inputs(groups)
    sources = synthesis.build_source_mapping(evidence)
    claim = raw["intelligence_gaps"][0]
    claim["acknowledged_contradictions"] = acknowledged
    for item in evidence[1:]:
        if item.contradiction_group in acknowledged:
            source = next(s for s in sources if s.doc_id == item.doc_id)
            claim["citations"].append(source.source_id)
            claim["supporting_quotes"].append({"source_id": source.source_id, "text": item.text})
    support = supporting_model()
    result = synthesis.synthesize_brief(
        "query", evidence, model=Mock(generate=Mock(return_value=raw)), support_model=support,
    )
    assert result.brief.intelligence_gaps[0].acknowledged_contradictions == acknowledged
    assert support.generate.call_count == 5


@pytest.mark.parametrize("field", ["title", "bluf", "reported_developments", "analytic_assessments", "intelligence_gaps"])
@pytest.mark.parametrize("groups,reference", [([], "S1"), ([], "C1"),
    ([], "anything"),
    (["C1"], "S1"), (["C1", "C2"], "C3")])
def test_invalid_references_reject_before_provenance_or_semantics(monkeypatch, field, groups, reference):
    evidence, raw = inputs(groups)
    claim = raw[field] if isinstance(raw[field], dict) else raw[field][0]
    claim["acknowledged_contradictions"] = [reference]
    before = deepcopy(raw)
    resolve = Mock(side_effect=AssertionError("provenance must not run"))
    monkeypatch.setattr(synthesis, "resolve_provenance", resolve)
    support = supporting_model()
    with pytest.raises(
        synthesis.StructuredOutputFailure,
        match="not present in the supplied contradiction-ID set",
    ):
        synthesis.synthesize_brief("query", evidence, model=Mock(generate=Mock(return_value=raw)), support_model=support)
    resolve.assert_not_called()
    support.generate.assert_not_called()
    assert raw == before


def test_mixed_valid_and_invalid_references_fail_closed(monkeypatch):
    evidence, raw = inputs(["C1", "C2"])
    raw["title"]["acknowledged_contradictions"] = ["C1", "S2"]
    provenance = Mock(side_effect=AssertionError("provenance must not run"))
    monkeypatch.setattr(synthesis, "resolve_provenance", provenance)

    with pytest.raises(
        synthesis.StructuredOutputFailure,
        match=r"supplied contradiction-ID set: \['S2'\]",
    ):
        synthesis.synthesize_brief(
            "query",
            evidence,
            model=Mock(generate=Mock(return_value=raw)),
            support_model=supporting_model(),
        )

    provenance.assert_not_called()


def test_citation_validation_precedes_contradiction_reference_validation(monkeypatch):
    evidence, raw = inputs()
    raw["title"]["citations"] = ["unknown-source"]
    raw["title"]["acknowledged_contradictions"] = ["unknown-contradiction"]
    contradiction_validation = Mock(
        side_effect=AssertionError("contradiction validation must not run")
    )
    provenance = Mock(side_effect=AssertionError("provenance must not run"))
    monkeypatch.setattr(
        synthesis, "_validate_contradiction_references", contradiction_validation
    )
    monkeypatch.setattr(synthesis, "resolve_provenance", provenance)

    with pytest.raises(
        synthesis.CitationValidationFailure,
        match="unknown source IDs",
    ):
        synthesis.synthesize_brief(
            "query",
            evidence,
            model=Mock(generate=Mock(return_value=raw)),
            support_model=supporting_model(),
        )

    contradiction_validation.assert_not_called()
    provenance.assert_not_called()


def test_downstream_unknown_reference_defense_remains_authoritative():
    evidence, raw = inputs()
    sources = synthesis.build_source_mapping(evidence)
    final = resolve_provenance(SynthesisDraft.model_validate(raw), evidence, sources)
    final.intelligence_gaps[0].acknowledged_contradictions = ["S1"]
    support = supporting_model()
    with pytest.raises(ClaimSupportValidationFailure) as exc:
        validate_claim_support(final, evidence, sources, support,
                               reference_time=datetime.now(timezone.utc))
    assert exc.value.report.judgments[0].claim_id == "intelligence_gaps[0]"
    assert "unknown_contradiction_reference" in exc.value.report.judgments[0].issues
    support.generate.assert_not_called()

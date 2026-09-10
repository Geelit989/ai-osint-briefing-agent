"""Verify observation preserves production acceptance and rejection."""

import importlib.util
import json
import logging
from pathlib import Path

import pytest

from osint_agent.models.document import EvidenceChunk
from osint_agent.reasoning.claim_support import ClaimSupportValidationFailure


@pytest.mark.parametrize(
    ("valid_span", "unsupported_claim"),
    [(True, None), (True, "bluf"), (False, None)],
)
def test_diagnostic_preserves_gate_and_restores_observers(
    monkeypatch, capsys, valid_span, unsupported_claim,
):
    path = Path(__file__).resolve().parents[1] / "scripts/claim_support_diagnostic.py"
    spec = importlib.util.spec_from_file_location("diagnostic", path)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    chunk = EvidenceChunk(chunk_id="c1", doc_id="d1", text="Reported event.", distance=0.2)
    monkeypatch.setattr(diagnostic.workflow, "semantic_search", lambda *a, **k: [chunk])
    monkeypatch.setattr("osint_agent.retrieval.evidence_identity.get_documents_by_ids", lambda ids: {})
    claim = {"text": chunk.text, "citations": ["S1"], "supporting_quotes": [{
        "source_id": "S1", "chunk_id": "c1",
        "text": chunk.text if valid_span else "Wrong quotation",
    }]}

    model_calls = []

    def generate(self, system, user, schema):
        model_calls.append(schema["title"])
        if schema["title"] == "SynthesisDraft":
            return {"title": claim, "bluf": claim, "reported_developments": [],
                    "analytic_assessments": [], "intelligence_gaps": []}
        payload = json.loads(user.split("\n\n", 1)[1])
        if payload["claim_id"] == unsupported_claim:
            return {"claim_id": payload["claim_id"], "status": "unsupported",
                    "issues": ["unsupported_claim"]}
        return {"claim_id": payload["claim_id"], "status": "supported", "issues": []}

    monkeypatch.setattr(diagnostic.synthesis.OllamaReasoningModel, "generate", generate)
    original = diagnostic.claim_support._validate_claim_structure
    original_generate = diagnostic.synthesis.OllamaReasoningModel.generate
    if valid_span and unsupported_claim is None:
        assert diagnostic.run("query", 0.6, min_evidence=1).status == "success"
    else:
        with pytest.raises(ClaimSupportValidationFailure):
            diagnostic.run("query", 0.6, min_evidence=1)
    captured = capsys.readouterr().out
    events = [json.loads(line) for line in captured.splitlines()]
    semantic_outputs = [e for e in events if e["event"] == "semantic_output"]
    semantic_decisions = [e for e in events if e["event"] == "semantic_decision"]
    assert len(semantic_outputs) == (2 if valid_span else 0)
    assert len(semantic_decisions) == (2 if valid_span else 0)
    assert all(
        set(event) == {"event", "claim_id", "status", "issues"}
        for event in semantic_decisions
    )
    if unsupported_claim is not None:
        assert next(
            event for event in semantic_decisions
            if event["claim_id"] == unsupported_claim
        ) == {
            "event": "semantic_decision",
            "claim_id": unsupported_claim,
            "status": "unsupported",
            "issues": ["unsupported_claim"],
        }
    assert len([e for e in events if e["event"] == "deterministic"]) == (2 if valid_span else 0)
    assert any(e["event"] == "parsed_draft" for e in events)
    assert any(e["event"] == "parsed_brief" for e in events) == valid_span
    assert events[-1]["event"] == (
        "result" if valid_span and unsupported_claim is None else "failure"
    )
    assert model_calls == (
        ["SynthesisDraft", "SemanticSupportDecision", "SemanticSupportDecision"]
        if valid_span else ["SynthesisDraft"]
    )
    assert chunk.text not in captured
    assert not any("user_prompt" in event for event in events)
    assert not any("system_prompt" in event for event in events)
    assert diagnostic.claim_support._validate_claim_structure is original
    assert diagnostic.synthesis.OllamaReasoningModel.generate is original_generate


def test_opt_in_semantic_content_aligns_claim_evidence_and_decision(
    monkeypatch, capsys, caplog,
):
    path = Path(__file__).resolve().parents[1] / "scripts/claim_support_diagnostic.py"
    spec = importlib.util.spec_from_file_location("diagnostic_content", path)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    chunk = EvidenceChunk(
        chunk_id="c-sensitive",
        doc_id="d-sensitive",
        text="Prefix. Exact support. Suffix.",
        distance=0.2,
    )
    monkeypatch.setattr(
        diagnostic.workflow, "semantic_search", lambda *args, **kwargs: [chunk]
    )
    monkeypatch.setattr(
        "osint_agent.retrieval.evidence_identity.get_documents_by_ids",
        lambda ids: {},
    )
    claim = {
        "text": "A generated claim.",
        "citations": ["S1"],
        "supporting_quotes": [
            {
                "source_id": "S1",
                "chunk_id": chunk.chunk_id,
                "text": "Exact support.",
            }
        ],
    }
    model_calls = []

    def generate(self, system, user, schema):
        model_calls.append(schema["title"])
        if schema["title"] == "SynthesisDraft":
            return {
                "title": claim,
                "bluf": claim,
                "reported_developments": [],
                "analytic_assessments": [],
                "intelligence_gaps": [],
            }
        payload = json.loads(user.split("\n\n", 1)[1])
        return {
            "claim_id": payload["claim_id"],
            "status": "unsupported",
            "issues": ["unsupported_claim"],
        }

    monkeypatch.setattr(
        diagnostic.synthesis.OllamaReasoningModel, "generate", generate
    )
    original_structure = diagnostic.claim_support._validate_claim_structure
    original_generate = diagnostic.synthesis.OllamaReasoningModel.generate

    with caplog.at_level(logging.INFO):
        with pytest.raises(ClaimSupportValidationFailure):
            diagnostic.run(
                "query",
                0.6,
                min_evidence=1,
                show_semantic_content=True,
            )

    captured = capsys.readouterr().out
    assert captured.count("=== ARGUS SEMANTIC CONTENT BEGIN ===") == 2
    assert "CLAIM: title" in captured
    assert "CLAIM: bluf" in captured
    assert "Generated claim:\nA generated claim." in captured
    assert "Citations:\n- S1" in captured
    assert "Referenced chunk IDs:\n- c-sensitive" in captured
    assert "Supporting span 1: [8:22]" in captured
    assert "Exact chunk-slice match: true" in captured
    assert "Supporting span text:\nExact support." in captured
    assert (
        "Complete evidence chunk text supplied to validator:\n" + chunk.text
        in captured
    )
    assert "status: unsupported\nissues:\n- unsupported_claim" in captured
    assert "complete evidence chunks with supporting-span offsets" in captured
    assert "all cited sources represented: true" in captured
    assert diagnostic.claim_support.SUPPORT_SYSTEM_PROMPT not in captured
    assert "ARGUS_CLAIM_SUPPORT_INPUT" not in captured
    assert chunk.text not in caplog.text
    assert model_calls == [
        "SynthesisDraft",
        "SemanticSupportDecision",
        "SemanticSupportDecision",
    ]
    assert diagnostic.claim_support._validate_claim_structure is original_structure
    assert diagnostic.synthesis.OllamaReasoningModel.generate is original_generate

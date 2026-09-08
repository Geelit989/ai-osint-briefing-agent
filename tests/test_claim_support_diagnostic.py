"""Verify observation preserves production acceptance and rejection."""

import importlib.util
import json
from pathlib import Path

import pytest

from osint_agent.models.document import EvidenceChunk
from osint_agent.reasoning.claim_support import ClaimSupportValidationFailure


@pytest.mark.parametrize("valid_span", [True, False])
def test_diagnostic_preserves_gate_and_restores_observers(monkeypatch, capsys, valid_span):
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

    def generate(self, system, user, schema):
        if schema["title"] == "SynthesisDraft":
            return {"title": claim, "bluf": claim, "reported_developments": [],
                    "analytic_assessments": [], "intelligence_gaps": []}
        payload = json.loads(user.split("\n\n", 1)[1])
        return {"claim_id": payload["claim_id"], "status": "supported", "issues": []}

    monkeypatch.setattr(diagnostic.synthesis.OllamaReasoningModel, "generate", generate)
    original = diagnostic.claim_support._validate_claim_structure
    if valid_span:
        assert diagnostic.run("query", 0.6, min_evidence=1).status == "success"
    else:
        with pytest.raises(ClaimSupportValidationFailure):
            diagnostic.run("query", 0.6, min_evidence=1)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len([e for e in events if e["event"] == "semantic_output"]) == (2 if valid_span else 0)
    assert len([e for e in events if e["event"] == "deterministic"]) == (2 if valid_span else 0)
    assert any(e["event"] == "parsed_draft" for e in events)
    assert any(e["event"] == "parsed_brief" for e in events) == valid_span
    assert events[-1]["event"] == ("result" if valid_span else "failure")
    assert diagnostic.claim_support._validate_claim_structure is original

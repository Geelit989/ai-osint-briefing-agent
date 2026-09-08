"""The workspace observes the production fail-closed path without live models."""

import json
import shlex

import pytest

from osint_agent import workflow
from osint_agent.models.document import EvidenceChunk
from osint_agent.models.workflow import WorkflowTrace
from osint_agent.reasoning import synthesis
from osint_agent.reasoning.synthesis import CitationValidationFailure
from osint_agent.workspace.runs import cli_equivalent, execute_run


def evidence(doc_id, chunk=0, distance=0.2):
    return EvidenceChunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}::chunk-{chunk}",
        text=f"Source {doc_id} reported a development in passage {chunk}.",
        distance=distance,
    )


class Model:
    def __init__(self, failure=None):
        self.failure = failure
        self.synthesis_calls = 0
        self.support_calls = 0

    def generate(self, system_prompt, user_prompt, schema):
        request = json.loads(user_prompt.split("\n\n", 1)[1])
        if schema["title"] == "SemanticSupportDecision":
            self.support_calls += 1
            if self.failure == "validator":
                raise OSError("local model unavailable")
            return {
                "claim_id": request["claim_id"],
                "status": "unsupported" if self.failure == "support" else "supported",
                "issues": ["unsupported_claim"] if self.failure == "support" else [],
            }
        self.synthesis_calls += 1
        if self.failure == "model":
            raise OSError("local model unavailable")
        if self.failure == "schema":
            return {"title": "invalid draft"}
        chunk = request["evidence"][0]
        claim = {
            "text": chunk["text"],
            "citations": ["S99"] if self.failure == "citation" else [chunk["source_id"]],
            "supporting_quotes": [{
                "source_id": chunk["source_id"],
                "chunk_id": chunk["chunk_id"],
                "text": "invented quote" if self.failure == "provenance" else chunk["text"],
            }],
            "acknowledged_contradictions": ["unknown"] if self.failure == "contradiction" else [],
        }
        return {
            "title": claim,
            "bluf": claim,
            "reported_developments": [claim],
            "analytic_assessments": [{**claim, "confidence": "moderate"}],
            "intelligence_gaps": [],
        }


@pytest.fixture
def configure(monkeypatch):
    # Identity accounting remains real, with isolated missing stored metadata.
    monkeypatch.setattr(
        "osint_agent.retrieval.evidence_identity.get_documents_by_ids", lambda ids: {},
    )

    def apply(chunks=None, failure=None):
        model = Model(failure)
        monkeypatch.setattr(
            workflow, "semantic_search",
            lambda query, n_results: chunks if chunks is not None else [evidence("b"), evidence("c")],
        )
        monkeypatch.setattr(synthesis, "OllamaReasoningModel", lambda *args: model)
        return model

    return apply


def run():
    return execute_run("A focused analyst question", 5, "test-run", "2026-09-08T12:00:00Z")


def stages(result):
    return {item["name"]: item["status"] for item in result["validation"]["stages"]}


def test_accepted_run_preserves_metrics_sources_and_exact_claim_links(configure):
    model = configure([evidence("a", distance=0.9), evidence("b"), evidence("b", 1), evidence("c")])
    result = run()

    assert result["status"] == "success"
    assert result["retrieval"] == {
        "retrieved_chunk_count": 4,
        "usable_chunk_count": 3,
        "independent_evidence_count": 2,
    }
    assert [(s["source_id"], s["doc_id"]) for s in result["sources"]] == [("S1", "b"), ("S2", "c")]
    assert len(result["evidence"]) == 4
    assert len(result["grouping_manifest"]) == 2
    assert set(stages(result).values()) == {"pass"}
    assert result["brief"]["bluf"]["citations"] == ["S1"]
    assert result["claims"][1]["supporting_spans"] == result["brief"]["bluf"]["supporting_spans"]
    assert result["claims"][1]["status"] == "supported"
    assert result["claims"][3]["confidence"] == "moderate"
    assert result["failure_detail"] is None
    assert model.synthesis_calls == 1
    assert model.support_calls == 4


@pytest.mark.parametrize("chunks,expected", [([], (0, 0, 0)), ([evidence("a", distance=0.9)], (1, 0, 0)), ([evidence("b"), evidence("b", 1)], (2, 2, 1))])
def test_insufficient_run_keeps_evidence_and_never_invokes_model(configure, chunks, expected):
    model = configure(chunks)
    result = run()

    assert result["status"] == "insufficient_evidence"
    assert tuple(result["retrieval"].values()) == expected
    assert result["sufficiency"]["status"] == "INSUFFICIENT"
    assert len(result["evidence"]) == len(chunks)
    assert result["brief"] is None
    assert result["claims"] == []
    assert result["failure_detail"] is None
    assert stages(result)["evidence_assessment"] == "pass"
    assert stages(result)["sufficiency"] == "fail"
    assert stages(result)["reasoning"] == "not_run"
    assert stages(result)["claim_support"] == "not_run"
    assert model.synthesis_calls == model.support_calls == 0


@pytest.mark.parametrize("failure,stage,next_stage", [
    ("schema", "structured_output", "citation_validation"),
    ("citation", "citation_validation", "contradiction_references"),
    ("contradiction", "contradiction_references", "provenance"),
    ("provenance", "provenance", "claim_support"),
    ("support", "claim_support", None),
])
def test_validation_failure_has_trace_but_no_accepted_draft(configure, failure, stage, next_stage):
    model = configure(failure=failure)
    result = run()

    assert result["status"] == "validation_failure"
    assert result["brief"] is None
    assert result["claims"] == []
    assert len(result["sources"]) == 2
    assert len(result["evidence"]) == 2
    assert result["sufficiency"]["status"] == "SUFFICIENT"
    assert stages(result)[stage] == "fail"
    if next_stage:
        assert stages(result)[next_stage] == "not_run"
    if failure in {"provenance", "support"}:
        assert result["validation"]["claim_support"]["status"] == "unsupported"
    else:
        assert result["validation"]["claim_support"] is None
    if failure != "support":
        assert model.support_calls == 0


@pytest.mark.parametrize("failure,stage", [("model", "reasoning"), ("validator", "claim_support")])
def test_model_and_validator_infrastructure_failures_remain_system_errors(configure, failure, stage):
    configure(failure=failure)
    result = run()

    assert result["status"] == "error"
    assert result["brief"] is None
    assert result["claims"] == []
    assert stages(result)[stage] == "fail"
    assert result["validation"]["claim_support"] is None


def test_retrieval_failure_preserves_unknown_metrics(configure, monkeypatch):
    configure()

    def fail(query, n_results):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(workflow, "semantic_search", fail)
    result = run()
    assert result["status"] == "error"
    assert result["retrieval"] is None
    assert result["sufficiency"] is None
    assert stages(result)["retrieval"] == "fail"
    assert stages(result)["evidence_assessment"] == "not_run"


def test_trace_does_not_change_existing_exception_contract(configure):
    configure(failure="citation")
    trace = WorkflowTrace()
    with pytest.raises(CitationValidationFailure):
        workflow.generate_brief_for_query("question", 0.5, 2, trace=trace)
    assert trace.assessment.sufficient
    assert trace.stages[5].status == "fail"


def test_cli_equivalent_uses_literal_query_and_real_flags():
    query = "analyst's question; $(touch /tmp/not-run) `commands`\nmore data"
    assert shlex.split(cli_equivalent(query, 7)) == [
        "python", "scripts/reasoning_smoke.py", query,
        "--max-distance", "0.5", "--min-evidence", "2", "--results", "7",
    ]

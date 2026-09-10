from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest
import requests

from osint_agent.config import settings
from osint_agent.models.brief import GeneratedBrief, SemanticSupportDecision, SynthesisDraft
from osint_agent.models.document import EvidenceChunk
from osint_agent.reasoning.claim_support import (
    ClaimSupportValidatorFailure,
    SUPPORT_SYSTEM_PROMPT,
    _build_validator_prompt,
    validate_claim_support,
)
from osint_agent.reasoning.synthesis import (
    ModelInvocationFailure,
    OllamaReasoningModel,
    SYSTEM_PROMPT,
    build_source_mapping,
)


class FakeResponse:
    def __init__(self, content: dict) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "message": {"content": json.dumps(self.content)},
            "prompt_eval_count": 42,
            "eval_count": 12,
            "done_reason": "stop",
        }


def validator_fixture() -> tuple[EvidenceChunk, GeneratedBrief]:
    chunk = EvidenceChunk(
        chunk_id="doc-a::chunk-000",
        doc_id="doc-a",
        text="Sensitive complete evidence text must not be logged.",
        distance=0.2,
    )
    statement = {
        "text": "The reported event occurred.",
        "citations": ["S1"],
        "supporting_spans": [
            {
                "source_id": "S1",
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "start": 0,
                "end": len(chunk.text),
            }
        ],
    }
    generated = GeneratedBrief.model_validate(
        {
            "title": statement,
            "bluf": statement,
            "reported_developments": [],
            "analytic_assessments": [],
            "intelligence_gaps": [],
        }
    )
    return chunk, generated


def test_claim_support_request_is_structured_bounded_and_safely_instrumented(
    monkeypatch, caplog,
):
    chunk, generated = validator_fixture()
    sources = build_source_mapping([chunk])
    user_prompt = _build_validator_prompt(
        "title",
        generated.title,
        {chunk.chunk_id: chunk},
        [],
        datetime.now(timezone.utc),
    )
    schema = SemanticSupportDecision.model_json_schema()
    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(
            {"claim_id": "title", "status": "supported", "issues": []}
        )

    monkeypatch.setattr("osint_agent.reasoning.synthesis.requests.post", post)
    model = OllamaReasoningModel("llama3.2", "http://localhost:11434", 300)

    with caplog.at_level(logging.INFO, logger="osint_agent.reasoning.synthesis"):
        result = model.generate(SUPPORT_SYSTEM_PROMPT, user_prompt, schema)

    assert result["status"] == "supported"
    assert captured["json"]["format"] == schema
    assert captured["json"]["options"] == {
        "temperature": 0,
        "num_predict": settings.CLAIM_SUPPORT_NUM_PREDICT,
    }
    assert captured["timeout"] == 300
    log_text = caplog.text
    assert "claim_support_validation" in log_text
    assert '"claim_id": "title"' in log_text
    assert '"success": true' in log_text
    assert chunk.text not in log_text


def test_claim_support_schema_bounds_issue_generation():
    schema = SemanticSupportDecision.model_json_schema()

    assert schema["properties"]["issues"]["maxItems"] == 11


def test_synthesis_request_does_not_receive_validator_generation_bound(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse({})

    monkeypatch.setattr("osint_agent.reasoning.synthesis.requests.post", post)
    model = OllamaReasoningModel("llama3.2", "http://localhost:11434", 300)
    user_prompt = 'ARGUS_SYNTHESIS_INPUT\n\n{"evidence": []}'

    model.generate(SYSTEM_PROMPT, user_prompt, SynthesisDraft.model_json_schema())

    assert captured["json"]["options"] == {"temperature": 0}


def test_claim_support_model_timeout_remains_fail_closed_and_is_instrumented(
    monkeypatch, caplog,
):
    chunk, generated = validator_fixture()

    def timeout(*args, **kwargs):
        raise requests.ReadTimeout("timed out")

    monkeypatch.setattr("osint_agent.reasoning.synthesis.requests.post", timeout)
    model = OllamaReasoningModel("llama3.2", "http://localhost:11434", 300)

    with caplog.at_level(logging.ERROR, logger="osint_agent.reasoning.synthesis"):
        with pytest.raises(ClaimSupportValidatorFailure) as exc_info:
            validate_claim_support(
                generated,
                [chunk],
                build_source_mapping([chunk]),
                model,
                reference_time=datetime.now(timezone.utc),
            )

    assert isinstance(exc_info.value.__cause__, ModelInvocationFailure)
    assert isinstance(exc_info.value.__cause__.__cause__, requests.ReadTimeout)
    assert '"success": false' in caplog.text
    assert '"exception_type": "ReadTimeout"' in caplog.text
    assert chunk.text not in caplog.text

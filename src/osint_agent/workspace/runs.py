"""Serialize an observed ARGUS execution for the local analyst workspace.

This adapter calls the existing application workflow. Retrieval, acceptance,
source identity, and claim judgments remain decisions of the ARGUS core.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any

from osint_agent.models.brief import BriefSuccess, ClaimSupportReport
from osint_agent.models.workflow import WorkflowTrace
from osint_agent.reasoning.claim_support import (
    ClaimSupportValidationFailure,
    ClaimSupportValidatorFailure,
    iter_substantive_claims,
)
from osint_agent.reasoning.synthesis import (
    CitationValidationFailure,
    StructuredOutputFailure,
    build_source_mapping,
)
from osint_agent.workflow import generate_brief_for_query


logger = logging.getLogger(__name__)

# Match the existing scripts/reasoning_smoke.py defaults.
MAX_DISTANCE = 0.5
MIN_EVIDENCE = 2


def cli_equivalent(query: str, n_results: int) -> str:
    """Return display-only shell syntax using actual repository CLI flags."""

    return shlex.join([
        "python", "scripts/reasoning_smoke.py", query,
        "--max-distance", str(MAX_DISTANCE),
        "--min-evidence", str(MIN_EVIDENCE),
        "--results", str(n_results),
    ])


def _claims(result: BriefSuccess) -> list[dict[str, Any]]:
    judgments = {item.claim_id: item for item in result.claim_support.judgments}
    claims = []
    for claim_id, statement in iter_substantive_claims(result.brief):
        judgment = judgments[claim_id]
        claims.append({
            "claim_id": claim_id,
            **statement.model_dump(mode="json"),
            "status": judgment.status,
            "issues": judgment.issues,
            "rationale": judgment.rationale,
        })
    return claims


def execute_run(
    query: str, n_results: int, run_id: str, created_at: str,
) -> dict[str, Any]:
    """Run once, preserving refusals and rejections as distinct audit outcomes.

Only an accepted BriefSuccess can populate the brief and claims fields.
Rejected claim text is confined to the existing validation report, which is
diagnostic information and never a replacement intelligence product.
"""

    trace = WorkflowTrace()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "query": query,
        "created_at": created_at,
        "status": "error",
        "n_results": n_results,
        "retrieval": None,
        "sufficiency": None,
        "brief": None,
        "claims": [],
        "sources": [],
        "evidence": [],
        "grouping_manifest": [],
        "known_contradictions": [],
        "validation": {"stages": [], "claim_support": None},
        "failure_detail": None,
        "cli_equivalent": cli_equivalent(query, n_results),
    }
    report: ClaimSupportReport | None = None
    try:
        result = generate_brief_for_query(
            query,
            max_distance=MAX_DISTANCE,
            min_evidence=MIN_EVIDENCE,
            n_results=n_results,
            trace=trace,
        )
        if isinstance(result, BriefSuccess):
            # Prepare the whole accepted representation before exposing it.
            claims = _claims(result)
            brief = result.brief.model_dump(mode="json")
            payload.update(status="success", brief=brief, claims=claims)
            report = result.claim_support
            trace.sources = result.brief.sources
            trace.known_contradictions = result.brief.known_contradictions
        else:
            payload["status"] = "insufficient_evidence"
            if trace.assessment is not None:
                # Aliases use the same supplied evidence mapping as synthesis.
                # Retrieved-but-unusable chunks remain separately inspectable.
                trace.sources = build_source_mapping(trace.assessment.usable_evidence)
    except (
        CitationValidationFailure,
        StructuredOutputFailure,
        ClaimSupportValidationFailure,
    ) as exc:
        payload["status"] = "validation_failure"
        if isinstance(exc, ClaimSupportValidationFailure):
            report = exc.report
        payload["failure_detail"] = {
            "type": type(exc).__name__,
            "message": "ARGUS rejected the generated draft. No brief was accepted.",
            "technical_detail": str(exc),
        }
        logger.info("ARGUS run %s rejected: %s", run_id, type(exc).__name__)
    except Exception as exc:
        payload["status"] = "error"
        message = (
            "The claim-support validator could not complete a trustworthy "
            "determination. No brief was accepted."
            if isinstance(exc, ClaimSupportValidatorFailure)
            else "ARGUS could not complete this run. Review diagnostics and local runtime readiness."
        )
        payload["failure_detail"] = {
            "type": type(exc).__name__,
            "message": message,
            "technical_detail": str(exc),
        }
        logger.exception("ARGUS run %s failed", run_id)

    assessment = trace.assessment
    if assessment is not None:
        payload["retrieval"] = {
            "retrieved_chunk_count": assessment.retrieved_chunk_count,
            "usable_chunk_count": assessment.usable_chunk_count,
            "independent_evidence_count": assessment.independent_evidence_count,
        }
        payload["sufficiency"] = {
            "status": "SUFFICIENT" if assessment.sufficient else "INSUFFICIENT",
            "reason": assessment.reason,
        }
        payload["grouping_manifest"] = [
            item.model_dump(mode="json") for item in assessment.grouping_manifest
        ]
    payload["evidence"] = [item.model_dump(mode="json") for item in trace.evidence]
    payload["sources"] = [item.model_dump(mode="json") for item in trace.sources]
    payload["known_contradictions"] = [
        item.model_dump(mode="json") for item in trace.known_contradictions
    ]
    payload["validation"] = {
        "stages": [item.model_dump(mode="json") for item in trace.stages],
        "claim_support": report.model_dump(mode="json") if report is not None else None,
    }
    return payload

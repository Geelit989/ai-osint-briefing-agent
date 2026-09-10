"""Opt-in observation of the production reasoning path; failures remain failures."""

import argparse
import json
import logging
import platform
import time
from unittest.mock import patch

from osint_agent import workflow
from osint_agent.config import settings
from osint_agent.models.brief import SemanticSupportDecision
from osint_agent.reasoning import claim_support, synthesis


def emit(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def emit_semantic_content(claim, payload, decision):
    """Print opt-in local content for human entailment inspection."""

    evidence = payload.get("evidence", [])
    expected_spans = {
        (span.source_id, span.chunk_id, span.start, span.end): span.text
        for span in claim.supporting_spans
    }
    represented_sources = {
        item.get("source_id") for item in evidence if isinstance(item, dict)
    }

    print("=== ARGUS SEMANTIC CONTENT BEGIN ===")
    print(f"CLAIM: {decision.claim_id}\n")
    print("Generated claim:")
    print(claim.text)
    print("\nCitations:")
    for citation in claim.citations:
        print(f"- {citation}")
    print("\nReferenced chunk IDs:")
    for chunk_id in dict.fromkeys(
        item.get("chunk_id") for item in evidence if isinstance(item, dict)
    ):
        print(f"- {chunk_id}")
    print("\nValidator evidence delivery:")
    print("- complete evidence chunks with supporting-span offsets")
    print(
        "- all cited sources represented: "
        + str(set(claim.citations).issubset(represented_sources)).lower()
    )

    for index, item in enumerate(evidence, start=1):
        print(f"\nEVIDENCE UNIT {index}")
        print(f"Source ID: {item.get('source_id')}")
        print(f"Chunk ID: {item.get('chunk_id')}")
        chunk_text = item.get("text", "")
        for span_index, span in enumerate(item.get("supporting_spans", []), start=1):
            start = span.get("start")
            end = span.get("end")
            span_text = (
                chunk_text[start:end]
                if isinstance(start, int) and isinstance(end, int)
                else ""
            )
            expected = expected_spans.get(
                (item.get("source_id"), item.get("chunk_id"), start, end)
            )
            print(f"\nSupporting span {span_index}: [{start}:{end}]")
            print(
                "Exact chunk-slice match: "
                + str(expected is not None and span_text == expected).lower()
            )
            print("Supporting span text:")
            print(span_text)
        print("\nComplete evidence chunk text supplied to validator:")
        print(chunk_text)

    print("\nSemantic decision:")
    print(f"status: {decision.status}")
    print("issues:")
    if decision.issues:
        for issue in decision.issues:
            print(f"- {issue}")
    else:
        print("- (none)")
    print("=== ARGUS SEMANTIC CONTENT END ===", flush=True)


def run(
    query,
    max_distance,
    min_evidence=2,
    n_results=5,
    *,
    show_semantic_content=False,
):
    """Delegate every intercepted call unchanged and emit normal program state."""
    search = workflow.semantic_search
    assess = workflow.check_retrieval_sufficiency
    citations = synthesis._validate_citations
    structure = claim_support._validate_claim_structure
    generate = synthesis.OllamaReasoningModel.generate
    resolve = synthesis.resolve_provenance
    claims_by_id = {}

    def observe_search(*args, **kwargs):
        evidence = search(*args, **kwargs)
        emit("retrieval", chunks=[
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
             "text_characters": len(chunk.text), "distance": chunk.distance,
             "passed_max_distance": chunk.distance <= max_distance}
            for chunk in evidence
        ])
        return evidence

    def observe_assessment(*args, **kwargs):
        result = assess(*args, **kwargs)
        emit("sufficiency", sufficient=result.sufficient, reason=result.reason,
             retrieved_chunk_count=result.retrieved_chunk_count,
             usable_chunk_count=result.usable_chunk_count,
             independent_evidence_count=result.independent_evidence_count)
        return result

    def observe_citations(generated, valid_ids):
        claims = list(claim_support.iter_substantive_claims(generated))
        emit("parsed_draft", claim_ids=[item[0] for item in claims],
             claim_count=len(claims), valid_source_ids=sorted(valid_ids))
        return citations(generated, valid_ids)

    def observe_resolution(*args, **kwargs):
        result = resolve(*args, **kwargs)
        claims = list(claim_support.iter_substantive_claims(result))
        emit("parsed_brief", claim_ids=[item[0] for item in claims],
             claim_count=len(claims))
        return result

    def observe_structure(claim_id, claim, evidence, duplicates, sources, conflicts):
        if show_semantic_content:
            claims_by_id[claim_id] = claim
        result = structure(claim_id, claim, evidence, duplicates, sources, conflicts)
        spans = []
        for span in claim.supporting_spans:
            chunk = evidence.get(span.chunk_id)
            source = sources.get(span.source_id)
            association = bool(
                chunk and source and span.chunk_id not in duplicates
                and span.chunk_id in source.chunk_ids and source.doc_id == chunk.doc_id
            )
            spans.append({
                "source_id": span.source_id, "chunk_id": span.chunk_id,
                "start": span.start, "end": span.end,
                "quoted_text_characters": len(span.text),
                "association_resolves": association,
                "actual_slice_characters": (
                    len(chunk.text[span.start:span.end]) if chunk else None
                ),
                "exact_match": bool(association and span.end <= len(chunk.text)
                                    and chunk.text[span.start:span.end] == span.text),
            })
        emit("deterministic", claim_id=claim_id,
             claim_text_characters=len(claim.text), citations=claim.citations,
             has_citations=bool(claim.citations),
             citation_check_note="Empty citation sets satisfy subset checks vacuously; the production uncited_claim check rejects them.",
             citations_resolve=all(item in sources for item in claim.citations),
             spans=spans,
             every_citation_has_span=set(claim.citations).issubset(
                 {span.source_id for span in claim.supporting_spans}),
             every_citation_has_valid_span=set(claim.citations).issubset(
                 {span["source_id"] for span in spans if span["exact_match"]}),
             judgment=(
                 {"status": result.status, "issues": result.issues}
                 if result else None
             ),
             semantic_checks="Modality, polarity, attribution and temporal meaning are model judgments, not deterministic checks.")
        return result

    def observe_generate(self, system_prompt, user_prompt, schema):
        stage = "semantic" if schema.get("title") == "SemanticSupportDecision" else "synthesis"
        schema_text = json.dumps(schema, ensure_ascii=False)
        payload = json.loads(user_prompt.split("\n\n", 1)[1])
        evidence = payload.get("evidence", [])
        evidence_value_occurrences = [
            user_prompt.count(json.dumps(item.get("text", ""), ensure_ascii=False))
            for item in evidence
        ]
        metadata = {
            "claim_id": payload.get("claim_id"),
            "evidence_items": len(evidence),
            "evidence_text_characters": sum(
                len(item.get("text", "")) for item in evidence
            ),
            "evidence_values_repeated": any(
                count > 1 for count in evidence_value_occurrences
            ),
            "supporting_span_text_characters": sum(
                len(span.get("text", ""))
                for item in evidence
                for span in item.get("supporting_spans", [])
            ),
            "system_prompt_characters": len(system_prompt),
            "user_prompt_characters": len(user_prompt),
            "combined_prompt_characters": len(system_prompt) + len(user_prompt),
            "schema_characters": len(schema_text),
            "total_request_text_characters": (
                len(system_prompt) + len(user_prompt) + len(schema_text)
            ),
        }
        emit(stage + "_input", **metadata)
        started = time.perf_counter()
        try:
            result = generate(self, system_prompt, user_prompt, schema)
        except Exception as exc:
            emit(stage + "_failure", **metadata,
                 elapsed_seconds=round(time.perf_counter() - started, 6),
                 exception_type=type(exc).__name__)
            raise
        # The adapter returns message.content only, never Ollama's thinking field.
        emit(stage + "_output", **metadata,
             elapsed_seconds=round(time.perf_counter() - started, 6),
             response_characters=len(json.dumps(result, ensure_ascii=False)),
             structured_output_returned=isinstance(result, dict))
        if stage == "semantic":
            try:
                decision = SemanticSupportDecision.model_validate(result)
            except Exception:
                # Production parsing remains authoritative and fail-closed.
                pass
            else:
                emit("semantic_decision", claim_id=decision.claim_id,
                     status=decision.status, issues=decision.issues)
                if show_semantic_content and decision.claim_id in claims_by_id:
                    emit_semantic_content(
                        claims_by_id[decision.claim_id], payload, decision
                    )
        return result

    emit("environment", python=platform.python_version(), machine=platform.machine(),
         platform=platform.platform(), reasoning_model=settings.REASONING_MODEL,
         embedding_model=settings.EMBEDDING_MODEL, ollama_host=settings.OLLAMA_HOST,
         reasoning_timeout=settings.REASONING_TIMEOUT_SECONDS,
         query=query, max_distance=max_distance, min_evidence=min_evidence,
         n_results=n_results)
    try:
        with patch.object(workflow, "semantic_search", observe_search), \
             patch.object(workflow, "check_retrieval_sufficiency", observe_assessment), \
             patch.object(synthesis, "_validate_citations", observe_citations), \
             patch.object(synthesis, "resolve_provenance", observe_resolution), \
             patch.object(claim_support, "_validate_claim_structure", observe_structure), \
             patch.object(synthesis.OllamaReasoningModel, "generate", observe_generate):
            result = workflow.generate_brief_for_query(
                query, max_distance=max_distance, min_evidence=min_evidence,
                n_results=n_results,
            )
        emit("result", status=result.status)
        return result
    except Exception as exc:
        report = getattr(exc, "report", None)
        emit("failure", exception=type(exc).__name__, message=str(exc),
             failed_claim_ids=(
                 [item.claim_id for item in report.judgments] if report else []
             ))
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--max-distance", type=float, required=True)
    parser.add_argument("--min-evidence", type=int, default=2)
    parser.add_argument("--results", type=int, default=5)
    parser.add_argument(
        "--show-semantic-content",
        action="store_true",
        help="Print full local claim/evidence content for entailment inspection.",
    )
    args = parser.parse_args()
    run(
        args.query,
        args.max_distance,
        args.min_evidence,
        args.results,
        show_semantic_content=args.show_semantic_content,
    )

"""Opt-in observation of the production reasoning path; failures remain failures."""

import argparse
import json
import platform
from unittest.mock import patch

from osint_agent import workflow
from osint_agent.config import settings
from osint_agent.reasoning import claim_support, synthesis


def emit(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def run(query, max_distance, min_evidence=2, n_results=5):
    """Delegate every intercepted call unchanged and emit normal program state."""
    search = workflow.semantic_search
    assess = workflow.check_retrieval_sufficiency
    citations = synthesis._validate_citations
    structure = claim_support._validate_claim_structure
    generate = synthesis.OllamaReasoningModel.generate
    resolve = synthesis.resolve_provenance

    def observe_search(*args, **kwargs):
        evidence = search(*args, **kwargs)
        emit("retrieval", chunks=[
            {**chunk.model_dump(mode="json"),
             "passed_max_distance": chunk.distance <= max_distance}
            for chunk in evidence
        ])
        return evidence

    def observe_assessment(*args, **kwargs):
        result = assess(*args, **kwargs)
        emit("sufficiency", assessment=result.model_dump(mode="json"))
        return result

    def observe_citations(generated, valid_ids):
        emit("parsed_draft", brief=generated.model_dump(mode="json"),
             valid_source_ids=sorted(valid_ids))
        return citations(generated, valid_ids)

    def observe_resolution(*args, **kwargs):
        result = resolve(*args, **kwargs)
        emit("parsed_brief", brief=result.model_dump(mode="json"))
        return result

    def observe_structure(claim_id, claim, evidence, duplicates, sources, conflicts):
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
                **span.model_dump(mode="json"),
                "association_resolves": association,
                "actual_slice": chunk.text[span.start:span.end] if chunk else None,
                "exact_match": bool(association and span.end <= len(chunk.text)
                                    and chunk.text[span.start:span.end] == span.text),
            })
        emit("deterministic", claim_id=claim_id, claim=claim.model_dump(mode="json"),
             has_citations=bool(claim.citations),
             citation_check_note="Empty citation sets satisfy subset checks vacuously; the production uncited_claim check rejects them.",
             citations_resolve=all(item in sources for item in claim.citations),
             spans=spans,
             every_citation_has_span=set(claim.citations).issubset(
                 {span.source_id for span in claim.supporting_spans}),
             every_citation_has_valid_span=set(claim.citations).issubset(
                 {span["source_id"] for span in spans if span["exact_match"]}),
             judgment=result.model_dump(mode="json") if result else None,
             semantic_checks="Modality, polarity, attribution and temporal meaning are model judgments, not deterministic checks.")
        return result

    def observe_generate(self, system_prompt, user_prompt, schema):
        stage = "semantic" if schema.get("title") == "SemanticSupportDecision" else "synthesis"
        emit(stage + "_input", user_prompt=user_prompt, schema=schema)
        result = generate(self, system_prompt, user_prompt, schema)
        # The adapter returns message.content only, never Ollama's thinking field.
        emit(stage + "_output", output=result)
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
        emit("result", result=result.model_dump(mode="json"))
        return result
    except Exception as exc:
        report = getattr(exc, "report", None)
        emit("failure", exception=type(exc).__name__, message=str(exc),
             report=report.model_dump(mode="json") if report else None)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--max-distance", type=float, required=True)
    parser.add_argument("--min-evidence", type=int, default=2)
    parser.add_argument("--results", type=int, default=5)
    args = parser.parse_args()
    run(args.query, args.max_distance, args.min_evidence, args.results)

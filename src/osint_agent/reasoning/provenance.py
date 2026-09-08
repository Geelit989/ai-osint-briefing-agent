"""Resolve exact model-selected quotes without repairing claims or citations."""

from osint_agent.models.brief import (
    ClaimSupportJudgment, ClaimSupportReport, GeneratedBrief, SourceReference,
    SynthesisDraft,
)
from osint_agent.models.document import EvidenceChunk
from osint_agent.reasoning.claim_support import ClaimSupportValidationFailure


def resolve_provenance(
    draft: SynthesisDraft,
    evidence: list[EvidenceChunk],
    sources: list[SourceReference],
) -> GeneratedBrief:
    """Require one exact occurrence, optionally restricted to a declared chunk.

    Overlapping occurrences count as ambiguous too. No invalid quote is dropped
    from an accepted result; any resolution failure rejects the entire brief.
    Citation contribution and semantic support remain downstream checks.
    """
    sources_by_id = {source.source_id: source for source in sources}
    payload = draft.model_dump()
    judgments = []
    fields = [("title", payload["title"]), ("bluf", payload["bluf"])]
    for section in ("reported_developments", "analytic_assessments", "intelligence_gaps"):
        fields.extend((f"{section}[{i}]", claim) for i, claim in enumerate(payload[section]))
    for claim_id, claim in fields:
        spans = []
        for quote in claim.pop("supporting_quotes"):
            source = sources_by_id.get(quote["source_id"])
            candidates = [chunk for chunk in evidence if source
                          and chunk.doc_id == source.doc_id
                          and chunk.chunk_id in source.chunk_ids
                          and (quote["chunk_id"] is None or chunk.chunk_id == quote["chunk_id"])]
            matches = []
            for chunk in candidates:
                start = chunk.text.find(quote["text"])
                while start != -1:
                    matches.append((chunk, start))
                    start = chunk.text.find(quote["text"], start + 1)
            if len(matches) != 1:
                judgments.append(ClaimSupportJudgment(
                    claim_id=claim_id, claim_text=claim["text"], status="unsupported",
                    issues=["invalid_span_provenance"],
                    rationale=f"Quote must resolve uniquely to supplied source/chunk; found {len(matches)} occurrences.",
                ))
                continue
            chunk, start = matches[0]
            # Duplicate IDs or inconsistent mappings cannot establish provenance.
            if sum(item.chunk_id == chunk.chunk_id for item in evidence) != 1:
                judgments.append(ClaimSupportJudgment(
                    claim_id=claim_id, claim_text=claim["text"], status="unsupported",
                    issues=["invalid_claim_evidence_association"],
                    rationale="Selected chunk identifier is not unique.",
                ))
                continue
            end = start + len(quote["text"])
            assert chunk.text[start:end] == quote["text"]
            spans.append(dict(source_id=quote["source_id"], chunk_id=chunk.chunk_id,
                              start=start, end=end, text=quote["text"]))
        claim["supporting_spans"] = spans
    if judgments:
        raise ClaimSupportValidationFailure(ClaimSupportReport(status="unsupported", judgments=judgments))
    return GeneratedBrief.model_validate(payload)

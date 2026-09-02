from __future__ import annotations

import sqlite3

import pytest

from osint_agent.config import settings
from osint_agent.models.document import Document, EvidenceChunk
from osint_agent.retrieval.evidence_identity import (
    canonicalize_url,
    content_fingerprint,
    group_usable_evidence,
)
from osint_agent.retrieval.sufficiency import check_retrieval_sufficiency
from osint_agent.storage.insert_data import upsert_document
from osint_agent.storage.sqlite import create_db
from osint_agent.workflow import reason_over_evidence


def document(
    doc_id: str,
    text: str,
    *,
    url: str | None = None,
    provider: str = "fixture",
) -> Document:
    return Document(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        source=f"Source {doc_id}",
        provider=provider,
        source_type="news",
        url=url,
        raw_text=text,
        text=text,
    )


def document_with_unvalidated_text(doc_id: str, text: str) -> Document:
    """Represent a persisted legacy record that bypassed model validation."""

    valid_document = document(doc_id, "placeholder authoritative content")
    return valid_document.model_copy(update={"text": text})


def chunk(
    doc_id: str,
    index: int = 0,
    *,
    distance: float = 0.2,
    url: str | None = None,
    provider: str = "fixture",
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=f"{doc_id}::chunk-{index:03}",
        doc_id=doc_id,
        text=f"Retrieved chunk {doc_id} {index}",
        provider=provider,
        url=url,
        distance=distance,
    )


def assess(
    chunks: list[EvidenceChunk],
    documents: list[Document],
    *,
    min_evidence: int = 2,
):
    return check_retrieval_sufficiency(
        chunks,
        max_distance=0.5,
        min_evidence=min_evidence,
        authoritative_documents={item.doc_id: item for item in documents},
    )


def test_distinct_documents_are_two_sufficient_units():
    chunks = [chunk("doc-a"), chunk("doc-b")]
    documents = [
        document("doc-a", "First distinct report.", url="https://a.test/one"),
        document("doc-b", "Second distinct report.", url="https://b.test/two"),
    ]

    result = assess(chunks, documents)

    assert result.sufficient is True
    assert (
        result.retrieved_chunk_count,
        result.usable_chunk_count,
        result.independent_evidence_count,
    ) == (2, 2, 2)


def test_same_document_chunks_are_one_insufficient_unit():
    chunks = [chunk("doc-a", 0), chunk("doc-a", 1)]

    result = assess(chunks, [document("doc-a", "One full report.")])

    assert result.sufficient is False
    assert result.usable_chunk_count == 2
    assert result.independent_evidence_count == 1
    assert result.grouping_manifest[0].grouping_reasons == ["same_doc_id"]


def test_exact_normalized_authoritative_content_is_one_unit():
    chunks = [chunk("doc-a"), chunk("doc-b")]
    documents = [
        document("doc-a", "CAFÉ\nreported   an UPDATE.", url="https://a.test"),
        document("doc-b", "cafe\u0301 reported an update.", url="https://b.test"),
    ]

    result = assess(chunks, documents)

    assert result.independent_evidence_count == 1
    assert result.grouping_manifest[0].grouping_reasons == [
        "content_fingerprint"
    ]
    first_fingerprint = content_fingerprint(documents[0].text)
    second_fingerprint = content_fingerprint(documents[1].text)
    assert first_fingerprint is not None
    assert first_fingerprint == second_fingerprint


def test_empty_authoritative_content_has_no_fingerprint_and_stays_separate():
    chunks = [chunk("doc-a"), chunk("doc-b")]
    documents = [
        document_with_unvalidated_text("doc-a", ""),
        document_with_unvalidated_text("doc-b", ""),
    ]

    result = assess(chunks, documents)

    assert content_fingerprint("") is None
    assert result.usable_chunk_count == 2
    assert result.independent_evidence_count == 2
    assert all(
        group.grouping_reasons == ["singleton"]
        for group in result.grouping_manifest
    )


def test_whitespace_only_authoritative_content_is_not_fingerprinted():
    chunks = [chunk("doc-a"), chunk("doc-b")]
    documents = [
        document_with_unvalidated_text("doc-a", " \n\t "),
        document_with_unvalidated_text("doc-b", "\r\n  "),
    ]

    result = assess(chunks, documents)

    assert content_fingerprint(" \n\t ") is None
    assert result.usable_chunk_count == 2
    assert result.independent_evidence_count == 2
    assert all(
        "content_fingerprint" not in group.grouping_reasons
        for group in result.grouping_manifest
    )


@pytest.mark.parametrize(
    ("first_url", "second_url"),
    [
        (
            "HTTPS://Example.TEST/path/#first",
            "https://example.test/path",
        ),
        (
            "https://example.test/",
            "https://EXAMPLE.test#section",
        ),
    ],
)
def test_canonical_equivalent_urls_are_one_unit(first_url, second_url):
    chunks = [chunk("doc-a"), chunk("doc-b")]
    documents = [
        document("doc-a", "First report.", url=first_url),
        document("doc-b", "Different report.", url=second_url),
    ]

    result = assess(chunks, documents)

    assert canonicalize_url(first_url) == canonicalize_url(second_url)
    assert result.independent_evidence_count == 1
    assert result.grouping_manifest[0].grouping_reasons == ["canonical_url"]


def test_same_canonical_url_overrides_different_text():
    chunks = [chunk("doc-a"), chunk("doc-b")]
    documents = [
        document("doc-a", "Old page content.", url="https://example.test/item/"),
        document("doc-b", "Updated page content.", url="https://EXAMPLE.test/item"),
    ]

    result = assess(chunks, documents)

    assert result.independent_evidence_count == 1
    assert result.grouping_manifest[0].grouping_reasons == ["canonical_url"]


def test_provider_diversity_does_not_inflate_identical_content():
    chunks = [
        chunk("doc-a", provider="provider-one"),
        chunk("doc-b", provider="provider-two"),
    ]
    documents = [
        document("doc-a", "Identical wire copy.", provider="provider-one"),
        document("doc-b", "Identical wire copy.", provider="provider-two"),
    ]

    result = assess(chunks, documents)

    assert result.independent_evidence_count == 1
    assert "content_fingerprint" in result.grouping_manifest[0].grouping_reasons


def test_same_provider_does_not_collapse_distinct_reports():
    chunks = [chunk("doc-a"), chunk("doc-b")]
    documents = [
        document("doc-a", "First report.", url="https://same.test/one"),
        document("doc-b", "Second report.", url="https://same.test/two"),
    ]

    result = assess(chunks, documents)

    assert result.independent_evidence_count == 2
    assert all(
        group.grouping_reasons == ["singleton"]
        for group in result.grouping_manifest
    )


def test_irrelevant_chunks_are_excluded_before_grouping():
    chunks = [chunk("doc-a", distance=0.2), chunk("doc-b", distance=0.8)]
    documents = [
        document("doc-a", "Same full content."),
        document("doc-b", "Same full content."),
    ]

    result = assess(chunks, documents, min_evidence=1)

    assert result.retrieved_chunk_count == 2
    assert result.usable_chunk_count == 1
    assert result.independent_evidence_count == 1
    assert result.grouping_manifest[0].member_document_ids == ["doc-a"]


def test_duplicate_pair_plus_distinct_record_is_two_units():
    chunks = [chunk("doc-a"), chunk("doc-a-copy"), chunk("doc-b")]
    documents = [
        document("doc-a", "Duplicated report."),
        document("doc-a-copy", "Duplicated report."),
        document("doc-b", "Independent distinct report."),
    ]

    result = assess(chunks, documents)

    assert result.usable_chunk_count == 3
    assert result.independent_evidence_count == 2
    assert result.sufficient is True


def test_raw_chunk_count_regression_no_longer_passes_threshold():
    chunks = [chunk("doc-a", 0), chunk("doc-a", 1)]

    result = assess(chunks, [document("doc-a", "One report.")])

    assert result.retrieved_chunk_count == 2
    assert result.sufficient is False
    assert "insufficient independent evidence" in result.reason


def test_missing_urls_fall_back_to_authoritative_content_identity():
    chunks = [chunk("doc-a"), chunk("doc-b")]
    documents = [
        document("doc-a", "Same report without URL."),
        document("doc-b", "Same report without URL."),
    ]

    result = assess(chunks, documents)

    assert result.independent_evidence_count == 1
    assert result.grouping_manifest[0].grouping_reasons == [
        "content_fingerprint"
    ]


def test_similar_but_nonidentical_reporting_remains_distinct():
    chunks = [chunk("doc-a"), chunk("doc-b")]
    documents = [
        document("doc-a", "Officials reported an exercise began Tuesday."),
        document("doc-b", "The Tuesday exercise involved naval forces."),
    ]

    result = assess(chunks, documents)

    assert result.independent_evidence_count == 2
    assert result.sufficient is True


def test_duplicate_chunks_remain_in_reasoning_context(monkeypatch):
    chunks = [chunk("doc-a", 0), chunk("doc-a", 1)]
    captured: dict[str, object] = {}

    def fake_synthesize(query, evidence, model=None, support_model=None):
        captured["evidence"] = evidence
        return "synthesized"

    monkeypatch.setattr("osint_agent.workflow.synthesize_brief", fake_synthesize)

    result = reason_over_evidence(
        "query", chunks, max_distance=0.5, min_evidence=1
    )

    assert result == "synthesized"
    assert captured["evidence"] == chunks
    assert captured["evidence"][0] is chunks[0]
    assert captured["evidence"][1] is chunks[1]


def test_grouping_manifest_is_complete_stable_and_auditable():
    chunks = [
        chunk("doc-a", 1),
        chunk("doc-url-copy"),
        chunk("doc-single"),
        chunk("doc-a-copy"),
        chunk("doc-url"),
        chunk("doc-a", 0),
    ]
    documents = [
        document("doc-a", "Content duplicate.", url="https://a.test/original"),
        document("doc-a-copy", "content   DUPLICATE.", url="https://b.test/copy"),
        document("doc-url", "First URL text.", url="https://url.test/item/"),
        document(
            "doc-url-copy",
            "Different URL text.",
            url="HTTPS://URL.test/item#fragment",
        ),
        document("doc-single", "Unique singleton."),
    ]

    result = assess(chunks, documents, min_evidence=3)
    reversed_manifest = group_usable_evidence(
        list(reversed(chunks)),
        {item.doc_id: item for item in documents},
    )

    assert result.independent_evidence_count == 3
    assert result.grouping_manifest == reversed_manifest
    assert sorted(
        member
        for group in result.grouping_manifest
        for member in group.member_chunk_ids
    ) == sorted(item.chunk_id for item in chunks)
    reasons = {
        reason
        for group in result.grouping_manifest
        for reason in group.grouping_reasons
    }
    assert reasons == {
        "same_doc_id",
        "canonical_url",
        "content_fingerprint",
        "singleton",
    }
    assert all(
        group.unit_id.startswith("evidence-unit-")
        for group in result.grouping_manifest
    )
    payload = result.model_dump()
    assert payload["retrieved_chunk_count"] == 6
    assert payload["usable_chunk_count"] == 6
    assert payload["independent_evidence_count"] == 3
    assert payload["grouping_manifest"][0]["unit_id"].startswith(
        "evidence-unit-"
    )


def test_authoritative_content_identity_loads_from_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "identity.db"
    create_db(db_path)
    documents = [
        document("doc-a", "Persisted authoritative content."),
        document("doc-b", "PERSISTED  authoritative\ncontent."),
    ]
    with sqlite3.connect(db_path) as con:
        for item in documents:
            upsert_document(con, item)
    monkeypatch.setattr(settings, "DB_PATH", db_path)

    result = check_retrieval_sufficiency(
        [chunk("doc-a"), chunk("doc-b")],
        max_distance=0.5,
        min_evidence=2,
    )

    assert result.independent_evidence_count == 1
    assert result.grouping_manifest[0].grouping_reasons == [
        "content_fingerprint"
    ]

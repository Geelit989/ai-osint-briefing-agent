from types import SimpleNamespace

from osint_agent.models.document import EvidenceChunk
from osint_agent.retrieval.semantic import semantic_search


def test_semantic_search_returns_evidence_chunks(monkeypatch):
    """Semantic search should convert Chroma results into EvidenceChunk models."""

    fake_embedding = [0.1, 0.2, 0.3]

    fake_results = {
        "ids": [
            [
                "doc-123::chunk-000::abc123",
                "doc-456::chunk-001::def456",
            ]
        ],
        "documents": [
            [
                "Saudi Arabia and Turkey expanded military cooperation.",
                "Regional defense coordination increased during the week.",
            ]
        ],
        "metadatas": [
            [
                {
                    "doc_id": "doc-123",
                    "title": "Military Cooperation Expands",
                    "source": "Example Source",
                    "provider": "currents",
                    "source_type": "news",
                    "published_date": "2026-08-10",
                    "event_time": "2026-08-09T20:00:00+00:00",
                    "retrieved_at": "2026-08-10T01:00:00+00:00",
                    "contradiction_group": "event-1",
                    "contradiction_position": "affirmation",
                },
                {
                    "doc_id": "doc-456",
                    "title": "Regional Defense Developments",
                    "source": "Second Source",
                    "provider": "currents",
                    "source_type": "news",
                    "published_date": "2026-08-11",
                },
            ]
        ],
        "distances": [
            [
                0.2834,
                0.4040,
            ]
        ],
    }

    monkeypatch.setattr(
        "osint_agent.retrieval.semantic.embed_query",
        lambda query: fake_embedding,
    )

    monkeypatch.setattr(
        "osint_agent.retrieval.semantic.query_chunks",
        lambda query_embedding, n_results: fake_results,
    )
    monkeypatch.setattr(
        "osint_agent.retrieval.semantic.assert_index_usable",
        lambda **kwargs: SimpleNamespace(expected_chunk_count=2),
    )

    results = semantic_search(
        "military cooperation",
        n_results=2,
    )

    assert len(results) == 2
    assert all(
        isinstance(result, EvidenceChunk)
        for result in results
    )

    first = results[0]

    assert first.chunk_id == "doc-123::chunk-000::abc123"
    assert first.doc_id == "doc-123"
    assert first.title == "Military Cooperation Expands"
    assert first.source == "Example Source"
    assert first.provider == "currents"
    assert first.source_type == "news"
    assert first.published_date == "2026-08-10"
    assert first.event_time == "2026-08-09T20:00:00+00:00"
    assert first.retrieved_at == "2026-08-10T01:00:00+00:00"
    assert first.contradiction_group == "event-1"
    assert first.contradiction_position == "affirmation"
    assert first.distance == 0.2834
    assert (
        first.text
        == "Saudi Arabia and Turkey expanded military cooperation."
    )

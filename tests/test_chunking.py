"""Tests for chunking documents"""
from osint_agent.models.document import Document
from osint_agent.preprocessing.chunking import chunk_document



def test_short_document_creates_one_chunk():

    document = Document(
        doc_id="test-doc",
        title="Test Article",
        raw_text="North Korea launched a missile.",
        text="North Korea launched a missile.",
        provider="test-provider",
        source_type="test",
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].doc_id == "test-doc"
    assert chunks[0].chunk_index == 0
    assert "Test Article" in chunks[0].text


def test_long_document_creates_multiple_chunks():

    long_text = "North Korea launched a missile. " * 1000

    document = Document(
        doc_id="test-doc",
        title="Test Article",
        raw_text=long_text,
        text=long_text,
        provider="test-provider",
        source_type="test",
    )

    chunks = chunk_document(document)

    assert len(chunks) > 1


def test_chunk_ids_are_deterministic():

    long_text = "North Korea launched a missile. " * 1000

    document = Document(
        doc_id="test-doc",
        title="Test Article",
        raw_text=long_text,
        text=long_text,
        provider="test-provider",
        source_type="test",
    )

    first_run = chunk_document(document)
    second_run = chunk_document(document)

    first_ids = [chunk.chunk_id for chunk in first_run]
    second_ids = [chunk.chunk_id for chunk in second_run]

    assert first_ids == second_ids
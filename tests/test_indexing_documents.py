import pytest
from unittest.mock import patch

from osint_agent.indexing.indexing_documents import index_document, index_documents, IndexingResult
from osint_agent.models.document import Document



@patch("osint_agent.indexing.indexing_documents.upsert_chunks")
@patch("osint_agent.indexing.indexing_documents.embed_documents")
def test_index_document_embeds_and_upserts_chunks(
    mock_embed_documents,
    mock_upsert_chunks,
):
    document = Document(
        doc_id="test-doc-001",
        title="Test Article",
        source="Test Source",
        raw_text="This is a test article about regional military cooperation.",
        text="This is a test article about regional military cooperation.",
        provider="currents",
        source_type="news api",
    )

    mock_embed_documents.return_value = [
        [0.1, 0.2, 0.3]
    ]

    result = index_document(document)

    assert result.doc_id == "test-doc-001"
    assert result.chunks_created == 1
    assert result.chunks_indexed == 1
    assert len(result.chunk_ids) == 1

    mock_embed_documents.assert_called_once()

    embedded_texts = mock_embed_documents.call_args.args[0]

    assert len(embedded_texts) == 1
    assert "Test Article" in embedded_texts[0]
    assert "regional military cooperation" in embedded_texts[0]

    mock_upsert_chunks.assert_called_once()


@patch("osint_agent.indexing.indexing_documents.upsert_chunks")
@patch("osint_agent.indexing.indexing_documents.embed_documents")
def test_index_document_rejects_embedding_count_mismatch(
    mock_embed_documents,
    mock_upsert_chunks,
):
    document = Document(
        doc_id="test-doc-002",
        title="Test Article",
        source="Test Source",
        raw_text="This is another test article.",
        text="This is another test article.",
        source_type="news api",
        provider="currents",
    )

    mock_embed_documents.return_value = []

    with pytest.raises(
        ValueError,
        match="Chunk/embedding count mismatch",
    ):
        index_document(document)

    mock_upsert_chunks.assert_not_called()


@patch("osint_agent.indexing.indexing_documents.index_document")
def test_index_documents_indexes_each_document(
    mock_index_document,
):
    documents = [
        Document(
            doc_id="doc-001",
            title="Article One",
            source="Test Source",
            provider="test-provider",
            source_type="news",
            raw_text="First article.",
            text="First article.",
        ),
        Document(
            doc_id="doc-002",
            title="Article Two",
            source="Test Source",
            provider="test-provider",
            source_type="news",
            raw_text="Second article.",
            text="Second article.",
        ),
    ]

    mock_index_document.side_effect = [
        IndexingResult(
            doc_id="doc-001",
            chunks_created=1,
            chunks_indexed=1,
            chunk_ids=["chunk-001"],
        ),
        IndexingResult(
            doc_id="doc-002",
            chunks_created=1,
            chunks_indexed=1,
            chunk_ids=["chunk-002"],
        ),
    ]

    results = index_documents(documents)

    assert len(results) == 2
    assert results[0].doc_id == "doc-001"
    assert results[1].doc_id == "doc-002"
    assert mock_index_document.call_count == 2
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


from osint_agent.models.document import Document
from osint_agent.indexing.indexing_documents import index_document
from osint_agent.storage.chroma import get_document_collection


TEST_DOC_ID = "stale-chunk-test"


def get_doc_records(doc_id: str) -> dict:
    """Return all Chroma records for one document."""

    collection = get_document_collection()

    return collection.get(
        where={"doc_id": doc_id}
    )


def test_reindexing_document_removes_stale_chunks():
    """Re-indexing a document should replace all prior Chroma chunks."""

    collection = get_document_collection()

    # Ensure the test starts from a clean state.
    collection.delete(
        where={"doc_id": TEST_DOC_ID}
    )

    long_text = (
        "Military forces conducted a large regional exercise involving "
        "aircraft, naval vessels, and ground units. "
    ) * 200

    doc_v1 = Document(
        doc_id=TEST_DOC_ID,
        title="Stale Chunk Test",
        source="test",
        provider="manual",
        source_type="test",
        published_date=None,
        url=None,
        raw_text=long_text,
        text=long_text,
    )

    result_v1 = index_document(doc_v1)
    records_v1 = get_doc_records(TEST_DOC_ID)

    old_ids = set(records_v1["ids"])

    assert result_v1.chunks_created > 1
    assert len(old_ids) == result_v1.chunks_created

    short_text = (
        "Military forces concluded the regional exercise and returned "
        "to their home stations."
    )

    doc_v2 = Document(
        doc_id=TEST_DOC_ID,
        title="Stale Chunk Test",
        source="test",
        provider="manual",
        source_type="test",
        published_date=None,
        url=None,
        raw_text=short_text,
        text=short_text,
    )

    result_v2 = index_document(doc_v2)
    records_v2 = get_doc_records(TEST_DOC_ID)

    new_ids = set(records_v2["ids"])
    stale_ids = old_ids - new_ids

    current_ids = set(
        get_doc_records(TEST_DOC_ID)["ids"]
    )

    surviving_stale_ids = stale_ids & current_ids

    assert result_v2.chunks_created == 1
    assert not surviving_stale_ids
    assert current_ids == new_ids
    assert len(current_ids) == result_v2.chunks_created

    # Cleanup so the test does not leave test records in Chroma.
    collection.delete(
        where={"doc_id": TEST_DOC_ID}
    )
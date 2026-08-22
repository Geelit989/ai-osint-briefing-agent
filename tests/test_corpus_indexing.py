from unittest.mock import MagicMock, patch

from osint_agent.indexing.corpus import index_corpus
from osint_agent.indexing.indexing_documents import IndexingResult


@patch("osint_agent.indexing.corpus.index_documents")
@patch("osint_agent.indexing.corpus.create_document_collection")
@patch("osint_agent.indexing.corpus.get_documents")
def test_index_corpus_indexes_all_documents(
    mock_get_documents,
    mock_create_collection,
    mock_index_documents,
):
    documents = [MagicMock(), MagicMock()]
    mock_get_documents.return_value = documents

    collection = MagicMock()
    collection.count.side_effect = [10, 12]
    mock_create_collection.return_value = collection

    mock_index_documents.return_value = [
        IndexingResult(
            doc_id="doc-1",
            chunks_created=1,
            chunks_indexed=1,
            chunk_ids=["chunk-1"],
        ),
        IndexingResult(
            doc_id="doc-2",
            chunks_created=2,
            chunks_indexed=2,
            chunk_ids=["chunk-2", "chunk-3"],
        ),
    ]

    result = index_corpus()

    assert result.documents_found == 2
    assert result.documents_indexed == 2
    assert result.chunks_indexed == 3
    assert result.chroma_count_before == 10
    assert result.chroma_count_after == 12

    mock_index_documents.assert_called_once_with(documents)
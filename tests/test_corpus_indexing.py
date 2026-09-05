from unittest.mock import patch

from osint_agent.indexing.corpus import CorpusIndexingResult, index_corpus


@patch("osint_agent.indexing.corpus.reconcile_index")
def test_index_corpus_uses_explicit_reconciliation(mock_reconcile):
    expected = CorpusIndexingResult(
        documents_found=2,
        documents_indexed=2,
        chunks_indexed=3,
        chroma_count_before=10,
        chroma_count_after=3,
        changed=True,
        corpus_status="current",
        compatibility_fingerprint="sha256-fixture",
    )
    mock_reconcile.return_value = expected

    assert index_corpus() is expected
    mock_reconcile.assert_called_once_with()

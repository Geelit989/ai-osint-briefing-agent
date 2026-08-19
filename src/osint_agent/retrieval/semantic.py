from osint_agent.indexing.embedding import embed_query
from osint_agent.models.document import EvidenceChunk
from osint_agent.storage.chroma import query_chunks


def semantic_search(
    query: str,
    n_results: int = 5,
) -> list[EvidenceChunk]:
    """Retrieve semantically similar chunks for a user query."""

    query_embedding = embed_query(query)

    results = query_chunks(
        query_embedding=query_embedding,
        n_results=n_results,
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    evidence = []

    for chunk_id, text, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        evidence.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                doc_id=metadata["doc_id"],
                text=text,
                title=metadata.get("title") or None,
                source=metadata.get("source") or None,
                provider=metadata.get("provider") or None,
                source_type=metadata.get("source_type") or None,
                published_date=metadata.get("published_date") or None,
                url=metadata.get("url") or None,
                distance=distance,
            )
        )

    return evidence

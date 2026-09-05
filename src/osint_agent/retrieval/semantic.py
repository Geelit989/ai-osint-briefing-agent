from osint_agent.indexing.embedding import embed_query
from osint_agent.indexing.state import assert_index_usable
from osint_agent.models.document import EvidenceChunk
from osint_agent.storage.chroma import query_chunks


def semantic_search(
    query: str,
    n_results: int = 5,
    *,
    runtime_manifest: dict | None = None,
) -> list[EvidenceChunk]:
    """Retrieve semantically similar chunks for a user query."""

    inspection = assert_index_usable(runtime_manifest=runtime_manifest)
    if inspection.expected_chunk_count == 0:
        return []

    query_embedding = embed_query(query)

    results = query_chunks(
        query_embedding=query_embedding,
        n_results=min(n_results, inspection.expected_chunk_count or n_results),
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
                event_time=metadata.get("event_time") or None,
                retrieved_at=metadata.get("retrieved_at") or None,
                url=metadata.get("url") or None,
                contradiction_group=(
                    metadata.get("contradiction_group") or None
                ),
                contradiction_position=(
                    metadata.get("contradiction_position") or None
                ),
                distance=distance,
            )
        )

    return evidence

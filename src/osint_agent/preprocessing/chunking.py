import hashlib

from transformers import AutoTokenizer

from osint_agent.models.document import Document, Chunk
from osint_agent.config import settings


CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP

tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_NAME)


def create_chunk_id(
    doc_id: str,
    chunk_index: int,
    text: str,
) -> str:
    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:12]

    return f"{doc_id}::chunk-{chunk_index:03d}::{content_hash}"


def normalize_for_comparison(text: str) -> str:
    return " ".join(text.lower().split())



def chunk_document(document: Document) -> list[Chunk]:

    if CHUNK_SIZE < 1 or not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError(
            "chunk size must be positive and overlap must be smaller than size"
        )

    tokens = tokenizer.encode(
        document.text,
        add_special_tokens=False,
    )

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):

        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
        ).strip()

        normalized_title = normalize_for_comparison(document.title or "")
        normalized_chunk = normalize_for_comparison(chunk_text)

        if document.title and not normalized_chunk.startswith(normalized_title):
            chunk_text = f"{document.title}\n\n{chunk_text}"

        chunks.append(
            Chunk(
                chunk_id=create_chunk_id(
                    document.doc_id,
                    chunk_index,
                    chunk_text,
                ),
                doc_id=document.doc_id,
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=len(chunk_tokens),
            )
        )

        chunk_index += 1

        if end >= len(tokens):
            break

        start = end - CHUNK_OVERLAP

    return chunks

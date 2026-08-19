from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator



class Document(BaseModel):
    doc_id: str
    title: str | None = None
    source: str | None = None
    provider: str
    source_type: str
    published_date: datetime | None = None
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    url: str | None = None
    raw_text: str
    text: str
    meta_data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("raw_text", "text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("Document text cannot be empty")
        return cleaned_value
    

    @field_validator("published_date", mode="before")
    def parse_published_date(cls, value):
        if value is None:
            return None
        
        if isinstance(value, datetime):
            return value
        
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 with timezone
            "%Y-%m-%dT%H:%M:%S",    # ISO 8601 without timezone
            "%Y-%m-%d %H:%M:%S",    # Common format
            "%Y-%m-%d",             # Date only
            "%Y/%m/%d",
            "%B %d, %Y",     # May 5, 2026
            "%b %d, %Y",     # May 5, 2026 (short)
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S %z",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue    

        raise ValueError(f"Unable to parse published_date: {value}")
    

    def to_record(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
    

class Entity(BaseModel):
    ent_text: str
    start_char: int
    end_char: int
    label: str
    doc_id: str

    def to_record(self) -> dict:
        return self.model_dump(mode="json")


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    token_count: int


class EvidenceChunk(BaseModel):
    """A semantically retrieved chunk used as evidence by ARGUS."""

    chunk_id: str
    doc_id: str
    text: str

    title: str | None = None
    source: str | None = None
    provider: str | None = None
    source_type: str | None = None
    published_date: str | None = None
    url: str | None = None

    distance: float = Field(
        ...,
        description="Raw Chroma retrieval distance; lower is more similar.",
    )

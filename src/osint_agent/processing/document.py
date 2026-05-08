from datetime import datetime

from pydantic import BaseModel, field_validator



class Document(BaseModel):
    doc_id: str
    title: str | None = None
    source: str | None = None
    published_date: datetime | None = None
    url: str | None = None
    raw_text: str
    text: str

    @field_validator("text")
    def text_must_not_be_empty(cls, text_value):
        if not text_value.strip():
            raise ValueError("Cleaned text cannot be empty")
        return text_value
    

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
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue    

        raise ValueError(f"Unable to parse published_date: {value}")
    

    def to_record(self) -> dict:
        return self.model_dump(mode="json")

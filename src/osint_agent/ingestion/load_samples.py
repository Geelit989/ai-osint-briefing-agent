import json
from pathlib import Path

from osint_agent.processing.clean_text import clean_text
from osint_agent.processing.document import Document


SAMPLE_PATH = Path("data/samples/sample_articles.json")
PROCESSED_PATH = Path("data/processed/cleaned_articles.json")


def load_sample_articles(path: str | Path) -> list[dict]:
    """Load articles for the ingestion workflow."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Sample file not found.: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def process_sample_articles(
    source_path: str | Path = SAMPLE_PATH,
    output_path: str | Path = PROCESSED_PATH,
) -> list[dict]:
    """Load, clean, validate, and save sample articles."""

    articles = load_sample_articles(source_path)
    processed_articles = []

    for idx, article in enumerate(articles):
        raw_text = article.get("text", "")

        structured_doc = {
            "doc_id": f"doc_{idx:03}",
            "title": article.get("title"),
            "source": article.get("source"),
            "published_date": article.get("published_date"),
            "url": article.get("url"),
            "raw_text": raw_text,
            "text": clean_text(raw_text),
        }

        doc = Document(**structured_doc)
        processed_articles.append(doc.to_record())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(processed_articles, file, indent=2, ensure_ascii=False)

    return processed_articles


if __name__ == "__main__":
    processed = process_sample_articles()

    for article in processed:
        print(json.dumps(article, indent=2, default=str))

    print(f"Saved {len(processed)} cleaned articles to {PROCESSED_PATH}")

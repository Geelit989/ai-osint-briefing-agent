import json
from pathlib import Path

from osint_agent.processing.clean_text import clean_text
from osint_agent.processing.document import Document


PROCESSED_PATH = Path("data/processed/cleaned_articles.json")

def load_sample_articles(path: str | Path) -> list[dict]:
    """
    Load articles for ingestion script.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Sample file not found.: {path}")
    
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
    
    
if __name__ == "__main__":
    articles = load_sample_articles("data/samples/sample_articles.json")
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

        print(json.dumps(doc.to_record(), indent=2, default=str)) # Print the docs

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists)


    with PROCESSED_PATH.open("w", encoding="utf-8") as f:
        json.dump(processed_articles, f, indent=2, ensure_ascii=False) # Save the cleaned articles

    print(f"Saved {len(processed_articles)} cleaned articles to {PROCESSED_PATH}")
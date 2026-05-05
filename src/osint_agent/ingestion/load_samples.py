import json
from pathlib import Path


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

    for idx, article in enumerate(articles):
        structured_doc = {
            "doc_id": f"doc_{idx:03}",
            "title": article.get("title"),
            "source": article.get("source"),
            "published_date": article.get("published_date"),
            "url": article.get("url"),
            "text": article.get("text"),
        }

        print(json.dumps(structured_doc, indent=2))
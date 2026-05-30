# pip install -U spacy
# python -m spacy download en_core_web_sm

import json
from pathlib import Path

import spacy


SRC_PATH = Path("data/processed/cleaned_articles.json")
ENT_PATH = Path("data/processed/extracted_entities.json")


def extract_entities(path: str | Path) -> list[dict]:
    """Extract named entities from a json file."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        articles = json.load(f)

    entities = []

    nlp = spacy.load("en_core_web_sm")

    for article in articles:
        doc = nlp(article.get("text", ""))
        doc_id = article.get("doc_id", "")

        for idx, ent in enumerate(doc.ents):
            entities.append(
                {
                    "entity_id": f"{doc_id}_ent_{idx:03}",
                    "doc_id": doc_id,
                    "text": ent.text,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "label": ent.label_,
                }
            )

    return entities


def save_entities(entities: list[dict], path: str | Path = ENT_PATH) -> None:
    """Save extracted entities to a JSON file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2)


if __name__ == "__main__":
    entities = extract_entities(SRC_PATH)
    save_entities(entities, ENT_PATH)

    print(f"Extracted {len(entities)} entities and saved to {ENT_PATH}")

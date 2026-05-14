# pip install -U spacy
# python -m spacy download en_core_web_sm

import json

import spacy
from pathlib import Path


SRC_PATH = Path("data/processed/cleaned_articles.json")

def extract_entities(path: str | Path) -> list[dict]:
    """Extract named entities from a json file."""

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


if __name__ == "__main__":
    
    entities = extract_entities(SRC_PATH)  

    with open("extracted_entities.json", "w", encoding="utf-8") as f:
        json.dump([{"doc_id": ent["doc_id"], "entity_id": ent["entity_id"], "text": ent["text"], "start_char": ent["start_char"], "end_char": ent["end_char"], "label": ent["label"]} for ent in entities], f, indent=2)
    print(f"Extracted {len(entities)} entities and saved to extracted_entities.json")
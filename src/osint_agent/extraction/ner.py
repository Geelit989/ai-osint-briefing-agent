# pip install -U spacy
# python -m spacy download en_core_web_sm

import json

import spacy
from pathlib import Path


SRC_PATH = Path("data/processed/cleaned_articles.json")

def extract_entities(path: str | Path) -> list[spacy.tokens.Span]:
    """Extract named entities from a json file."""

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(type(data))
    print(len(data))
    print(data.keys())

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(data.get("text", ""))

    return doc.ents


if __name__ == "__main__":
    
    ents = extract_entities(SRC_PATH)  

    for ent in ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
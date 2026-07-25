import spacy
from spacy.language import Language

from osint_agent.processing.document import Document, Entity


def load_ner_model() -> Language:
    return spacy.load("en_core_web_sm")


def extract_entities(
    document: Document,
    nlp: Language,
) -> list[Entity]:
    """Extract named entities from one normalized document."""

    spacy_doc = nlp(document.text)

    return [
        Entity(
            ent_text=entity.text,
            start_char=entity.start_char,
            end_char=entity.end_char,
            label=entity.label_,
            doc_id=document.doc_id,
        )
        for entity in spacy_doc.ents
    ]
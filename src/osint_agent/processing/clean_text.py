import re
import unicodedata

def clean_text(text: str) -> str:
    """ Normalize article text for downstream OSINT processing."""
    if not text:
        return ""
    
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

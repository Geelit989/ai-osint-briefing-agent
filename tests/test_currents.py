from osint_agent.ingestion.currents import currents_to_document


def test_currents_to_document() -> None:
    article = {
        "id": "67fb0677-d00e-5231-9cb0-9e4dd57fd851",
        "title": (
            "Russia Waiting for New US Proposals on Settlement "
            "in Ukraine - Kremlin"
        ),
        "description": (
            "MOSCOW, (Sputnik) - Russia is waiting for new proposals "
            "from the United States on a settlement in Ukraine, Kremlin "
            "spokesman Dmitry Peskov said on Friday."
        ),
        "url": (
            "https://sputnikglobe.com/20260724/"
            "russia-waiting-for-new-us-proposals-on-settlement-"
            "in-ukraine---kremlin-1124491110.html"
        ),
        "author": "Sputnik International",
        "language": "en",
        "category": ["general"],
        "published": "2026-07-24 12:15:38 +0000",
    }

    document = currents_to_document(article)

    assert document.doc_id == (
        "currents-67fb0677-d00e-5231-9cb0-9e4dd57fd851"
    )
    assert document.title.startswith("Russia Waiting")
    assert document.source == "Sputnik International"
    assert document.url == article["url"]
    assert document.published_date is not None
    assert article["description"] in document.raw_text
    assert document.text.strip()
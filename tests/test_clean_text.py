from osint_agent.processing.clean_text import clean_text


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_normalizes_whitespace():
    text = "This   is\n\n messy\ttext."
    assert clean_text(text) == "This is messy text."


def test_clean_text_replaces_smart_quotes_and_dashes():
    text = "The official said “hello”—then left."
    assert clean_text(text) == 'The official said "hello"-then left.'


def test_clean_text_normalizes_unicode():
    text = "Ｔｅｓｔ text"
    assert clean_text(text) == "Test text"
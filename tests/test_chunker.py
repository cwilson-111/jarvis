from jarvis.chunker import SentenceChunker, speakable


def stream(text: str, chunker: SentenceChunker, step: int = 3) -> list[str]:
    """Feed ``text`` in small token-like pieces, as an LLM would."""
    out: list[str] = []
    for i in range(0, len(text), step):
        out += chunker.feed(text[i : i + step])
    return out + chunker.flush()


def test_splits_sentences_as_soon_as_next_one_starts() -> None:
    c = SentenceChunker()
    assert c.feed("Paris is the capital of France.") == []  # could still be "France.com"
    assert c.feed(" It") == ["Paris is the capital of France."]
    assert c.flush() == ["It"]


def test_question_and_exclamation_marks() -> None:
    text = "Are you sure about that? Absolutely certain! Then let's go."
    assert stream(text, SentenceChunker()) == [
        "Are you sure about that?",
        "Absolutely certain!",
        "Then let's go.",
    ]


def test_does_not_split_decimals_abbreviations_or_initials() -> None:
    text = "Pi is about 3.14 according to Dr. Smith and J. R. R. Tolkien, e.g. in books. Done now."
    assert stream(text, SentenceChunker()) == [
        "Pi is about 3.14 according to Dr. Smith and J. R. R. Tolkien, e.g. in books.",
        "Done now.",
    ]


def test_short_fragments_merge_with_the_next_sentence() -> None:
    assert stream("Sure. The answer is forty two.", SentenceChunker(min_chars=10)) == [
        "Sure. The answer is forty two."
    ]


def test_numbered_list_markers_are_not_sentences() -> None:
    text = "Here are two tips.\n1. Drink water.\n2. Sleep well."
    assert stream(text, SentenceChunker()) == [
        "Here are two tips.",
        "1. Drink water.",
        "2. Sleep well.",
    ]


def test_closing_quote_stays_with_its_sentence() -> None:
    assert stream('He said "hello there." Then he left.', SentenceChunker()) == [
        'He said "hello there."',
        "Then he left.",
    ]


def test_long_run_on_is_split_at_a_comma() -> None:
    c = SentenceChunker(min_chars=5, max_chars=40)
    text = "one two three four five, six seven eight nine ten eleven twelve thirteen"
    chunks = stream(text, c)
    assert chunks[0] == "one two three four five,"
    assert all(len(x) <= 40 for x in chunks)
    assert " ".join(chunks) == text


def test_eager_first_releases_first_clause_only() -> None:
    c = SentenceChunker(min_chars=10, eager_first=True)
    text = "The tallest building is the Burj Khalifa, in Dubai, at 828 metres. It opened in 2010."
    assert stream(text, c) == [
        "The tallest building is the Burj Khalifa,",
        "in Dubai, at 828 metres.",
        "It opened in 2010.",
    ]


def test_eager_first_ignores_thousands_separators_and_short_clauses() -> None:
    c = SentenceChunker(min_chars=10, eager_first=True)
    assert stream("Sure, it is 3,000 feet tall, roughly. Yes.", c) == [
        "Sure, it is 3,000 feet tall,",
        "roughly. Yes.",
    ]


def test_flush_resets_state_for_the_next_reply() -> None:
    c = SentenceChunker(eager_first=True)
    c.feed("First reply, with a clause")
    c.flush()
    assert c.feed("Second reply begins, ") == ["Second reply begins,"]


def test_markdown_and_control_characters_are_removed() -> None:
    assert speakable("**Bold** and `code`\x1b[31m red") == "Bold and code [31m red"
    assert stream("# Title\n- **one** item.\n", SentenceChunker()) == ["Title - one item."]


def test_invalid_bounds_rejected() -> None:
    try:
        SentenceChunker(min_chars=10, max_chars=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")

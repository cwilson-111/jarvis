import pytest

from jarvis.wer import WerCounts, align, normalize, number_to_words, wer, wer_counts


def test_identical_after_normalisation() -> None:
    ref = "HE SAID THAT HE'D PAY TWENTY ONE DOLLARS"
    hyp = "He said that he'd pay 21 dollars."
    assert wer(ref, hyp) == 0.0


def test_counts_each_error_type() -> None:
    counts = align(["a", "b", "c", "d"], ["a", "x", "c", "d", "e"])
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 1)
    counts = align(["a", "b", "c", "d"], ["a", "c"])
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 2, 0)
    assert counts.wer == 0.5


def test_textbook_example() -> None:
    # sat->sit, then "the mat" vs "mat down": two substitutions or a deletion plus an
    # insertion cost the same; ties prefer substitution. Either way 3 errors / 6 words.
    counts = wer_counts("the cat sat on the mat", "the cat sit on mat down")
    assert counts == WerCounts(3, 0, 0, 6)
    assert counts.wer == pytest.approx(0.5)


def test_corpus_wer_is_pooled_not_averaged() -> None:
    total = wer_counts("one two three four", "one two three four") + wer_counts("five", "six")
    assert total.wer == pytest.approx(1 / 5)


def test_empty_cases() -> None:
    assert wer("", "") == 0.0
    assert wer_counts("", "extra words").wer == float("inf")
    assert wer("some words", "") == 1.0


def test_normalize() -> None:
    assert normalize("Well-known, 'quoted' don\u2019t 1,200!") == [
        "well", "known", "quoted", "don't", "one", "thousand", "two", "hundred",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (13, "thirteen"),
        (40, "forty"),
        (99, "ninety nine"),
        (100, "one hundred"),
        (305, "three hundred five"),
        (2024, "two thousand twenty four"),
        (1_000_001, "one million one"),
    ],
)
def test_number_to_words(n: int, words: str) -> None:
    assert number_to_words(n) == words

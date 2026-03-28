import pytest


@pytest.fixture
def sample_text() -> str:
    return "This is a sample text for testing TF-IDF calculations."


@pytest.fixture
def sample_corpus():
    return [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog runs fast",
        "The lazy cat sleeps all day",
    ]


@pytest.fixture
def sample_ngrams():
    return ["quick brown", "brown fox", "fox jumps", "jumps over"]

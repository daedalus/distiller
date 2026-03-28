from llm_distiller.core.tfidf import TFIDFHelper


def test_tfidf_initialization(sample_corpus) -> None:
    helper = TFIDFHelper(corpus=sample_corpus)
    assert helper.corpus == sample_corpus
    assert helper.min_tfidf_score == 0.01
    assert helper.min_ngrams == 1
    assert helper.max_ngrams == 1


def test_tfidf_with_custom_params(sample_corpus) -> None:
    helper = TFIDFHelper(
        corpus=sample_corpus,
        min_tfidf_score=0.1,
        min_ngrams=2,
        max_ngrams=3,
    )
    assert helper.min_tfidf_score == 0.1
    assert helper.min_ngrams == 2
    assert helper.max_ngrams == 3


def test_tfidf_compute_tf(sample_text) -> None:
    helper = TFIDFHelper(corpus=[sample_text])
    tf = helper.compute_tf(sample_text)
    assert isinstance(tf, dict)
    assert len(tf) > 0
    assert "sample" in tf


def test_tfidf_compute_idf(sample_corpus) -> None:
    helper = TFIDFHelper(corpus=sample_corpus)
    idf = helper.compute_idf()
    assert isinstance(idf, dict)
    assert len(idf) > 0


def test_tfidf_compute_tfidf(sample_corpus) -> None:
    helper = TFIDFHelper(corpus=sample_corpus)
    tfidf = helper.compute_tfidf()
    assert isinstance(tfidf, dict)
    assert len(tfidf) > 0


def test_tfidf_get_tfidf_for_doc(sample_corpus) -> None:
    helper = TFIDFHelper(corpus=sample_corpus)
    tfidf = helper.get_tfidf_for_doc(0)
    assert isinstance(tfidf, dict)


def test_tfidf_get_top_tfidf_words(sample_corpus) -> None:
    helper = TFIDFHelper(corpus=sample_corpus)
    top_words = helper.get_top_tfidf_words(n=3)
    assert isinstance(top_words, list)


def test_tfidf_empty_corpus() -> None:
    helper = TFIDFHelper(corpus=[])
    assert helper.corpus == []


def test_tfidf_single_document() -> None:
    helper = TFIDFHelper(corpus=["single document only"])
    tfidf = helper.compute_tfidf()
    assert isinstance(tfidf, dict)


def test_tfidf_all_ngrams(sample_corpus) -> None:
    helper = TFIDFHelper(
        corpus=sample_corpus,
        min_ngrams=1,
        max_ngrams=2,
    )
    ngrams = helper.all_ngrams("the quick brown fox")
    assert isinstance(ngrams, list)

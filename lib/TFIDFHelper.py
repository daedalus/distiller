from sklearn.feature_extraction.text import TfidfVectorizer
from colorama import Fore, Style

class TFIDFHelper:
    def __init__(self, corpus = [], min_tfidf_score = 0.01, min_ngrams=1, max_ngrams=6):
        self.corpus = corpus
        self.min_tfidf_score = min_tfidf_score
        self.min_ngrams = min_ngrams
        self.max_ngrams = max_ngrams
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        self.tfidf_vectorizer.fit(self.corpus)

        print(Fore.BLUE +  f"[!] TFIDFHelper initialized with: corpus_len: {len(corpus)}, min_tfidf_score: {min_tfidf_score}, min_ngrams: {min_ngrams}, max_ngrams: {max_ngrams}" + Style.RESET_ALL)

    def get_tfidf_scores(self, text: str):
        text = text.lower().replace("\n", "")
        tfidf_matrix = self.tfidf_vectorizer.transform([text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        scores = zip(feature_names, tfidf_matrix.toarray().flatten())
        return {ngram: score for ngram, score in scores if score > self.min_tfidf_score}

    def all_ngrams(self, text: str):
        text = text.lower().replace("\n", "")
        tfidf_scores = self.get_tfidf_scores(text)
        words = text.split()
        ngrams = set()
        for n in range(min(self.max_ngrams, len(words)), self.min_ngrams - 1, -1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                ngrams.add(ngram)

        scored_ngrams = [(ngram, tfidf_scores.get(ngram, 0)) for ngram in ngrams]
        scored_ngrams.sort(key=lambda x: x[1], reverse=True)

        for ngram, score in scored_ngrams:
            if score >= self.min_tfidf_score or len(scored_ngrams) < self.min_ngrams:
                yield ngram

"""
# Test
helper = TFIDFHelper()
list(helper.all_ngrams("the cat sat on the mat"))
"""

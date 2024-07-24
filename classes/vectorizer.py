import numpy as np

from navec import Navec
from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    def __init__(self, embeddings_path, dim=300):
        self.vectorizer = Navec.load(embeddings_path)
        self.tfidf = TfidfVectorizer()
        self.dim = dim

    def fit_tfidf(self, texts):
        self.tfidf.fit(texts)
        self.idf_dict = dict(zip(self.tfidf.get_feature_names_out(), self.tfidf.idf_))

    def get_embedding(self, text):
        vectors = []

        for word in text:
            if word in self.tfidf.get_feature_names_out() and word in self.vectorizer:
                vectors.append(self.vectorizer[word] * self.idf_dict[word])

        if len(vectors) == 0:
            return np.zeros(self.dim)

        return np.mean(vectors, axis=0)

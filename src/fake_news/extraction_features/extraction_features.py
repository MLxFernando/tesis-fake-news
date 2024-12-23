import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    def __init__(self, max_features=10000, max_df=0.95, min_df=5, ngram_range=(1, 3)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df, min_df=min_df, ngram_range=ngram_range)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, path):
        joblib.dump(self.vectorizer, path)
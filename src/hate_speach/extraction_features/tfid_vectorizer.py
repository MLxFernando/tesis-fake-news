from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
class TfidfFeatureExtractor:
    """
    Clase para extracción de características usando TF-IDF.
    """
    def __init__(self, max_features=10000, ngram_range=(1, 2), max_df=0.75, min_df=5):
        """
        Inicializa el vectorizador TF-IDF.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df
        )
        self.tfidf_features = None

    def extract_features(self, texts):
        """
        Extrae características TF-IDF de un conjunto de textos.

        Args:
            texts (pd.Series or list): Lista o serie de textos procesados.

        Returns:
            scipy.sparse.csr.csr_matrix: Matriz TF-IDF.
        """
        self.tfidf_features = self.vectorizer.fit_transform(texts)
        return self.tfidf_features
    
    def transform(self, texts, vectorizer):
        """
        Transforma un conjunto de textos en características TF-IDF.

        Args:
            texts (pd.Series or list): Lista o serie de textos procesados.
            vectorizer (TfidfVectorizer): Vectorizador TF-IDF.

        Returns:
            scipy.sparse.csr.csr_matrix: Matriz TF-IDF.
        """
        return vectorizer.transform(texts)

    def save_vectorizer(self, path):
        """
        Guarda el vectorizador TF-IDF en disco.

        Args:
            path (str): Ruta donde se guardará el vectorizador.
        """
        joblib.dump(self.vectorizer, path)

    def get_feature_names(self):
        """
        Devuelve los nombres de las características extraídas.

        Returns:
            list: Lista de nombres de características TF-IDF.
        """
        return self.vectorizer.get_feature_names_out()

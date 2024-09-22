import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

def extract_features(text_data, max_features=None):
    """
    Extrae características de los textos utilizando TF-IDF.

    :param text_data: Lista o Serie de textos a vectorizar.
    :param max_features: Número máximo de características (opcional).
    :return: Matriz TF-IDF, vectorizador entrenado.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(text_data)

    return X_tfidf, vectorizer

def select_best_features(X, y, k=5000):
    """
    Selecciona las mejores características utilizando chi-cuadrado.

    :param X: Matriz de características.
    :param y: Etiquetas correspondientes.
    :param k: Número de mejores características a seleccionar.
    :return: Matriz con las mejores características, selector entrenado.
    """
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)

    return X_new, selector

def save_vectorizer_and_selector(vectorizer, selector, vectorizer_filename='tfidf_vectorizer.pkl', selector_filename='selector.pkl'):
    """
    Guarda tanto el vectorizador como el selector en archivos .pkl.

    :param vectorizer: Vectorizador entrenado.
    :param selector: Selector entrenado.
    :param vectorizer_filename: Nombre del archivo para guardar el vectorizador.
    :param selector_filename: Nombre del archivo para guardar el selector.
    """
    joblib.dump(vectorizer, vectorizer_filename)
    joblib.dump(selector, selector_filename)

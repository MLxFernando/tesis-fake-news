import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
class ExtactionFeatures:
    def __init__(self, num_features  =5000):
        self.num_features = num_features
        self.vectorizer = TfidfVectorizer()
        self.selector = None

        pass
    def extract_features(self,text_data, labels):
        """
        Extrae características de los textos utilizando TF-IDF.

        :param text_data: Lista o Serie de textos a vectorizar.
        :param max_features: Número máximo de características (opcional).
        :return: Matriz TF-IDF, vectorizador entrenado.
        """
        x = self.vectorizer.fit_transform(text_data)
        X_new = self.select_best_features(x, labels)
        feature_names = self.vectorizer.get_feature_names_out()
        selected_feature_names = [feature_names[i] for i in self.selector.get_support(indices=True)]
        tfidt_df = pd.DataFrame(X_new.toarray(), columns=selected_feature_names)
        return tfidt_df, self.vectorizer, self.selector


    def select_best_features(self, X, y):
        """
        Selecciona las mejores características utilizando chi-cuadrado.

        :param X: Matriz de características.
        :param y: Etiquetas correspondientes.
        :param k: Número de mejores características a seleccionar.
        :return: Matriz con las mejores características, selector entrenado.
        """
        self.selector = SelectKBest(chi2, k=self.num_features)
        X_new = self.selector.fit_transform(X, y)
        return X_new


        

    def save_vectorizer_and_selector(self,vectorizer, selector, vectorizer_filename='tfidf_vectorizer.pkl', selector_filename='selector.pkl'):
        """
        Guarda tanto el vectorizador como el selector en archivos .pkl.

        :param vectorizer: Vectorizador entrenado.
        :param selector: Selector entrenado.
        :param vectorizer_filename: Nombre del archivo para guardar el vectorizador.
        :param selector_filename: Nombre del archivo para guardar el selector.
        """

        joblib.dump(vectorizer, vectorizer_filename)
        joblib.dump(selector, selector_filename)

import numpy as np
from textstat import textstat
import pandas as pd

class AdditionalFeatureExtractor:
    """
    Clase para extraer características adicionales basadas en legibilidad, sílabas y métricas textuales.
    """
    def __init__(self):
        """
        Inicialización de la clase sin parámetros adicionales.
        """
        pass

    def additional_features(self, tweet):
        """
        Calcula características adicionales para un texto.

        Args:
            tweet (str): Texto del tweet o documento.

        Returns:
            list: Lista de características adicionales calculadas.
        """
        # Contar sílabas en el texto
        syllables = textstat.syllable_count(tweet)

        # Caracteres en palabras y caracteres totales
        num_chars = sum(len(w) for w in tweet)
        num_chars_total = len(tweet)

        # Número total de palabras
        num_words = len(tweet.split())

        # Promedio de sílabas por palabra
        avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)

        # Número de términos únicos en el texto
        num_unique_terms = len(set(tweet.split()))

        # Flesch–Kincaid Readability Score (FRE)
        FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

        # Flesch–Kincaid Grade Level (FKRA)
        FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)

        # Lista de características calculadas
        add_features = [
            FKRA,  # Flesch–Kincaid Grade Level
            FRE,  # Flesch Reading Ease
            syllables,  # Número de sílabas
            avg_syl,  # Promedio de sílabas por palabra
            num_chars,  # Número total de caracteres en palabras
            num_chars_total,  # Número total de caracteres
            num_words,  # Número de palabras
            num_unique_terms  # Número de términos únicos
        ]
        return add_features

    def extract_features(self, tweets):
        """
        Extrae características adicionales para una lista de textos.

        Args:
            tweets (list or pd.Series): Lista o serie de textos.

        Returns:
            np.ndarray: Matriz con las características calculadas para cada texto.
        """
        features = [self.additional_features(tweet) for tweet in tweets]
        return np.array(features)

    def get_feature_dataframe(self, tweets):
        """
        Devuelve un DataFrame con las características adicionales.

        Args:
            tweets (list or pd.Series): Lista o serie de textos.

        Returns:
            pd.DataFrame: DataFrame con las características adicionales calculadas.
        """
        features_array = self.extract_features(tweets)
        return pd.DataFrame(features_array, columns=[
            'FKRA', 'FRE', 'Syllables', 'Avg_Syl', 'Num_Chars', 'Num_Chars_Total', 'Num_Words', 'Num_Unique_Terms'
        ])

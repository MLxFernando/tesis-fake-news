import numpy as np
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS

class SentimentFeatureExtractor:
    """
    Clase para extraer características basadas en análisis de sentimientos y métricas de Twitter.
    """
    def __init__(self):
        """
        Inicializa el analizador de sentimientos.
        """
        self.sentiment_analyzer = VS()
        self.sentiment_features = None

    def count_tags(self, tweet):
        """
        Cuenta URLs, menciones y hashtags en el texto.

        Args:
            tweet (str): Texto del tweet.

        Returns:
            tuple: Conteo de URLs, menciones y hashtags.
        """
        space_pattern = r'\s+'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        mention_pattern = r'@[\w\-]+'
        hashtag_pattern = r'#[\w\-]+'
        parsed_text = re.sub(space_pattern, ' ', tweet)
        parsed_text = re.sub(url_pattern, 'URLHERE', parsed_text)
        parsed_text = re.sub(mention_pattern, 'MENTIONHERE', parsed_text)
        parsed_text = re.sub(hashtag_pattern, 'HASHTAGHERE', parsed_text)
        return parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE')

    def sentiment_analysis(self, tweet):
        """
        Analiza el sentimiento de un texto y calcula métricas adicionales.

        Args:
            tweet (str): Texto del tweet.

        Returns:
            list: Características de sentimientos y métricas.
        """
        sentiment = self.sentiment_analyzer.polarity_scores(tweet)
        twitter_objs = self.count_tags(tweet)
        features = [
            sentiment['neg'], 
            sentiment['pos'], 
            sentiment['neu'], 
            sentiment['compound'],
            twitter_objs[0], 
            twitter_objs[1], 
            twitter_objs[2]
        ]
        return features

    def extract_features(self, tweets):
        """
        Extrae características de análisis de sentimientos para un conjunto de datos.

        Args:
            tweets (pd.Series or list): Lista o serie de tweets/textos.

        Returns:
            np.ndarray: Matriz de características de sentimientos y métricas.
        """
        self.sentiment_features = np.array([self.sentiment_analysis(tweet) for tweet in tweets])
        return self.sentiment_features

    def get_feature_dataframe(self, tweets):
        """
        Devuelve un DataFrame con las características extraídas.

        Args:
            tweets (pd.Series or list): Lista o serie de tweets/textos.

        Returns:
            pd.DataFrame: DataFrame con las características de análisis de sentimientos.
        """
        features_array = self.extract_features(tweets)
        return pd.DataFrame(features_array, columns=[
            'Neg', 'Pos', 'Neu', 'Compound', 'url_tag', 'mention_tag', 'hash_tag'
        ])

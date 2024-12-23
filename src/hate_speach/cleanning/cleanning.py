import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import contractions

stopwords = nltk.corpus.stopwords.words("english")
# Opcional: Excluir ciertas palabras de las stopwords si son importantes
important_words = ["doesn't", "not", "do"]
stopwords = [word for word in stopwords if word not in important_words]

lemmatizer = WordNetLemmatizer()

class CleanningData:
    def __init__(self, data):
        self.data = data
    
    def cleaning_data(self):
        # Expandir contracciones
        data_cleaning = self.data.apply(lambda x: contractions.fix(x))

        # Eliminar espacios en blanco
        data_cleaning = data_cleaning.str.replace(r'\s+', ' ', regex=True)

        # Eliminar menciones a usuarios
        data_cleaning = data_cleaning.str.replace(r'@[\w\-]+', '', regex=True)

        # Eliminar URLs
        data_cleaning = data_cleaning.str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', regex=True)

        # Eliminar caracteres especiales pero conservar contracciones
        data_cleaning = data_cleaning.str.replace(r'[^\w\s\'-]', '', regex=True)

        # Convertir a minúsculas
        data_cleaning = data_cleaning.str.lower()

        # Tokenizar el texto
        data_cleaning = data_cleaning.apply(lambda x: x.split())

        # Eliminar stopwords
        data_cleaning = data_cleaning.apply(lambda x: [word for word in x if word not in stopwords])

        # Lematización
        data_cleaning = data_cleaning.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

        # Unir el texto
        data_cleaning = data_cleaning.apply(lambda x: ' '.join(x))

        return data_cleaning

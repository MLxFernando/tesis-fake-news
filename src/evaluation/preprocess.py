import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios para NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Inicializar el lematizador y las stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Diccionario de contracciones y sus expansiones correspondientes
contractions_dict = {
    "i'm": "i am", "he's": "he is", "she's": "she is", "that's": "that is",
    "what's": "what is", "where's": "where is", "won't": "will not",
    "can't": "cannot", "n't": " not", "it's": "it is", "i've": "i have",
    "you're": "you are", "they're": "they are", "we're": "we are",
    "i'll": "i will", "he'll": "he will", "she'll": "she will", "we'll": "we will",
    "they'll": "they will", "i'd": "i would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would"
}

# Función para expandir contracciones
def expand_contractions(text, contractions_dict=contractions_dict):
    contractions_pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    expanded_text = contractions_pattern.sub(lambda x: contractions_dict[x.group()], text)
    return expanded_text

# Función de preprocesamiento del texto
def preprocess_text(text):
    """
    Preprocesa el texto expandiendo contracciones, lematizando y eliminando stopwords.

    :param text: Texto a preprocesar.
    :return: Texto preprocesado.
    """
    # Expandir contracciones
    text = expand_contractions(text)

    # Tokenizar el texto en palabras
    tokens = nltk.word_tokenize(text)

    # Lematizar cada token y eliminar stopwords
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    # Unir los tokens en un solo string preprocesado
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Función para preprocesar un DataFrame (sobreescribe la columna 'text')
def preprocess_data(df):
    """
    Aplica el preprocesamiento a un DataFrame, sobrescribiendo la columna de texto.

    :param df: DataFrame que contiene los datos ya limpiados.
    :return: DataFrame con el texto preprocesado.
    """
    df['text'] = df['text'].apply(preprocess_text)  # Sobrescribir la columna 'text'
    return df

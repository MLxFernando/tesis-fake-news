import re
import pandas as pd

# Función para detectar palabras con números
def contains_number(word):
    return any(char.isdigit() for char in word)

# Función para limpiar texto
def clean_text(text):
    """
    Limpia el texto eliminando caracteres especiales, palabras con números y realiza otras tareas de limpieza.

    :param text: Texto a limpiar.
    :return: Texto limpio.
    """
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar comillas, caracteres especiales, puntos, comas, @ y letras solitarias
    text = re.sub(r'[“”‘’"\'\-\–\—•.,@]|(?<!\w)\b\w\b(?!\w)', ' ', text)
    
    # Eliminar palabras que contengan números (si es necesario)
    cleaned_text = ' '.join([word for word in text.split() if not contains_number(word)])
    
    return cleaned_text

# Función para eliminar nulos y hacer muestreo si es necesario
def clean_data(df, sample_size=None):
    """
    Limpia el DataFrame eliminando filas con datos nulos y, opcionalmente, realizando un muestreo.
    
    :param df: DataFrame con los datos originales.
    :param sample_size: Si se especifica, se toma una muestra del DataFrame.
    :return: DataFrame limpio.
    """
    # Eliminar filas con valores nulos en la columna 'text'
    print("Limpiando texto")
    df = df.dropna(subset=['text'])
    # Si se proporciona un tamaño de muestra, realizar el muestreo
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    return df

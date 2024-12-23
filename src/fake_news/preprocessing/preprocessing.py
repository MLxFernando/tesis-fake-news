import pandas as pd
import sys
import os
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor


load_dotenv()

# Agregar el directorio raíz al PYTHONPATH
sys.path.append(os.getenv("PYTHONPATH"))
from src.fake_news.cleaning.cleanning import CleannedText
from src.fake_news.extraction_features.extraction_features import FeatureExtractor
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, cleaner, feature_extractor):
        self.cleaner = cleaner
        self.feature_extractor = feature_extractor

    def preprocess_data(self, path, path_processed, path_vectorizer):
        # Leer los datos
        data = pd.read_csv(path)
        data = data.dropna()  # Eliminar filas nulas
        #Eliminar columnas innecesarias
        data = data.drop(columns=['title', 'author', 'id'])
        print("Limpiando Dataset")
        with ProcessPoolExecutor(max_workers=14) as executor:
            # Procesamos el texto en paralelo
            data['text'] = list(executor.map(cleaner.clean_text, data['text']))
        #Separar los datos en entrenamiento y prueba
        print("Separando datos")
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=0)

        # Extraer características con TfidfVectorizer
        print("Extrayendo características")
        X_train = self.feature_extractor.fit_transform(X_train)
        X_test = self.feature_extractor.transform(X_test)
        # Guardar características y etiquetas
        print("Guardando características")
        self.save_features(X_train, y_train, path_processed)
        self.save_features(X_test, y_test, path_processed.replace('train', 'test'))
        print("Guardando vectorizador")
        self.feature_extractor.save_vectorizer(path_vectorizer)
        print("Proceso de preprocesamiento completado")
        #Retornar tabla de características y etiquetas
        

    def parallel_clean(self, texts):
        for x in range(len(texts)):
            texts[x] = self.cleaner.clean_text(texts[x])
        return texts
        
    def save_features(self, X, y, path):
        data = pd.DataFrame(X.toarray())
        data['label'] = y
        data.to_csv(path, index=False)


if __name__ == '__main__':
    
    cleaner = CleannedText()
    feature_extractor = FeatureExtractor()
    preprocessor = Preprocessor(cleaner, feature_extractor)
    preprocessor.preprocess_data('data/fake_new/raw/dataset.csv', 'data/fake_new/processed/train.csv', 'data/fake_new/model/vectorizer.pkl')
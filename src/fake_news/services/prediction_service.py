

import sys
import os
from dotenv import load_dotenv
import joblib
import pandas as pd

load_dotenv()

# Agregar el directorio raíz al PYTHONPATH
sys.path.append(os.getenv("PYTHONPATH"))
from src.fake_news.utils.constants import FAKE_NEWS_MODEL_PATH, FAKE_NEWS_VECTORIZER_PATH, FAKE_NEWS_TEST_PATH

from src.fake_news.cleaning.cleanning import CleannedText


class PredictionService:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def load_model(self):
        self.model = joblib.load(FAKE_NEWS_MODEL_PATH)
        self.vectorizer = joblib.load(FAKE_NEWS_VECTORIZER_PATH)

    def preprocess_text(self, text):
        cleaner = CleannedText()
        text = cleaner.clean_text(text)
        return self.vectorizer.transform([text])

    # def predict(self, text: str) -> str:
    #     data = self.preprocess_text(text)
    #     prediction = self.model.predict(data)
    #     print(prediction)
    def predict(self, text: str) -> dict:
        data = self.preprocess_text(text)
        prediction = self.model.predict(data)  # Predicción binaria: [0] o [1]
        predicted_class = "False" if prediction[0] == 1 else "True"
        return {
            "predicted_class": predicted_class,
            "raw_output": int(prediction[0]) # Opcional, incluye el valor numérico original
        }



if __name__ == '__main__':
    prediction_service = PredictionService()
    prediction_service.load_model()
    

    data = pd.read_csv(FAKE_NEWS_TEST_PATH)
    #obtener la columna texto fila 20974
    text = data.loc[data['id'] == 20800, 'text'].iloc[0]
    
    prediction_service.predict(text)


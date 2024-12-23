

import sys
import os
from dotenv import load_dotenv
import joblib
import pandas as pd

load_dotenv()

# Agregar el directorio raíz al PYTHONPATH
sys.path.append(os.getenv("PYTHONPATH"))
from src.hate_speach.utils.constants import HATE_SPEACH_TEST_PATH, HATE_SPEACH_MODEL_PATH

from src.hate_speach.preprocessing.preprocess import PreprocessData


class PredictionService:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def load_model(self):
        self.model = joblib.load(HATE_SPEACH_MODEL_PATH)

    def preprocess_text(self, text):
        preprocecer = PreprocessData()
        preprocecer.cleanning_and_preprocessing_data_for_predict(text)
        data_preproceced = preprocecer.combine_feature_for_predict()
        
        return data_preproceced

    # def predict(self, text: str) -> str:
    #     data = self.preprocess_text(text)
    #     prediction = self.model.predict(data)
    #     print(prediction)
    def predict(self, text: str) -> dict:
        data = self.preprocess_text(text)
        prediction = self.model.predict(data)  # Predicción categórica: [0, 1, 2, 3]
        probabilities = self.model.predict_proba(data)[0]  # Probabilidades por clase (si el modelo las soporta)

        # Mapear clases a nombres legibles
        class_mapping = {
            0: "Ninguno",
            1: "Lenguaje ofensivo",
            2: "Discurso de odio",
            3: "Otra clase"  # Modifica según lo que represente.
        }

        predicted_class = class_mapping.get(prediction[0], "Desconocido")
        return {
            "predicted_class": class_mapping.get(int(prediction[0]), "Desconocido"),  # Asegúrate de convertir aquí también
            "probabilities": probabilities.tolist(),  # Esto ya es compatible con JSON
            "raw_output": int(prediction[0])  # Convierte a int nativo
        }



if __name__ == '__main__':
    prediction_service = PredictionService()
    prediction_service.load_model()
    

    data = pd.read_csv(HATE_SPEACH_TEST_PATH)
    #obtener la columna texto fila 20974
    text = data.loc[data['id'] == 111, 'tweet'].iloc[0]
    prediction_service.predict(text)


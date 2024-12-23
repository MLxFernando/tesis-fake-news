from flask import Flask, request, jsonify


import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Agregar el directorio raíz al PYTHONPATH
sys.path.append(os.getenv("PYTHONPATH"))

from src.fake_news.services.prediction_service import PredictionService as FakeNewsPredictionService
from src.hate_speach.services.prediction_service import PredictionService as HateSpeechPredictionService


class FakeNewsAPI:
    def __init__(self):
        self.service = FakeNewsPredictionService()
        self.service.load_model()

    def predict(self, text):
        return self.service.predict(text)


class HateSpeechAPI:
    def __init__(self):
        self.service = HateSpeechPredictionService()
        self.service.load_model()

    def predict(self, text):
        # result = self.service.predict(text)
        # print(result)
        # return {
        #     "predicted_class": result['predicted_class'],
        #     "probabilities": result['probabilities']
        # }
        return self.service.predict(text)

class FlaskServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.fake_news_api = FakeNewsAPI()
        self.hate_speech_api = HateSpeechAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/predict', methods=['POST'])
        def predict():
            # Verificar si el texto está presente en la solicitud
            data = request.get_json()
            if 'text' not in data:
                return jsonify({'error': 'El campo "text" es obligatorio'}), 400
            
            text = data['text']

            # Predicción de Fake News
            fake_news_prediction = self.fake_news_api.predict(text)

            # Predicción de Discurso de Odio
            hate_speech_prediction = self.hate_speech_api.predict(text)

            # Respuesta combinada
            response = {
                "fake_news_prediction": fake_news_prediction,
                "hate_speech_prediction": hate_speech_prediction
            }

            return jsonify(response)

    def run(self, debug=True):
        self.app.run(debug=debug)


if __name__ == '__main__':
    server = FlaskServer()
    server.run()

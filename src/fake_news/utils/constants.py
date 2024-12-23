import os
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

FAKE_NEWS_TEST_PATH = os.getenv("FAKE_NEWS_TEST_PATH")
FAKE_NEWS_MODEL_PATH = os.getenv("FAKE_NEWS_MODEL_PATH")
FAKE_NEWS_VECTORIZER_PATH = os.getenv("FAKE_NEWS_VECTORIZER_PATH")


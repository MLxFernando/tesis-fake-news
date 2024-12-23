import os
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

# Cargar la ruta del archivo HATE_SPEACH_PATH
HATE_SPEACH_PATH = os.getenv("HATE_SPEACH_PATH")
HATE_SPEACH_PROCESSED_PATH = os.getenv("HATE_SPEACH_PROCESSED_PATH")
HATE_SPEACH_VECTORIZER_PATH = os.getenv("HATE_SPEACH_VECTORIZER_PATH")
HATE_SPEACH_MODEL_PATH = os.getenv("HATE_SPEACH_MODEL_PATH")
HATE_SPEACH_TEST_PATH = os.getenv("HATE_SPEACH_TEST_PATH")

# Depuración: verifica si se cargó correctamente
if HATE_SPEACH_PATH is None:
    raise ValueError("Error: HATE_SPEACH_PATH no está definido en el archivo .env")
else:
    print(f"HATE_SPEACH_PATH cargado correctamente: {HATE_SPEACH_PATH}")

if HATE_SPEACH_PROCESSED_PATH is None:
    raise ValueError("Error: HATE_SPEACH_PROCESSED_PATH no está definido en el archivo .env")
else:
    print(f"HATE_SPEACH_PROCESSED_PATH cargado correctamente: {HATE_SPEACH_PROCESSED_PATH}")
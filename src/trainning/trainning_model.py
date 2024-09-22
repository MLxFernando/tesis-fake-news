import time
import joblib
from sklearn.metrics import accuracy_score

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa un modelo dado usando el método normal de train/test split.

    :param model: Modelo a entrenar.
    :param model_name: Nombre del modelo.
    :param X_train: Datos de entrenamiento vectorizados.
    :param X_test: Datos de prueba vectorizados.
    :param y_train: Etiquetas de entrenamiento.
    :param y_test: Etiquetas de prueba.
    :return: Nombre del modelo, modelo entrenado, tiempo de ejecución y precisión en el conjunto de prueba.
    """
    start_time = time.time()

    # Entrenar el modelo
    model.fit(X_train, y_train)

    end_time = time.time()
    execution_time = end_time - start_time

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy for {model_name}: {accuracy:.2f}')
    print(f'Time taken for {model_name}: {execution_time:.2f} seconds')

    return model_name, model, execution_time, accuracy

def save_model(model, model_name):
    """
    Guarda el modelo entrenado en un archivo .pkl.

    :param model: Modelo entrenado.
    :param model_name: Nombre del archivo donde se guardará el modelo.
    """
    joblib.dump(model, f'{model_name}_model.pkl')

def load_model(model_path):
    """
    Carga un modelo previamente guardado desde un archivo .pkl.

    :param model_path: Ruta del archivo del modelo.
    :return: Modelo cargado.
    """
    return joblib.load(model_path)

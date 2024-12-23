from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
#from sklearn.externals import joblib
import pandas as pd
import sys
import os
from dotenv import load_dotenv


load_dotenv()

# Agregar el directorio raíz al PYTHONPATH
sys.path.append(os.getenv("PYTHONPATH"))
print("PYTHONPATH:", os.getenv("PYTHONPATH"))
from src.hate_speach.utils.constants import HATE_SPEACH_PROCESSED_PATH, HATE_SPEACH_MODEL_PATH

class TrainningModel:
    def __init__(self):
        self.x = None
        self.y = None
        self.scaler = None
        self.model = None

    def read_preprocessed_data(self, path):
        df = pd.read_csv(path)
        self.x = df.drop(columns=['class'], axis=1).values
        self.y = df['class'].values

    def train_and_evaluate_model(self):
        # Separar los datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)


        # Entrenar el modelo
        self.model = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
        self.model.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, average='weighted')
        recall = recall_score(y_test, y_pred, pos_label=1, average='weighted')
        f1 = f1_score(y_test, y_pred, pos_label=1, average='weighted')


        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1: {f1:.2f}')
    
    def save_model(self, model_name):
        """
        Guarda el modelo entrenado y el scaler en archivos .pkl.
        """
        joblib.dump(self.model, model_name)


if __name__ == "__main__":
    model = TrainningModel()
    model.read_preprocessed_data(HATE_SPEACH_PROCESSED_PATH)
    model.train_and_evaluate_model()
    model.save_model(HATE_SPEACH_MODEL_PATH)
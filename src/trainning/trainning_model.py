import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd

class TrainningModel:

    def __init__(self):
        self.x = None
        self.y = None
        self.scaler = None
        self.model = None

    def read_preprocessed_data(self, path):
        df = pd.read_csv(path)
        self.x = df.drop(columns=['result'], axis=1).values
        self.y = df['result'].values

    def train_and_evaluate_model(self):
        # Separar los datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # Aplicar el escalado solo al conjunto de entrenamiento
        self.scaler = StandardScaler(with_mean=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Entrenar el modelo
        self.model = RandomForestClassifier(n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)

        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')

    def save_model(self, model_name, scaler_name):
        """
        Guarda el modelo entrenado y el scaler en archivos .pkl.
        """
        joblib.dump(self.model, f'data/{model_name}_model.pkl')
        joblib.dump(self.scaler, f'data/scaler.pkl')

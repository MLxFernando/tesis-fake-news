
import joblib
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd

class TrainningModel:

    def __init__(self):
        self.pd = None
        self.scaler = None


    def read_preprocessed_data(self, path):
        self.pd = pd.read_csv(path)
        self.x = self.pd.drop(columns=['result'], axis=1).values
        self.y = self.pd['result'].values

        self.scaler = StandardScaler(with_mean=False)
        self.x = self.scaler.fit_transform(self.x)
    


    def train_and_evaluate_model(self,):
        
        #Separar los datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # Entrenar el modelo
        self.model = RandomForestClassifier( n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precition = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precition:.2f}')
        print(f'Recall: {recall:.2f}')

    def save_model(self, model_name, scaler_name):
        """
        Guarda el modelo entrenado en un archivo .pkl.
        :param model_name: Nombre del archivo donde se guardará el modelo.
        """
        joblib.dump(self.model, f'data/{model_name}_model.pkl')
        """
        Guardar Scaler 

        :param model_name: Nombre del archivo donde se guardará el modelo.
        """
        joblib.dump(self.scaler, f'data/scaler.pkl')

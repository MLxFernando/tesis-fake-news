import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class PreprocessText:
    def __init__(self):
        self.stop_words = list(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.texts = []

    def remove_characters_special(self, text):
        return re.sub(r'[^a-zA-Z\s]', '', text)
    
    def to_lower(self, text):
        return text.lower()
    
    def tokenize(self, text):
        return nltk.word_tokenize(text)
    
    def remove_stop_words(self, texts):
        for y in texts:
            if y not in self.stop_words:
                self.texts.append(self.lemmatizer.lemmatize(y))

    def join_text(self):
        return ' '.join(self.texts)
    
    def preprocess(self, text):
        text = self.remove_characters_special(text)
        text = self.to_lower(text)
        texts = self.tokenize(text)
        self.remove_stop_words(texts)
        text = self.join_text()
        return text

class PreprocessData:
    def __init__(self):
        self.data = None
        self.preprocess_text = PreprocessText()
    
    def read_data(self, path):
        self.data = pd.read_csv(path)
        # Eliminar las columnas que no se necesitan para el entrenamiento
        self.data = self.data.drop(columns=['title', 'author', 'id'])
        # Eliminar las filas con valores nulos
        self.data = self.data.dropna()

    def preprocess_data(self):
        # Usamos todos los núcleos disponibles
        with ProcessPoolExecutor(max_workers=14) as executor:
            # Procesamos el texto en paralelo
            self.data['text'] = list(executor.map(self.preprocess_text.preprocess, self.data['text']))
        return self.data
    
    def split_and_vectorize_data(self, save_vectorizer_path=None):
        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(self.data['text'], self.data['label'], test_size=0.2, random_state=0)
        
        # Crear y ajustar el vectorizador solo en los datos de entrenamiento
        vectorizer = TfidfVectorizer(max_features=10000, max_df=0.95, min_df=5, ngram_range=(1, 3))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Guardar el vectorizador si se especifica una ruta
        if save_vectorizer_path:
            with open(save_vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
                
        return X_train_tfidf, X_test_tfidf, y_train, y_test

class ModelTrainer:
    def __init__(self, vectorizer_path=None):
        self.vectorizer = None
        if vectorizer_path:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

    def train_model_Logistic(self, X_train, X_test, y_train, y_test):
        model_logistic = LogisticRegression()
        model_logistic.fit(X_train, y_train)
        y_pred = model_logistic.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("Model Logistic Regression")
        print(f"Model Accuracy: {accuracy}")
        print(f"Model Precision: {precision}")
        print(f"Model Recall: {recall}")
        print(f"Model F1 Score: {f1}")
        return model_logistic
    
    def train_model_Ramdom(self, X_train, X_test, y_train, y_test):
        model_Ramdom = RandomForestClassifier(n_jobs=-1)
        model_Ramdom.fit(X_train, y_train)
        y_pred = model_Ramdom.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("Model Random Forest")
        print(f"Model Accuracy: {accuracy}")
        print(f"Model Precision: {precision}")
        print(f"Model Recall: {recall}")
        print(f"Model F1 Score: {f1}")
        return model_Ramdom
    
    def train_model_SVC(self, X_train, X_test, y_train, y_test):
        model_svm = SVC()
        model_svm.fit(X_train, y_train)
        y_pred = model_svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("Model Support Vector Machine")
        print(f"Model Accuracy: {accuracy}")
        print(f"Model Precision: {precision}")
        print(f"Model Recall: {recall}")
        print(f"Model F1 Score: {f1}")
        return model_svm
    
    def  save_model(self, model):
        joblib.dump(model, f'data/fake_new/model/fake_news_model.pkl')
        
    
       



if __name__ == '__main__':
    # Preprocesamiento y vectorización
    preprocess_data = PreprocessData()
    preprocess_data.read_data("data/fake_new/raw/dataset.csv")
    preprocess_data.preprocess_data()
    X_train_tfidf, X_test_tfidf, y_train, y_test = preprocess_data.split_and_vectorize_data("data/fake_new/model/vectorizer.pkl")
    print("Preprocesamiento y vectorización completados.")
    # Entrenamiento del modelo
    model_trainer = ModelTrainer(vectorizer_path="data/fake_new/model/vectorizer.pkl")
    model = model_trainer.train_model_Logistic(X_train_tfidf, X_test_tfidf, y_train, y_test)
    # model = model_trainer.train_model_Ramdom(X_train_tfidf, X_test_tfidf, y_train, y_test)
    # model = model_trainer.train_model_SVC(X_train_tfidf, X_test_tfidf, y_train, y_test)
    model_trainer.save_model(model)
    

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pandas as pd

class ModelTrainer:
    def __init__(self ):
        pass
    
    def read_data(self, path_train, path_test):
        df_train = pd.read_csv(path_train)
        df_test = pd.read_csv(path_test)
        self.x_train = df_train.drop(columns=['label'], axis=1).values
        self.y_train = df_train['label'].values
        self.x_test = df_test.drop(columns=['label'], axis=1).values
        self.y_test = df_test['label'].values



    def train_model_Logistic(self):
        model_logistic = LogisticRegression()
        model_logistic.fit(self.x_train, self.y_train)
        y_pred = model_logistic.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        print("Model Logistic Regression")
        print(f"Model Accuracy: {accuracy}")
        print(f"Model Precision: {precision}")
        print(f"Model Recall: {recall}")
        print(f"Model F1 Score: {f1}")

        


        return model_logistic
    
    def  save_model(self, model):
        joblib.dump(model, f'data/fake_new/model/fake_news_model.pkl')
        
    
       



if __name__ == '__main__':

    # Entrenamiento del modelo
    model_trainer = ModelTrainer()
    model_trainer.read_data("data/fake_new/processed/train.csv", "data/fake_new/processed/test.csv")
    model = model_trainer.train_model_Logistic()
    model_trainer.save_model(model)


    
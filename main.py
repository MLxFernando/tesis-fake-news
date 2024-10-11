import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from src.trainning.trainning_model import TrainningModel
from src.cleaning.cleanning import clean_data, clean_text
from src.evaluation.preprocess import preprocess

def main():
    # Cargar los datos originales
    raw_df = None
    cleaned_df = None
    opcion = 0
    # Limpieza de datos (modifica directamente la columna 'text')
    
    while(opcion != 6):
        print("Procedimiento para entrenar modelo de machine learning")
        print("1. Leer los datos\n")
        print("2. Limpiar los datos\n")
        print("3. Preprocesar los datos\n")
        print("4. Entrenar el modelo\n")
        print("5. Evaluar el modelo\n")
        print("6. Salir\n")
        opcion = input("Seleccione una opcion: ")
        if opcion == "1":
            print("Leyendo los datos")
            raw_df = pd.read_csv('data/raw/dataset_reducido.csv')
        elif opcion == "2":
            print("Limpiando los datos")
            cleaned_df = limpiar_dataset(raw_df)
        elif opcion == "3":
            print("Preprocesando los datos")
            preprocesar_dataset("data/processed/data_cleaned_output.csv")
        elif opcion == "4":
            print("Entrenando el modelo")
            trainig_model("data/processed/data_preprocessed_final.csv")


def limpiar_dataset(raw_df):
    # Limpiar los datos y aplicar transformaciones al texto
    cleaned_df = clean_data(raw_df)
    cleaned_df.loc[:, 'text'] = cleaned_df['text'].apply(clean_text)

    # Eliminar la columna 'title' ya que no se usará en el entrenamiento
    cleaned_df = cleaned_df.drop(columns=['title'], errors='ignore')
    # Guardar el dataset limpio en un archivo CSV sin incluir el índice
    cleaned_df.to_csv("data/processed/data_cleanned_final.csv", index=False)
    return cleaned_df



def preprocesar_dataset(path):
    preprocess(path)


def trainig_model(preproces_data):
    model = TrainningModel()
    model.read_preprocessed_data(preproces_data)
    model.train_and_evaluate_model()
    model.save_model('Random_Forest', 'scaler')


if __name__ == "__main__":
    main()



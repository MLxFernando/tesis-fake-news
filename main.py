import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import concurrent.futures
import joblib
from src.evaluation.feature_extraction import extract_features, select_best_features, save_vectorizer_and_selector
from src.trainning.trainning_model import train_and_evaluate_model, save_model, load_model
from src.cleaning.cleanning import clean_data, clean_text
from src.evaluation.preprocess import preprocess_data

def main():
    # Cargar los datos originales
    raw_df = pd.read_csv('data/raw/dataset_final_combinado.csv')

    # Limpieza de datos (modifica directamente la columna 'text')
    cleaned_df = clean_data(raw_df,sample_size=60000)
    cleaned_df['text'] = cleaned_df['text'].apply(clean_text)

    # Eliminar la columna 'title' ya que no se usará en el entrenamiento
    cleaned_df = cleaned_df.drop(columns=['title'], errors='ignore')

    # Preprocesamiento (modifica directamente la columna 'text')
    preprocessed_df = preprocess_data(cleaned_df)

    # Vectorización de los textos usando TF-IDF
    X_tfidf, vectorizer = extract_features(preprocessed_df['text'])

    # Obtener las etiquetas
    y = preprocessed_df['label']

    # Seleccionar las mejores características
    X_new, selector = select_best_features(X_tfidf, y)

    # Escalar las características seleccionadas
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_new)

    # Guardar el vectorizador, selector y scaler
    save_vectorizer_and_selector(vectorizer, selector)
    joblib.dump(scaler, 'scaler.pkl')

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Evaluar el modelo si ya está entrenado
    try:
        model = load_model('Random_Forest_model.pkl')
        print("Modelo cargado correctamente.")
        
        # Evaluar el modelo en el conjunto de prueba
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print("Reporte de clasificación:\n", report)

    except FileNotFoundError:
        print("No se encontró un modelo guardado. Entrenando un nuevo modelo...")

        # Entrenar un nuevo modelo
        models = {
            "Random_Forest": RandomForestClassifier(random_state=42, n_jobs=-1)
        }

        trained_models = {}
        model_times = {}
        model_accuracies = {}

        # Entrenar y evaluar cada modelo en paralelo usando ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(train_and_evaluate_model, model, model_name, X_train, X_test, y_train, y_test)
                for model_name, model in models.items()
            ]

            # Obtener resultados de los futuros a medida que se completen
            for future in concurrent.futures.as_completed(futures):
                model_name, trained_model, execution_time, accuracy = future.result()
                trained_models[model_name] = trained_model
                model_times[model_name] = execution_time
                model_accuracies[model_name] = accuracy

        # Guardar los modelos entrenados
        for model_name, model in trained_models.items():
            save_model(model, model_name)

        # Imprimir tiempos de ejecución de los modelos
        print("\nTiempos de ejecución de los modelos:")
        for model_name, exec_time in model_times.items():
            print(f'{model_name}: {exec_time:.2f} seconds')

        # Imprimir las precisiones de los modelos
        print("\nPrecisión de los modelos en el conjunto de prueba:")
        for model_name, accuracy in model_accuracies.items():
            print(f'{model_name}: {accuracy:.2f}')

if __name__ == "__main__":
    main()

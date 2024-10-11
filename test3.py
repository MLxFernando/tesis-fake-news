import pandas as pd

# Función para leer un CSV, eliminar filas donde 'text' no es string, y guardar el archivo limpio
def clean_non_string_values_in_text_column(csv_path, output_path=None):
    try:
        # Leer el archivo CSV
        df = pd.read_csv(csv_path)
        
        # Contar cuántas filas tienen valores que no son cadenas
        non_string_rows = df[~df['text'].apply(lambda x: isinstance(x, str))]
        print(f"Se encontraron {len(non_string_rows)} filas donde 'text' no es una cadena.")
        
        # Eliminar las filas donde 'text' no es string
        df_cleaned = df[df['text'].apply(lambda x: isinstance(x, str))]

        # Guardar el DataFrame limpio en un nuevo archivo CSV
        if output_path is None:
            output_path = csv_path  # Sobrescribir el archivo original si no se proporciona una ruta de salida

        df_cleaned.to_csv(output_path, index=False)
        print(f"Archivo limpio guardado en: {output_path}")

    except FileNotFoundError:
        print(f"El archivo {csv_path} no fue encontrado. Verifica la ruta del archivo.")
    except KeyError:
        print("El archivo no contiene la columna 'text'. Verifica el formato del archivo CSV.")

# Ejemplo de uso
csv_path = 'data/processed/data_cleanned_final.csv'  # Cambia esto por la ruta correcta de tu archivo CSV
output_path = 'data/processed/data_cleaned_output.csv'  # Cambia si deseas guardar en un archivo diferente
clean_non_string_values_in_text_column(csv_path, output_path)

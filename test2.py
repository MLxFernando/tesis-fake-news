import pandas as pd

# Función para leer un CSV y verificar si alguna fila en la columna 'text' no es de tipo string
def check_non_string_values_in_text_column(csv_path):
    try:
        # Leer el archivo CSV
        df = pd.read_csv(csv_path)
        
        # Iterar sobre las filas y verificar si 'text' no es de tipo string
        non_string_rows = df[~df['text'].apply(lambda x: isinstance(x, str))]
        
        # Imprimir el resultado
        if not non_string_rows.empty:
            print(f"Se encontraron {len(non_string_rows)} filas donde 'text' no es una cadena:")
            print(non_string_rows)
        else:
            print("Todos los valores en la columna 'text' son cadenas.")
    except FileNotFoundError:
        print(f"El archivo {csv_path} no fue encontrado. Verifica la ruta del archivo.")

# Ejemplo de uso
csv_path = 'data/processed/data_cleaned_output.csv'  # Cambia esto por la ruta correcta de tu archivo CSV
check_non_string_values_in_text_column(csv_path)

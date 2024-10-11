import pandas as pd

# Cargar el dataset
df = pd.read_csv('data/raw/dataset_reducido.csv')


# Filtrar por etiqueta
fake_df = df[df['label'] == 'FAKE']
real_df = df[df['label'] == 'REAL']

print(f"El dataset tiene {fake_df.shape[0]} filas de fakes y {real_df.shape[0]} filas de reales.")

print("Nuevo dataset creado con éxito.")
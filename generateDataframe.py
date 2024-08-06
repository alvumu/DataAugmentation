import os
import pandas as pd

# Definir la ruta de la carpeta que contiene los archivos .txt
carpeta = 'DataAugmentation/Data/distemist_zenodo/test_background_unannotated/text_files'

# Crear una lista para almacenar el contenido de cada archivo
contenido_archivos = []

# Recorrer todos los archivos en la carpeta
for nombre_archivo in os.listdir(carpeta):
    # Verificar si el archivo tiene extensión .txt
    if nombre_archivo.endswith('.txt'):
        # Construir la ruta completa del archivo
        ruta_archivo = os.path.join(carpeta, nombre_archivo)
        # Leer el contenido del archivo
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            contenido = archivo.read()
            contenido = contenido.replace('\n', ' ').replace('\r', '')
            # Añadir el contenido a la lista
            contenido_archivos.append({'Clinical notes': contenido})

# Crear un DataFrame a partir de la lista de contenidos
df = pd.DataFrame(contenido_archivos)

# Definir la ruta del archivo CSV de salida
ruta_csv_salida = 'DataAugmentation/Data/clinicalData_test.csv'

# Exportar el DataFrame a un archivo CSV
df.to_csv(ruta_csv_salida, index=False, encoding='utf-8')

# Mostrar un mensaje de confirmación
print(f"DataFrame exportado a {ruta_csv_salida}")


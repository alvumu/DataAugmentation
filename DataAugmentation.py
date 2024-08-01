import os
from openai import OpenAI
import pandas as pd

def generate_response(query):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {'role': 'system', 'content': "You are a data augmentation tool that helps users with a similar structure of the provided data"},
            {'role': 'user', 'content': f"{query}"}
        ]
    )
    return response.choices[0].message.content

def process_response(response,data):

    # Dividir la respuesta en bloques de texto
    texts = response.split("Text:")
    texts = [text.strip() for text in texts if text.strip()]

    for text in texts:
        text_parts = text.split("Entities:")
        if len(text_parts) == 2:
            text_content = text_parts[0].strip()
            entities_content = text_parts[1].strip()
            entities_lines = entities_content.split("\n")
            entities = ", ".join([line.strip() for line in entities_lines if line.strip()])
            # Añadir un nuevo diccionario a la lista de datos
            data.append({"Text": text_content, "Entities": entities})

    return data

# Ejemplo de uso
query1 = """
        Given the following texts:
        Text: According to the mother, her son had been hospitalised for 2 weeks because of suspected meningitis when he was four years old.
        Entities: 
        Person : mother, son  
        Procedure : hospitalised 
        Suspected Findings : suspected meningitis
        Duration : for 2 weeks

        Text: According to the uncle, his nephew was treated in the hospital for 4 days for influenza when he was eight years old.
        Entities:
        Person :uncle, nephew
        Procedure : treated in the hospital
        Findings : suspected influenza
        Duration : for 4 days

        Text: According to the father, his daughter had been hospitalised for 3 weeks because of suspected pneumonia when she was six years old.
        Entities:
        Person : father, daughter
        Procedure : hospitalized
        Suspected Findings : suspected pneumonia
        Duration : for 3 weeks

        Task: Generate similar medical texts with the same kind of entities

        """

query2 = """
Given the following text: “According to the mother, her son had been hospitalised for 2 weeks because of  suspected meningitis when he was four years old”
where: mother and son are Person; hospitalized is a Procedure; suspected meningitis is a Finding with the severity qualifier suspected; for 2 weeks and four years old is a temporal qualifier.
Generate similar medical texts with the same kind of entities
"""

query3 = """
Given the following text: “The right knee shows moderate swelling. The right knee has minor effusion. There are no deformities present in the right knee. There is tenderness over the medial joint line of the right knee. There is tenderness over the medial collateral ligament (MCL) of the right knee. There is no tenderness over the patella of the right knee. There is no tenderness over the lateral joint line of the right knee. Flexion of the right knee is 90 degrees. Extension of the right knee is minus 10 degrees.”, where right knee, medial joint line of the right knee, medial collateral ligament (MCL) of the right knee, patella of the right knee, lateral joint line of the right knee are BodyStructure, moderate swelling is a Finding with the severity qualifier moderate, minor effusion is a Finding with the severity qualifier minor, deformities is a Finding, tenderness is a Finding, flexion and extension are ObservableEntity, 90 degrees and minus 10 degrees are NumericValue. Generate similar medical texts with the same kind of entities
"""

# Crear una lista de diccionarios para almacenar cada texto y sus entidades
datalist = []

for i in range(10):
    response = generate_response(query3)
    print(response)
    # Procesar la respuesta y obtener los datos en forma de lista de diccionarios
    data = process_response(response,data = datalist)

# Convertir la lista de diccionarios en un DataFrame
df = pd.DataFrame(data)

# Exportar el DataFrame a un archivo CSV
df.to_csv("augmented_data_2.csv", index=False)

print(df)
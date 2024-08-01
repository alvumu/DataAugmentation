import os
from openai import OpenAI



def generate_response(query):

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {'role': 'system', 'content': "You are a data augmentation tool that helps users with a similar structure of the provided data"},
            {'role': 'user', 'content': f"{query}"}], 
    )
    return response.choices[0].message.content


query = """
        Given the following texts:
        Text 1: According to the mother, her son had been hospitalised for 2 weeks because of suspected meningitis when he was four years old.
        Entities: 
        Person -> mother, son  
        Procedure -> hospitalised 
        Suspected Findings -> suspected meningitis
        Duration -> for 2 weeks

        Text 2: According to the uncle, his nephew was treated in the hospital for 4 days for influenza when he was eight years old.
        Entities:
        Person -> uncle, nephew
        Procedure -> treated in the hospital
        Findings -> suspected influenza
        Duration -> for 4 days

        Text 3: According to the father, his daughter had been hospitalised for 3 weeks because of suspected pneumonia when she was six years old.
        Entities:
        Person -> father, daughter
        Procedure -> hospitalized
        Suspected Findings -> suspected pneumonia
        Duration -> for 3 weeks

        Task: Generate similar medical texts with the same kind of entities

        """


print(generate_response(query))
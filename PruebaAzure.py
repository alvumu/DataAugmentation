import os
import requests
import json
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def retrieve_similar_instances(seed_text, external_texts, top_n=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    seed_embedding = model.encode(seed_text, convert_to_tensor=True)
    external_embeddings = model.encode(external_texts, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(seed_embedding, external_embeddings)[0]
    print("Cosine scores:")
    print(cos_scores)
    top_results = torch.topk(cos_scores, k=top_n)
    print("Top results:")
    print(top_results)
    similar_texts = [external_texts[idx] for idx in top_results[1]]
    
    return similar_texts

def generate_response(query, external_texts):
    GPT4V_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_KEY,
    }
    
    # Retrieve similar instances
    similar_texts = retrieve_similar_instances(query, external_texts)
    print("Retrieved similar texts:")
    print(similar_texts)
    print("---------------------")
    retrieved_context = " ".join(similar_texts)

    payload = {
        "messages": [
            {'role': 'system', 'content': "You are a data augmentation tool that helps users with a similar structure of the provided data"},
            {'role': 'user', 'content': f"Seed Data: {query}\nRetrieved Context: {retrieved_context}\nTask: Generate similar medical texts with the same kind of entities"}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }
    GPT4V_ENDPOINT = "https://umu.openai.azure.com/openai/deployments/setupLLM/chat/completions?api-version=2024-02-15-preview"
    try:
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")
    response_json = json.loads(response.content)
    answer = response_json['choices'][0]['message']['content']
    return answer

# Sample external texts for retrieval
external_texts = pd.read_csv("DataAugmentation/Data/clinicalData.csv")["Clinical notes"].tolist()

query = """Given the following text: “According to the mother, her son had been hospitalised for 2 weeks because of  suspected meningitis when he was four years old”, where: mother and son are Person; hospitalized is a Procedure; suspected meningitis is a Finding with the severity qualifier suspected; for 2 weeks and four years old is a temporal qualifier. Generate similar medical texts with the same kind of entities"""

print(generate_response(query, external_texts))

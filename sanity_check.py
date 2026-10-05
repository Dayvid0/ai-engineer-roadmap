import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
HF_HEADERS = {"Authorization": f"Bearer {os.environ['HF_TOKEN'].strip()}"}

def get_embedding(text):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    response.raise_for_status()
    return response.json()

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    return dot / (mag_a * mag_b)

# Test 1: identical text should score ~1.0
text = "December covers ML Foundations"
e1 = get_embedding(text)
e2 = get_embedding(text)
print("Identical text similarity:", cosine_similarity(e1, e2))

# Test 2: clearly related texts should score reasonably high
a = get_embedding("December covers ML Foundations, including supervised and unsupervised learning")
b = get_embedding("What happens in December")
print("Related text similarity:", cosine_similarity(a, b))

# Test 3: clearly unrelated texts should score low
c = get_embedding("Pizza is a popular food in Italy")
print("Unrelated text similarity:", cosine_similarity(a, c))

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

with open("my_roadmap_notes.txt", "r", encoding="utf-8") as f:
    text = f.read()
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

query = "What happens in December"
query_emb = get_embedding(query)

scores = []
for i, p in enumerate(paragraphs):
    emb = get_embedding(p)
    score = cosine_similarity(query_emb, emb)
    scores.append((i, score, p[:70]))

scores.sort(key=lambda x: x[1], reverse=True)
print("Top 6 ranked by similarity:\n")
for i, score, preview in scores[:6]:
    print(f"[{i}] score={score:.4f} - {preview}...")

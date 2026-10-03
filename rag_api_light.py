import os
import requests
from fastapi import FastAPI
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
HF_HEADERS = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}

def get_embedding(text):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    response.raise_for_status()
    return response.json()

documents = []
embeddings = []

def load_and_index(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        documents.append(para)
        embeddings.append(get_embedding(para))

load_and_index("my_roadmap_notes.txt")

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    return dot / (mag_a * mag_b)

def search(query, top_n=2):
    query_embedding = get_embedding(query)
    scored = [(doc, cosine_similarity(query_embedding, emb)) for doc, emb in zip(documents, embeddings)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored[:top_n]]

@app.get("/")
def home():
    return {"message": "Lightweight RAG API is running"}

@app.get("/ask")
def ask(question: str):
    retrieved = search(question)
    context_text = "\n".join(retrieved)
    prompt = f"""Answer using only the context below. If it doesn't contain the answer, say so.

Context:
{context_text}

Question: {question}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"question": question, "retrieved_chunks": retrieved, "answer": response.choices[0].message.content}

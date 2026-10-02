from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="rag_api_notes")

def load_and_index(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        embedding = embed_model.encode(para).tolist()
        collection.add(ids=[str(i)], embeddings=[embedding], documents=[para])

# This runs once, when the server starts - not on every request
load_and_index("my_roadmap_notes.txt")

@app.get("/")
def home():
    return {"message": "RAG API is running"}

@app.get("/ask")
def ask(question: str):
    question_embedding = embed_model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=2)
    retrieved = results["documents"][0]

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

    return {
        "question": question,
        "retrieved_chunks": retrieved,
        "answer": response.choices[0].message.content
    }

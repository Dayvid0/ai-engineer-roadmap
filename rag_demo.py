import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

documents = [
    "The AI Engineer roadmap runs from September to December.",
    "Week 1 covers LLM APIs, prompting, tokens, and context windows.",
    "RAG stands for Retrieval-Augmented Generation.",
    "Groq offers a free API tier with fast inference using custom hardware.",
    "The capstone project combines RAG, agents, evaluation, and deployment.",
]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="roadmap_docs")

for i, doc in enumerate(documents):
    embedding = embed_model.encode(doc).tolist()
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[doc]
    )

question = "What months does the roadmap cover?"
question_embedding = embed_model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5
)

retrieved_chunks = results["documents"][0]
print("Retrieved chunks:", retrieved_chunks)

context_text = "\n".join(retrieved_chunks)
prompt = f"""Answer the question using only the context below.

Context:
{context_text}

Question: {question}
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    max_tokens=200,
    messages=[{"role": "user", "content": prompt}]
)

print("\nAnswer:", response.choices[0].message.content)

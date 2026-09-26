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

# --- Load real text from a file instead of a hardcoded string ---
def load_document(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, max_chars=350):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            sentences = para.split(". ")
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_chars:
                    current_chunk += sentence + ". "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            if current_chunk:
                chunks.append(current_chunk.strip())
    return chunks

document_text = load_document("ai_engineering_notes.txt")
chunks = chunk_text(document_text)

print(f"Loaded document, split into {len(chunks)} chunks.\n")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="notes_doc")

for i, chunk in enumerate(chunks):
    embedding = embed_model.encode(chunk).tolist()
    collection.add(ids=[str(i)], embeddings=[embedding], documents=[chunk])

# --- Interactive loop: ask as many questions as you want ---
print("Ask questions about your document. Type 'quit' to exit.\n")

while True:
    question = input("Question: ")
    if question.lower() == "quit":
        break

    question_embedding = embed_model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=2)
    retrieved = results["documents"][0]

    context_text = "\n".join(retrieved)
    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context_text}

Question: {question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"\nAnswer: {response.choices[0].message.content}\n")

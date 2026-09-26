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

# A longer real piece of text - imagine this came from a PDF or article
long_text = """
AI engineering is the practice of building applications on top of large language models rather than training models from scratch. It differs from traditional machine learning engineering, which focuses on building and training models directly.

The core skills of an AI engineer include prompt engineering, retrieval-augmented generation, and agent orchestration. Prompt engineering involves crafting inputs that reliably produce useful outputs from a model. RAG allows models to answer questions using external data they were never trained on. Agent orchestration involves chaining multiple tool calls and reasoning steps together to complete complex tasks.

Deployment is another critical skill. AI engineers must know how to package applications using tools like Docker, deploy them to cloud platforms, and set up continuous integration pipelines. Without proper deployment practices, even a brilliant prototype never reaches real users.

Evaluation is often overlooked by beginners but is essential in production. Teams need to know whether their AI system's outputs are accurate, safe, and useful before shipping updates. This involves both automated metrics and human review.

Safety and ethics round out the skill set. This includes protecting against prompt injection attacks, ensuring outputs don't leak private data, and being transparent about a system's limitations.
"""

# --- Chunking function: split by paragraph, with size limits ---
def chunk_text(text, max_chars=200):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # split long paragraphs further, by sentence
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

chunks = chunk_text(long_text)

print(f"Split into {len(chunks)} chunks:\n")
for i, c in enumerate(chunks):
    print(f"[{i}] ({len(c)} chars): {c}\n")

# --- Embed and store the chunks ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="ai_eng_doc")

for i, chunk in enumerate(chunks):
    embedding = embed_model.encode(chunk).tolist()
    collection.add(ids=[str(i)], embeddings=[embedding], documents=[chunk])

# --- Ask a real question ---
question = "What does an AI engineer need to know about shipping to production?"
question_embedding = embed_model.encode(question).tolist()

results = collection.query(query_embeddings=[question_embedding], n_results=2)
retrieved = results["documents"][0]

print("Retrieved chunks for the question:")
for r in retrieved:
    print(f"- {r}\n")

context_text = "\n".join(retrieved)
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

print("Answer:", response.choices[0].message.content)

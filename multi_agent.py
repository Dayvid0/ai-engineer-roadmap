import os
import json
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, START, END

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

def ask_llm(prompt, max_tokens=300):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Set up retrieval for the research agent (same as before) ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="multi_agent_notes")

def load_and_index(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        embedding = embed_model.encode(para).tolist()
        collection.add(ids=[str(i)], embeddings=[embedding], documents=[para])

load_and_index("my_roadmap_notes.txt")

def search_notes(query):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return "\n".join(results["documents"][0])

class State(TypedDict):
    question: str
    route: str
    answer: str

# --- The supervisor: decides who should handle this ---
def supervisor(state):
    decision = ask_llm(
        "Decide which specialist should handle this question. "
        "Reply with EXACTLY one word: 'research' or 'math'.\n\n"
        f"Question: {state['question']}",
        max_tokens=150
    )
    print(f"[DEBUG - raw decision: '{decision}']")
    decision_lower = decision.lower()
    route = "math" if ("math" in decision_lower or "calculat" in decision_lower) else "research"
    print(f"[Supervisor routed to: {route}]")
    return {"route": route}

# --- Specialist 1: research agent ---
def research_agent(state):
    context = search_notes(state["question"])
    answer = ask_llm(
        f"Answer using only this context. If it's not covered, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}"
    )
    return {"answer": answer}

# --- Specialist 2: math agent ---
def math_agent(state):
    expression = ask_llm(
        f"Extract ONLY the math expression from this question, nothing else. "
        f"Reply with just the expression, like '45 * 12'.\n\nQuestion: {state['question']}",
        max_tokens=150
    )
    print(f"[DEBUG - raw expression: '{expression}']")
    try:
        result = eval(expression.strip())
        answer = f"{expression.strip()} = {result}"
    except Exception:
        answer = "Could not parse a math expression from that question."
    return {"answer": answer}

# --- Routing function: reads what the supervisor decided ---
def route_decision(state):
    return state["route"]

builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("research_agent", research_agent)
builder.add_node("math_agent", math_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_decision, {
    "research": "research_agent",
    "math": "math_agent"
})
builder.add_edge("research_agent", END)
builder.add_edge("math_agent", END)

graph = builder.compile()

question = input("Ask something: ")
result = graph.invoke({"question": question, "route": "", "answer": ""})

print("\n=== ANSWER ===")
print(result["answer"])

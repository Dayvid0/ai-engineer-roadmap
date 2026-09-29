import os
import json
import operator
from typing import TypedDict, Annotated
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

# --- Tools (same as agent_v3.py) ---
def calculate(expression):
    return str(eval(expression))

def count_words(text):
    return str(len(text.split()))

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="graph_agent_notes")

def load_and_index(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        embedding = embed_model.encode(para).tolist()
        collection.add(ids=[str(i)], embeddings=[embedding], documents=[para])

load_and_index("my_roadmap_notes.txt")

def search_roadmap_notes(query):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return "\n".join(results["documents"][0])

available_functions = {
    "calculate": calculate,
    "count_words": count_words,
    "search_roadmap_notes": search_roadmap_notes,
}

tools = [
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate a basic math expression",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "count_words", "description": "Count the number of words in a piece of text",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "search_roadmap_notes", "description": "Search the user's roadmap notes",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

SYSTEM_PROMPT = {
    "role": "system",
    "content": "When you use the search_roadmap_notes tool, base your answer only on what that tool actually returns. Do not add details not present in the retrieved text."
}

# --- State: a list of messages that GROWS (operator.add appends instead of replacing) ---
class State(TypedDict):
    messages: Annotated[list, operator.add]

# --- Node 1: the "thinking" step ---
def agent_node(state):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[SYSTEM_PROMPT] + state["messages"],
        tools=tools
    )
    reply = response.choices[0].message
    return {"messages": [reply]}   # gets appended, not replaced, because of operator.add

# --- Node 2: the "acting" step ---
def tools_node(state):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        result = available_functions[func_name](**args)
        preview = result if len(result) <= 80 else result[:80] + "..."
        print(f"[{func_name}({args}) -> {preview}]")
        results.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
    return {"messages": results}

# --- The routing function: does the last message ask for a tool? ---
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "done"

builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "done": END})
builder.add_edge("tools", "agent")   # the loop

graph = builder.compile()

# --- Run it ---
question = input("Ask something: ")
result = graph.invoke({"messages": [{"role": "user", "content": question}]})

print("\n=== FINAL ANSWER ===")
print(result["messages"][-1].content)

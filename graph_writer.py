import os
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

class State(TypedDict):
    topic: str
    draft: str
    feedback: str
    final: str

def write_draft(state):
    draft = ask_llm(f"Explain '{state['topic']}' to a beginner in 3 sentences.")
    return {"draft": draft}

def review_draft(state):
    feedback = ask_llm(f"Give ONE short suggestion to improve this explanation:\n\n{state['draft']}")
    return {"feedback": feedback}

def rewrite(state):
    final = ask_llm(
        f"Rewrite this explanation in 3 sentences, applying the suggestion.\n\n"
        f"Explanation:\n{state['draft']}\n\nSuggestion:\n{state['feedback']}"
    )
    return {"final": final}

builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("review_draft", review_draft)
builder.add_node("rewrite", rewrite)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "review_draft")
builder.add_edge("review_draft", "rewrite")
builder.add_edge("rewrite", END)

graph = builder.compile()

result = graph.invoke({"topic": "embeddings", "draft": "", "feedback": "", "final": ""})

print("DRAFT:\n", result["draft"])
print("\nFEEDBACK:\n", result["feedback"])
print("\nFINAL:\n", result["final"])

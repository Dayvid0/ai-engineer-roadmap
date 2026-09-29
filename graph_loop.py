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
    revisions: int

def write_draft(state):
    draft = ask_llm(f"Explain '{state['topic']}' to a beginner in 3 sentences.")
    print("\nFIRST DRAFT:\n", draft)
    return {"draft": draft, "revisions": 0}

def review(state):
    feedback = ask_llm(
        "You are a strict reviewer. If this explanation is under 50 words, contains a real-life analogy, and is "
        "beginner-friendly, reply with exactly the single word APPROVED. "
        "Otherwise give ONE short suggestion.\n\n"
        f"{state['draft']}"
    )
    print("\nREVIEW:", feedback)
    return {"feedback": feedback}

def rewrite(state):
    draft = ask_llm(
        "Rewrite this explanation so it is under 50 words AND contains a real-life analogy. Also apply the suggestion.\n\n"
        f"Explanation:\n{state['draft']}\n\nSuggestion:\n{state['feedback']}"
    )
    print("\nREVISED DRAFT:\n", draft)
    return {"draft": draft, "revisions": state["revisions"] + 1}

# The routing function: returns a label, not a state update
def decide(state):
    if state["feedback"].strip().upper().startswith("APPROVED"):
        return "done"
    if state["revisions"] >= 2:      # safety cap: max 2 rewrites
        return "done"
    return "revise"

builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("review", review)
builder.add_node("rewrite", rewrite)

builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "review")

# The decision point: after "review", follow whichever label decide() returns
builder.add_conditional_edges("review", decide, {"revise": "rewrite", "done": END})

# The loop: after rewriting, go back to review
builder.add_edge("rewrite", "review")

graph = builder.compile()

topic = input("Topic: ")
result = graph.invoke({"topic": topic, "draft": "", "feedback": "", "revisions": 0})

print("\n=== FINAL ===")
print(result["draft"])
print(f"(Rewrites used: {result['revisions']})")

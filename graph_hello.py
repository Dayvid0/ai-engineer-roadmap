from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# The state: a dictionary with a declared shape
class State(TypedDict):
    name: str
    greeting: str
    shout: str

# Node 1: reads "name", writes "greeting"
def make_greeting(state):
    return {"greeting": f"Hello, {state['name']}!"}

# Node 2: reads "greeting", writes "shout"
def make_shout(state):
    return {"shout": state["greeting"].upper()}

# Describe the graph
builder = StateGraph(State)
builder.add_node("make_greeting", make_greeting)
builder.add_node("make_shout", make_shout)
builder.add_edge(START, "make_greeting")
builder.add_edge("make_greeting", "make_shout")
builder.add_edge("make_shout", END)

# Compile it, then run it
graph = builder.compile()
result = graph.invoke({"name": "Dayvid", "greeting": "", "shout": ""})
print(result)

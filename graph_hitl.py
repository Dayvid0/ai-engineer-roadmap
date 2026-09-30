import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
import operator
from typing import TypedDict, Annotated

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

def calculate(expression):
    return str(eval(expression))

def send_message(recipient, text):
    return f"Message actually sent to {recipient}: '{text}'"

tools = [
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate a math expression",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "send_message", "description": "Send a message to someone",
        "parameters": {"type": "object", "properties": {
            "recipient": {"type": "string"}, "text": {"type": "string"}
        }, "required": ["recipient", "text"]}}},
]

RISKY_TOOLS = {"send_message"}

class State(TypedDict):
    messages: Annotated[list, operator.add]

def agent_node(state):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=state["messages"],
        tools=tools
    )
    reply = response.choices[0].message

    # Convert the SDK object into a plain dict - safe to checkpoint, safe to resend
    message_dict = {"role": "assistant", "content": reply.content}
    if reply.tool_calls:
        message_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            }
            for tc in reply.tool_calls
        ]

    return {"messages": [message_dict]}

def tools_node(state):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.get("tool_calls", []):
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        if func_name in RISKY_TOOLS:
            decision = interrupt({
                "action": func_name,
                "args": args,
                "question": f"Approve calling {func_name} with {args}? (yes/no)"
            })
            if decision != "yes":
                result = f"Action '{func_name}' was rejected by the human. Do not attempt it again."
            else:
                result = send_message(**args)
        else:
            result = calculate(**args)

        print(f"[{func_name}({args}) -> {result}]")
        results.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})
    return {"messages": results}

def should_continue(state):
    last_message = state["messages"][-1]
    return "tools" if last_message.get("tool_calls") else "done"

builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "done": END})
builder.add_edge("tools", "agent")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "demo-1"}}

question = input("Ask something (try: 'send a message to Sam saying the meeting moved to 3pm'): ")
result = graph.invoke({"messages": [{"role": "user", "content": question}]}, config=config)

if "__interrupt__" in result:
    pause = result["__interrupt__"][0].value
    print(f"\n>>> PAUSED: {pause['question']}")
    answer = input(">>> Your answer: ")
    result = graph.invoke(Command(resume=answer), config=config)

print("\n=== FINAL ANSWER ===")
print(result["messages"][-1]["content"])

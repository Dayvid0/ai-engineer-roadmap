import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# --- Real functions the agent can call ---
def calculate(expression):
    return str(eval(expression))

def count_words(text):
    return str(len(text.split()))

# Map tool names to actual Python functions
available_functions = {
    "calculate": calculate,
    "count_words": count_words,
}

# --- Describe the tools to the model ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '23 * 47'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_words",
            "description": "Count the number of words in a piece of text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to count words in"}
                },
                "required": ["text"]
            }
        }
    }
]

def run_agent(user_question, max_iterations=5):
    messages = [{"role": "user", "content": user_question}]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools
        )
        reply = response.choices[0].message

        if not reply.tool_calls:
            # No more tools needed - this is the final answer
            print(f"Final Answer: {reply.content}")
            return reply.content

        messages.append(reply)

        # Handle every tool call the model requested this round
        for tool_call in reply.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"Thought: I should use the '{func_name}' tool")
            print(f"Action: {func_name}({args})")

            result = available_functions[func_name](**args)
            print(f"Observation: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    print("Max iterations reached without a final answer.")
    return None

# --- Try it ---
run_agent("What is 45 times 12?")

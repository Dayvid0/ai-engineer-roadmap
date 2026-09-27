import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

def calculate(expression):
    return str(eval(expression))

def count_words(text):
    return str(len(text.split()))

available_functions = {
    "calculate": calculate,
    "count_words": count_words,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. '23 * 47'"}},
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
                "properties": {"text": {"type": "string", "description": "The text to count words in"}},
                "required": ["text"]
            }
        }
    }
]

class Agent:
    def __init__(self, max_iterations=5):
        self.messages = []  # persists across calls - this IS the long-term memory
        self.max_iterations = max_iterations

    def ask(self, user_question):
        self.messages.append({"role": "user", "content": user_question})

        for iteration in range(self.max_iterations):
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=self.messages,
                tools=tools
            )
            reply = response.choices[0].message

            if not reply.tool_calls:
                self.messages.append({"role": "assistant", "content": reply.content})
                return reply.content

            self.messages.append(reply)

            for tool_call in reply.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = available_functions[func_name](**args)
                print(f"[{func_name}({args}) -> {result}]")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        return "Max iterations reached."

# --- Try it: a genuine follow-up that depends on memory ---
agent = Agent()

print("Q1:", agent.ask("How many words are in 'the quick brown fox jumps over the lazy dog'?"))
print("\nQ2:", agent.ask("Now multiply that number by 12."))
print("\nQ3:", agent.ask("What number did I just ask you to multiply by?"))

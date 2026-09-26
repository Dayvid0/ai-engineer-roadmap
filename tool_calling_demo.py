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
    return eval(expression)

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression like '23 * 47' or '(12+8)/4'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

messages = [{"role": "user", "content": "What is 847 multiplied by 293, plus 1500?"}]

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    tools=tools
)

reply = response.choices[0].message

if reply.tool_calls:
    tool_call = reply.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"Model wants to call: {tool_call.function.name}({args['expression']})")

    result = calculate(args["expression"])
    print(f"Actual result: {result}")

    messages.append(reply)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    })

    final_response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    print(f"\nFinal answer: {final_response.choices[0].message.content}")
else:
    print(reply.content)

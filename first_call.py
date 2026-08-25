import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    max_tokens=200,
    messages=[
        {"role": "user", "content": "In one sentence, explain what an LLM API actually does when you call it."}
    ]
)

print(response.choices[0].message.content)

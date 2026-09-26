import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


class ChatSession:
    def __init__(self):
        self.history = []

    def send(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=200,
            messages=self.history
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply


session = ChatSession()
print("Chat started. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    reply = session.send(user_input)
    print(f"AI: {reply}\n")
    print(f"[Conversation now has {len(session.history)} messages]")

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

def calculate(expression):
    return str(eval(expression))

def count_words(text):
    return str(len(text.split()))

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="agent_notes")

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
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
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
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_roadmap_notes",
            "description": "Search the user's personal AI Engineer roadmap notes for relevant information about the course, schedule, or concepts covered",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for"}},
                "required": ["query"]
            }
        }
    }
]

class Agent:
    def __init__(self, max_iterations=5):
        self.messages = []
        self.max_iterations = max_iterations

    def ask(self, user_question):
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": "When you use the search_roadmap_notes tool, base your answer only on what that tool actually returns. Do not add details, statistics, comparisons, or examples that were not present in the retrieved text. If the retrieved text is brief, give a brief answer."
            })
        self.messages.append({"role": "user", "content": user_question})

        for _ in range(self.max_iterations):
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
                preview = result if len(result) <= 80 else result[:80] + "..."
                print(f"[{func_name}({args}) -> {preview}]")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        return "Max iterations reached."

agent = Agent()

print("Q1:", agent.ask("What does the November part of my roadmap cover?"))
print("\nQ2:", agent.ask("What is 340 times 6?"))
print("\nQ3:", agent.ask("Why is Groq being used instead of Anthropic in this course?"))

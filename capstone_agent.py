import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# --- Tool 1: calculator ---
def calculate(expression):
    return str(eval(expression))

# --- Tool 2: word counter ---
def count_words(text):
    return str(len(text.split()))

# --- Tool 3: RAG over your roadmap notes ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="capstone_notes")

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

# --- Tool 4: real external weather API ---
def get_weather(city):
    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}).json()
    if "results" not in geo or len(geo["results"]) == 0:
        return f"Could not find a location called '{city}'."
    lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
    weather = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "current": "temperature_2m,wind_speed_10m"
    }).json()
    return f"In {city}: {weather['current']['temperature_2m']}°C, wind {weather['current']['wind_speed_10m']} km/h."

available_functions = {
    "calculate": calculate,
    "count_words": count_words,
    "search_roadmap_notes": search_roadmap_notes,
    "get_weather": get_weather,
}

tools = [
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate a basic math expression",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "count_words", "description": "Count words in a piece of text",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "search_roadmap_notes", "description": "Search the user's personal AI Engineer roadmap notes",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful assistant with four tools: calculate, count_words, "
        "search_roadmap_notes, and get_weather. When you use search_roadmap_notes, "
        "base your answer only on what it returns - never add facts, statistics, or "
        "examples that weren't in the retrieved text. Use tools only when the question "
        "actually needs them; answer directly otherwise."
    )
}

class Agent:
    def __init__(self, max_iterations=6):
        self.messages = [SYSTEM_PROMPT]
        self.max_iterations = max_iterations

    def ask(self, user_question):
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
                print(f"[{func_name}({args}) -> {result[:100]}]")
                self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        return "Max iterations reached."

# --- Interactive loop: a real multi-turn session ---
agent = Agent()
print("Capstone agent ready (calculate, count_words, search_roadmap_notes, get_weather). Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = agent.ask(question)
    print(f"\nAgent: {answer}\n")

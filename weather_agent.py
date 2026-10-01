import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# --- A real external API call, wrapped as a tool ---
def get_weather(city):
    # Step 1: turn a city name into coordinates (Open-Meteo's free geocoding endpoint)
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_response = requests.get(geo_url, params={"name": city, "count": 1})
    geo_data = geo_response.json()

    if "results" not in geo_data or len(geo_data["results"]) == 0:
        return f"Could not find a location called '{city}'."

    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]

    # Step 2: get the actual weather for those coordinates
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_response = requests.get(weather_url, params={
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,wind_speed_10m"
    })
    weather_data = weather_response.json()

    temp = weather_data["current"]["temperature_2m"]
    wind = weather_data["current"]["wind_speed_10m"]
    return f"In {city}: {temp}°C, wind speed {wind} km/h."

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name, e.g. 'Kampala'"}},
                "required": ["city"]
            }
        }
    }
]

available_functions = {"get_weather": get_weather}

def run_agent(question):
    messages = [{"role": "user", "content": question}]

    for _ in range(5):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools
        )
        reply = response.choices[0].message

        if not reply.tool_calls:
            return reply.content

        messages.append(reply)
        for tool_call in reply.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(f"[Calling get_weather({args})]")
            result = available_functions[tool_call.function.name](**args)
            print(f"[Result: {result}]")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return "Max iterations reached."

question = input("Ask about the weather somewhere: ")
print("\n" + run_agent(question))

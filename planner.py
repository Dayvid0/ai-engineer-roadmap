import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# A small helper: send one prompt, get text back
def ask_llm(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# STEP 1: ask the model to make a plan (a list of steps)
def make_plan(goal):
    prompt = f"""Break this goal into 3 short, simple steps.
Reply with ONLY a JSON list of strings, nothing else.
Example: ["step one", "step two", "step three"]

Goal: {goal}"""
    reply = ask_llm(prompt)
    try:
        return json.loads(reply)          # turn the JSON text into a Python list
    except json.JSONDecodeError:
        print("Could not read the plan. Model replied:", reply)
        return []

# STEP 2: do each step, passing along what we've done so far
def run_plan(goal, steps):
    results = []                          # will hold one result per step
    for i, step in enumerate(steps):
        print(f"\n--- Step {i + 1}: {step} ---")
        previous_work = "\n".join(results)
        prompt = f"""Overall goal: {goal}

Work completed so far:
{previous_work}

Now do ONLY this step, briefly: {step}"""
        result = ask_llm(prompt)
        print(result)
        results.append(result)            # save it for the next step
    return results

# STEP 3: run everything
goal = "Explain RAG to a beginner: give the key idea, one simple analogy, and one common mistake to avoid."

steps = make_plan(goal)
print("PLAN:", steps)

if steps:
    run_plan(goal, steps)

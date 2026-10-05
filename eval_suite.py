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

API_URL = "http://127.0.0.1:8000/ask"  # your local rag_api_light.py, must be running

# --- Test set: question + what a correct answer should contain ---
test_cases = [
    {
        "question": "What is a ReAct agent?",
        "expected_topic": "a loop of Thought, Action, and Observation"
    },
    {
        "question": "Why is Groq used instead of Anthropic?",
        "expected_topic": "Groq has a free tier with no credit card required"
    },
    {
        "question": "What happens in December?",
        "expected_topic": "ML Foundations, including supervised vs unsupervised learning and a capstone project"
    },
    {
        "question": "What is the capital of France?",
        "expected_topic": "REFUSAL - the notes don't cover this, it should say so"
    },
]

def ask_api(question):
    response = requests.get(API_URL, params={"question": question})
    return response.json()

# --- The judge: a separate LLM call that scores the response ---
def judge_response(question, expected_topic, actual_answer):
    prompt = f"""You are grading an AI system's answer. Score it from 1 to 5:
5 = fully correct and matches the expected topic
3 = partially correct or vague
1 = wrong, or hallucinated information not grounded in context

If expected_topic says "REFUSAL", a 5 means the AI correctly said it doesn't have that information.

Question: {question}
Expected topic: {expected_topic}
Actual answer: {actual_answer}

Reply with ONLY a JSON object: {{"score": <number>, "reason": "<one sentence>"}}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"score": 0, "reason": "Could not parse judge response"}

# --- Run the full eval suite ---
print(f"Running {len(test_cases)} test cases...\n")
total_score = 0

for i, case in enumerate(test_cases):
    result = ask_api(case["question"])
    actual_answer = result.get("answer", "ERROR: no answer returned")

    verdict = judge_response(case["question"], case["expected_topic"], actual_answer)
    score = verdict.get("score", 0)
    total_score += score

    print(f"[{i+1}] Q: {case['question']}")
    print(f"    Answer: {actual_answer[:100]}...")
    print(f"    Score: {score}/5 - {verdict.get('reason', '')}\n")

average = total_score / len(test_cases)
print(f"=== OVERALL: {average:.1f}/5 average across {len(test_cases)} test cases ===")

import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
HF_HEADERS = {"Authorization": f"Bearer {os.environ['HF_TOKEN'].strip()}"}

response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": "What happens in December"})
data = response.json()

print("Type:", type(data))
print("Length of outer list:", len(data))
if isinstance(data[0], list):
    print("It's nested! Length of first inner item:", len(data[0]))
    print("Type of innermost item:", type(data[0][0]))

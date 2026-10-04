from fastapi.testclient import TestClient
from rag_api_light import app

client = TestClient(app)

def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Lightweight RAG API is running"}

def test_ask_route_returns_answer():
    response = client.get("/ask", params={"question": "What is a ReAct agent?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "retrieved_chunks" in data

def test_ask_route_refuses_unrelated_question():
    response = client.get("/ask", params={"question": "What is the capital of France?"})
    data = response.json()
    assert "does not contain" in data["answer"].lower() or "context" in data["answer"].lower()

import httpx
import json

BASE_URL = "http://localhost:8000"

def test_ingest():
    payload = {
        "document_type": "markdown",
        "content": "# Test Document\n\nThis is a test for the RAG system.",
        "metadata": {"source": "manual_test"}
    }
    try:
        response = httpx.post(f"{BASE_URL}/ingest", json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ingest()

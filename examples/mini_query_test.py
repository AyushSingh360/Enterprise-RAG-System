import httpx
import json

BASE_URL = "http://localhost:8000"

def test_query():
    print("Testing Query...")
    payload = {
        "question": "How many vacation days do employees get?"
    }
    try:
        response = httpx.post(f"{BASE_URL}/query", json=payload, timeout=60)
        print(f"Status: {response.status_code}")
        try:
            print(f"Response Body: {json.dumps(response.json(), indent=2)}")
        except Exception:
            print(f"Raw Response Content: {response.text}")
    except Exception as e:
        print(f"Request Error: {e}")

if __name__ == "__main__":
    test_query()

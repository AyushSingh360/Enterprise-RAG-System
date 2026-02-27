"""
Test script to verify Enterprise RAG System behavior WITHOUT an API key.
Tests all endpoints: /, /health, /ingest, /query
"""

import httpx
import json
import sys

BASE_URL = "http://localhost:8001"

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_result(label, data):
    print(f"\n--- {label} ---")
    if isinstance(data, dict):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)

def main():
    client = httpx.Client(timeout=30.0)
    results = {}

    # ============================================================
    # TEST 1: Root Endpoint (GET /)
    # ============================================================
    print_section("TEST 1: Root Endpoint (GET /)")
    try:
        resp = client.get(f"{BASE_URL}/")
        results["root"] = {
            "status_code": resp.status_code,
            "response": resp.json()
        }
        print(f"Status Code: {resp.status_code}")
        print_result("Response", resp.json())
    except Exception as e:
        results["root"] = {"error": str(e)}
        print(f"ERROR: {e}")

    # ============================================================
    # TEST 2: Health Check (GET /health)
    # ============================================================
    print_section("TEST 2: Health Check (GET /health)")
    try:
        resp = client.get(f"{BASE_URL}/health")
        results["health"] = {
            "status_code": resp.status_code,
            "response": resp.json()
        }
        print(f"Status Code: {resp.status_code}")
        print_result("Response", resp.json())
    except Exception as e:
        results["health"] = {"error": str(e)}
        print(f"ERROR: {e}")

    # ============================================================
    # TEST 3: Ingest Markdown Content (POST /ingest)
    # ============================================================
    print_section("TEST 3: Ingest Markdown (POST /ingest)")
    try:
        ingest_payload = {
            "document_type": "markdown",
            "content": "# Company Policy\n\nEmployees receive 20 vacation days per year.\n\n## Remote Work\n\nRemote work is allowed 3 days per week.",
            "metadata": {"source": "test_policy", "department": "HR"}
        }
        resp = client.post(
            f"{BASE_URL}/ingest",
            json=ingest_payload
        )
        results["ingest_markdown"] = {
            "status_code": resp.status_code,
            "response": resp.json()
        }
        print(f"Status Code: {resp.status_code}")
        print_result("Request Payload", ingest_payload)
        print_result("Response", resp.json())
    except Exception as e:
        results["ingest_markdown"] = {"error": str(e)}
        print(f"ERROR: {e}")

    # ============================================================
    # TEST 4: Query (POST /query) - Without any ingested documents
    # ============================================================
    print_section("TEST 4: Query (POST /query)")
    try:
        query_payload = {
            "question": "How many vacation days do employees get?",
            "top_k": 5,
            "similarity_threshold": 0.7
        }
        resp = client.post(
            f"{BASE_URL}/query",
            json=query_payload
        )
        results["query"] = {
            "status_code": resp.status_code,
            "response": resp.json()
        }
        print(f"Status Code: {resp.status_code}")
        print_result("Request Payload", query_payload)
        print_result("Response", resp.json())
    except Exception as e:
        results["query"] = {"error": str(e)}
        print(f"ERROR: {e}")

    # ============================================================
    # TEST 5: Query with empty question (validation test)
    # ============================================================
    print_section("TEST 5: Query Validation (POST /query with empty question)")
    try:
        query_payload = {
            "question": ""
        }
        resp = client.post(
            f"{BASE_URL}/query",
            json=query_payload
        )
        results["query_validation"] = {
            "status_code": resp.status_code,
            "response": resp.json()
        }
        print(f"Status Code: {resp.status_code}")
        print_result("Response", resp.json())
    except Exception as e:
        results["query_validation"] = {"error": str(e)}
        print(f"ERROR: {e}")

    # ============================================================
    # TEST 6: Ingest with missing file_path (PDF validation)
    # ============================================================
    print_section("TEST 6: Ingest Validation (POST /ingest - PDF without file_path)")
    try:
        ingest_payload = {
            "document_type": "pdf"
        }
        resp = client.post(
            f"{BASE_URL}/ingest",
            json=ingest_payload
        )
        results["ingest_validation"] = {
            "status_code": resp.status_code,
            "response": resp.json()
        }
        print(f"Status Code: {resp.status_code}")
        print_result("Response", resp.json())
    except Exception as e:
        results["ingest_validation"] = {"error": str(e)}
        print(f"ERROR: {e}")

    # ============================================================
    # TEST 7: OpenAPI Docs availability
    # ============================================================
    print_section("TEST 7: OpenAPI Docs (GET /openapi.json)")
    try:
        resp = client.get(f"{BASE_URL}/openapi.json")
        results["openapi"] = {
            "status_code": resp.status_code,
            "endpoints_count": len(resp.json().get("paths", {})),
            "title": resp.json().get("info", {}).get("title", ""),
            "version": resp.json().get("info", {}).get("version", "")
        }
        print(f"Status Code: {resp.status_code}")
        print(f"Title: {results['openapi']['title']}")
        print(f"Version: {results['openapi']['version']}")
        print(f"Endpoints: {results['openapi']['endpoints_count']}")
    except Exception as e:
        results["openapi"] = {"error": str(e)}
        print(f"ERROR: {e}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print_section("SUMMARY")
    for test_name, result in results.items():
        status = result.get("status_code", "ERROR")
        error = result.get("error", "")
        if error:
            print(f"  {test_name}: FAILED - {error}")
        else:
            resp_data = result.get("response", {})
            success = resp_data.get("success", resp_data.get("status", "N/A"))
            print(f"  {test_name}: HTTP {status} - success/status={success}")

    client.close()
    print("\nDone!")

if __name__ == "__main__":
    main()

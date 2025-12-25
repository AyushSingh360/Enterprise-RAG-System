"""
Example script to test the RAG system.

Usage:
1. First, ensure your .env file has OPENAI_API_KEY set
2. Start the server: uvicorn app.main:app --reload
3. Run this script: python examples/test_rag.py
"""

import httpx
import time

BASE_URL = "http://localhost:8000"


def check_health():
    """Check if server is running."""
    print("=" * 60)
    print("1. HEALTH CHECK")
    print("=" * 60)
    
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except httpx.ConnectError:
        print("ERROR: Cannot connect to server. Is it running?")
        print("Start with: uvicorn app.main:app --reload")
        return False


def ingest_document():
    """Ingest the sample policy document."""
    print("\n" + "=" * 60)
    print("2. INGEST DOCUMENT")
    print("=" * 60)
    
    # Ingest markdown content directly
    payload = {
        "document_type": "markdown",
        "content": """# Company Policy Document

## Remote Work Policy

Employees are permitted to work remotely up to 3 days per week with prior manager approval. Remote work arrangements must be documented and reviewed quarterly.

### Eligibility

All full-time employees who have completed their probationary period (90 days) are eligible for remote work. Contract workers must obtain written approval from HR.

### Equipment

The company provides a laptop and monitor for remote work. Employees are responsible for maintaining a stable internet connection with minimum 50 Mbps download speed.

## Vacation Policy

Full-time employees receive 20 paid vacation days per year. Vacation days accrue monthly at a rate of 1.67 days per month.

### Carryover

Unused vacation days can be carried over to the next year, up to a maximum of 5 days. Days beyond the carryover limit will be forfeited on December 31st.

### Requesting Time Off

Vacation requests must be submitted at least 2 weeks in advance through the HR portal. Requests for 5 or more consecutive days require department head approval.

## Sick Leave

Employees receive 10 paid sick days per year. Sick days do not accrue and reset on January 1st each year.

### Documentation

For absences exceeding 3 consecutive days, a doctor's note is required upon return to work.

## Professional Development

The company allocates $2,000 per employee annually for professional development. This includes conferences, courses, certifications, and books.

### Approval Process

Professional development requests must be submitted to your manager with a brief explanation of how the training benefits your role. Approval typically takes 5-7 business days.
""",
        "metadata": {
            "source": "company_policy",
            "department": "HR"
        }
    }
    
    response = httpx.post(
        f"{BASE_URL}/ingest",
        json=payload,
        timeout=60
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Success: {result.get('success')}")
    print(f"Message: {result.get('message')}")
    print(f"Total chunks: {result.get('total_chunks')}")
    print(f"Processing time: {result.get('processing_time_ms'):.2f} ms")
    
    return result.get("success", False)


def query_with_answer(question: str):
    """Send a query that should have an answer."""
    print(f"\nQuery: {question}")
    print("-" * 40)
    
    response = httpx.post(
        f"{BASE_URL}/query",
        json={"question": question},
        timeout=60
    )
    
    result = response.json()
    
    print(f"Answer: {result.get('answer')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Sources: {len(result.get('sources', []))}")
    
    for i, source in enumerate(result.get("sources", []), 1):
        print(f"  [{i}] {source.get('source')} - Section: {source.get('section')} (score: {source.get('relevance_score')})")
    
    print(f"Query time: {result.get('query_time_ms'):.2f} ms")
    
    return result


def query_without_answer(question: str):
    """Send a query that should NOT have an answer (tests hallucination prevention)."""
    print(f"\nQuery: {question}")
    print("-" * 40)
    
    response = httpx.post(
        f"{BASE_URL}/query",
        json={"question": question},
        timeout=60
    )
    
    result = response.json()
    
    print(f"Answer: {result.get('answer')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Sources: {result.get('sources')}")
    
    # Check if hallucination was prevented
    if result.get("confidence") == 0.0 and "not found" in result.get("answer", "").lower():
        print("✓ HALLUCINATION PREVENTED - System correctly refused to answer")
    else:
        print("⚠ WARNING: System may have hallucinated")
    
    return result


def run_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ENTERPRISE RAG SYSTEM - TEST SUITE")
    print("=" * 60)
    
    # 1. Health check
    if not check_health():
        return
    
    # 2. Ingest document
    if not ingest_document():
        print("Ingestion failed!")
        return
    
    # Wait a moment for indexing
    time.sleep(1)
    
    # 3. Queries with expected answers
    print("\n" + "=" * 60)
    print("3. QUERIES WITH EXPECTED ANSWERS")
    print("=" * 60)
    
    questions = [
        "How many vacation days do employees get?",
        "What is the remote work policy?",
        "How much is the professional development budget?",
        "What happens to unused vacation days?",
        "When is a doctor's note required?",
    ]
    
    for q in questions:
        query_with_answer(q)
    
    # 4. Queries that should NOT be answered (hallucination test)
    print("\n" + "=" * 60)
    print("4. HALLUCINATION PREVENTION TEST")
    print("=" * 60)
    print("These questions are NOT in the documents.")
    print("The system should refuse to answer.")
    
    out_of_scope_questions = [
        "What is the CEO's name?",
        "What is the company's stock price?",
        "When was the company founded?",
    ]
    
    for q in out_of_scope_questions:
        query_without_answer(q)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

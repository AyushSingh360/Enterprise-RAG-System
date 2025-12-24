# Enterprise RAG System

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions **strictly from private data** and **never hallucinates**.

## 🚀 Features

- **Zero Hallucination**: Answers are generated ONLY from ingested documents
- **Mandatory Source Citation**: Every answer includes document sources with page numbers
- **Multi-Format Support**: PDF, DOCX, Markdown, and SQL database data
- **Semantic Chunking**: Intelligent document splitting with overlap handling
- **Deterministic Embeddings**: Rebuild-safe indexing with caching
- **Production-Ready**: FAISS vector store with persistence, structured logging, health checks

## 📋 Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| RAG Engine | LlamaIndex |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT-4 Turbo |
| Vector Store | FAISS (local, persistent) |
| Validation | Pydantic v2 |

## 📁 Project Structure

```
RAG/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── main.py                # FastAPI application entry
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py    # Dependency injection
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py      # GET /health
│   │       ├── ingest.py      # POST /ingest
│   │       └── query.py       # POST /query
│   ├── core/
│   │   ├── __init__.py
│   │   ├── embeddings.py      # Embedding service
│   │   ├── llm.py             # LLM with no-hallucination prompt
│   │   ├── retriever.py       # RAG pipeline orchestration
│   │   └── vector_store.py    # FAISS vector store
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py         # Semantic chunking
│   │   ├── loader.py          # Document loaders
│   │   └── pipeline.py        # Ingestion orchestration
│   └── schemas/
│       ├── __init__.py
│       ├── common.py          # Common response schemas
│       ├── documents.py       # Document/ingestion schemas
│       └── query.py           # Query/response schemas
├── data/
│   ├── documents/             # Document metadata storage
│   └── faiss_index/           # FAISS index persistence
├── .env.example               # Environment template
├── requirements.txt           # Python dependencies
└── README.md
```

## 🔧 Installation

### 1. Clone and Setup

```bash
cd RAG
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📖 API Reference

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "vector_store": "healthy",
    "llm": "healthy",
    "embeddings": "healthy"
  }
}
```

### Ingest Document

```http
POST /ingest
Content-Type: application/json
```

**Request Body (PDF/DOCX):**
```json
{
  "document_type": "pdf",
  "file_path": "/path/to/document.pdf",
  "metadata": {
    "department": "Engineering",
    "project": "RAG System"
  }
}
```

**Request Body (Markdown):**
```json
{
  "document_type": "markdown",
  "content": "# Document Title\n\nDocument content here...",
  "metadata": {
    "source": "manual_entry"
  }
}
```

**Request Body (SQL):**
```json
{
  "document_type": "sql",
  "sql_query": "SELECT * FROM knowledge_base WHERE active = 1",
  "connection_string": "sqlite:///./data/knowledge.db",
  "metadata": {
    "table": "knowledge_base"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully ingested document with 45 chunks",
  "documents": [
    {
      "document_id": "doc_abc123def456",
      "source": "technical_manual.pdf",
      "document_type": "pdf",
      "chunk_count": 45,
      "ingested_at": "2024-01-15T10:30:00Z",
      "metadata": {}
    }
  ],
  "total_chunks": 45,
  "processing_time_ms": 1523.45
}
```

### File Upload

```http
POST /ingest/file
Content-Type: multipart/form-data
```

Upload files directly via multipart form.

### Query Documents

```http
POST /query
Content-Type: application/json
```

**Request Body:**
```json
{
  "question": "What is the company's remote work policy?",
  "top_k": 5,
  "similarity_threshold": 0.7,
  "include_context": false,
  "metadata_filter": {
    "department": "HR"
  }
}
```

**Response (Answer Found):**
```json
{
  "success": true,
  "answer": "According to the HR policy document, employees can work remotely up to 3 days per week with manager approval. [Source 1]",
  "sources": [
    {
      "document_id": "doc_abc123",
      "source": "hr_policies.pdf",
      "page_number": 15,
      "section": "Remote Work Policy",
      "relevance_score": 0.92,
      "chunk_text": "Employees may request remote work..."
    }
  ],
  "confidence": 0.89,
  "query_time_ms": 1245.67,
  "retrieval_time_ms": 89.23,
  "generation_time_ms": 1156.44
}
```

**Response (No Context Found):**
```json
{
  "success": true,
  "answer": "Answer not found in documents.",
  "sources": [],
  "confidence": 0.0,
  "query_time_ms": 234.56
}
```

## 🔒 No-Hallucination Guarantee

The system enforces strict context-only answers through:

1. **System Prompt**: Explicit instructions to NEVER use prior knowledge
2. **Empty Context Handling**: Returns "Answer not found" when no relevant docs
3. **Similarity Threshold**: Filters out low-relevance matches (default: 0.7)
4. **Confidence Scoring**: Tracks answer reliability (0.0 = no answer)
5. **Mandatory Citations**: Every answer must reference source documents

## ⚙️ Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |
| `LLM_MODEL` | gpt-4-turbo-preview | LLM for generation |
| `LLM_TEMPERATURE` | 0.0 | Zero for deterministic output |
| `SIMILARITY_THRESHOLD` | 0.7 | Minimum relevance score |
| `TOP_K` | 5 | Max documents to retrieve |
| `CHUNK_SIZE` | 512 | Tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |

## 🧪 Example Usage

### Python Client

```python
import httpx

BASE_URL = "http://localhost:8000"

# Ingest a PDF
response = httpx.post(f"{BASE_URL}/ingest", json={
    "document_type": "pdf",
    "file_path": "./documents/handbook.pdf",
    "metadata": {"category": "policies"}
})
print(response.json())

# Query
response = httpx.post(f"{BASE_URL}/query", json={
    "question": "What are the vacation policies?",
    "include_context": True
})
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {[s['source'] for s in result['sources']]}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Ingest markdown
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "markdown",
    "content": "# Company Policy\n\nAll employees must...",
    "metadata": {"category": "policies"}
  }'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What must all employees do?"}'
```

## 📊 Performance

- **Embedding Generation**: ~100 chunks/second
- **FAISS Search**: <50ms for 100K vectors
- **End-to-End Query**: ~1-2 seconds (depends on LLM)

## 🛡️ Production Considerations

1. **API Key Security**: Use secrets management in production
2. **Rate Limiting**: Add middleware for API rate limits
3. **Monitoring**: Integrate with observability tools (metrics, traces)
4. **Scaling**: Consider IVF FAISS index for >1M vectors
5. **Backup**: Regularly backup `data/` directory

## 📄 License

MIT License - See LICENSE file for details.

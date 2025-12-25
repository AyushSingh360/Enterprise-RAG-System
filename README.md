# Enterprise RAG System

Backend-only Retrieval-Augmented Generation system. Answers questions strictly from ingested documents. Does not hallucinate.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FastAPI                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ POST /ingest │  │ POST /query  │  │ GET /health  │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘                  │
└─────────┼─────────────────┼─────────────────────────────────────────────┘
          │                 │
          ▼                 ▼
┌─────────────────┐  ┌─────────────────────────────────────────────────┐
│ IngestionService│  │              RetrieverService                    │
│                 │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│ ┌─────────────┐ │  │  │Embedding│─▶│ FAISS   │─▶│  LLM    │          │
│ │DocumentLoader│ │  │  │ Service │  │ Search  │  │ Service │          │
│ ├─────────────┤ │  │  └─────────┘  └─────────┘  └─────────┘          │
│ │  Chunker    │ │  └─────────────────────────────────────────────────┘
│ └─────────────┘ │
└─────────────────┘
          │                 │
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         VectorStoreService                               │
│                      (FAISS + JSON metadata)                             │
│                       Persisted to ./data/                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## RAG Flow

### Ingestion (`POST /ingest`)

```
Document (PDF/DOCX/MD)
        │
        ▼
┌───────────────┐
│ DocumentLoader│  Extract text with metadata (page, section)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│SemanticChunker│  Split into chunks (512 tokens, 50 overlap)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│EmbeddingService│  Generate embeddings (text-embedding-3-small)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ VectorStore   │  Store in FAISS + persist to disk
└───────────────┘
```

### Query (`POST /query`)

```
Question
    │
    ▼
┌──────────────────┐
│ Embed Question   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ FAISS Similarity │  top_k=5, threshold=0.7
│     Search       │
└────────┬─────────┘
         │
         ├── NO RESULTS ──▶ Return: "Answer not found in documents."
         │                  confidence=0.0, sources=[]
         │
         ▼ (has results)
┌──────────────────┐
│   LLM Generate   │  System prompt: use ONLY context
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Return answer +  │  Mandatory citations in sources[]
│    citations     │
└──────────────────┘
```

## Hallucination Prevention

The system prevents hallucination through multiple mechanisms:

### 1. Retrieval Gate
If similarity search returns no results above threshold (default 0.7), generation is skipped entirely. Response:
```json
{
  "answer": "Answer not found in documents.",
  "sources": [],
  "confidence": 0.0
}
```

### 2. System Prompt Constraint
The LLM receives this system prompt:

```
ABSOLUTE RULES - VIOLATION IS FORBIDDEN:
1. You may ONLY use information explicitly stated in the provided CONTEXT.
2. You must NEVER use your training data, prior knowledge, or make assumptions.
3. If the CONTEXT does not contain enough information to answer, you MUST respond:
   "I don't know based on the provided documents."
4. If the CONTEXT is empty or completely irrelevant, you MUST respond:
   "Answer not found in documents."
```

### 3. Context Isolation
The LLM only receives retrieved chunks, not the full document corpus. It cannot access information that wasn't retrieved.

### 4. Mandatory Citations
Every response must include source citations. Empty sources = no answer.

## Setup

### Requirements
- Python 3.10+
- OpenAI API key

### Installation

```bash
cd RAG
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
```

### Run

```bash
uvicorn app.main:app --reload --port 8000
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `APP_ENV` | No | development | Environment (development/production) |
| `LOG_LEVEL` | No | INFO | Logging level |
| `EMBEDDING_MODEL` | No | text-embedding-3-small | OpenAI embedding model |
| `EMBEDDING_DIMENSION` | No | 1536 | Embedding vector size |
| `LLM_MODEL` | No | gpt-4-turbo-preview | OpenAI chat model |
| `LLM_TEMPERATURE` | No | 0.0 | LLM temperature (0=deterministic) |
| `SIMILARITY_THRESHOLD` | No | 0.7 | Min similarity for retrieval |
| `TOP_K` | No | 5 | Max documents to retrieve |
| `CHUNK_SIZE` | No | 512 | Chunk size in tokens |
| `CHUNK_OVERLAP` | No | 50 | Overlap between chunks |
| `FAISS_INDEX_PATH` | No | ./data/faiss_index | FAISS persistence path |
| `DOCUMENT_STORE_PATH` | No | ./data/documents | Metadata storage path |

## API

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "embeddings": "healthy",
    "vector_store": "healthy",
    "llm": "healthy"
  }
}
```

### Ingest Document

**PDF:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "pdf",
    "file_path": "C:/path/to/document.pdf",
    "metadata": {"department": "HR"}
  }'
```

**Markdown content:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "markdown",
    "content": "# Policy\n\nEmployees get 20 vacation days per year.",
    "metadata": {"source": "hr_policy"}
  }'
```

Response:
```json
{
  "success": true,
  "message": "Successfully ingested with 5 chunks",
  "documents": [
    {
      "document_id": "doc_a1b2c3d4e5f6",
      "source": "document.pdf",
      "document_type": "pdf",
      "chunk_count": 5,
      "ingested_at": "2024-01-15T10:30:00Z",
      "metadata": {"department": "HR"}
    }
  ],
  "total_chunks": 5,
  "processing_time_ms": 1234.56
}
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many vacation days do employees get?"
  }'
```

**Response (answer found):**
```json
{
  "success": true,
  "answer": "Employees get 20 vacation days per year. [Source 1]",
  "sources": [
    {
      "document_id": "doc_a1b2c3d4e5f6",
      "source": "hr_policy",
      "page_number": null,
      "section": "Policy",
      "relevance_score": 0.89,
      "chunk_text": "Employees get 20 vacation days per year."
    }
  ],
  "confidence": 0.85,
  "query_time_ms": 1456.78,
  "retrieval_time_ms": 45.23,
  "generation_time_ms": 1411.55
}
```

**Response (no answer):**
```json
{
  "success": true,
  "answer": "Answer not found in documents.",
  "sources": [],
  "confidence": 0.0,
  "query_time_ms": 45.23,
  "retrieval_time_ms": 45.23,
  "generation_time_ms": 0.0
}
```

### Query with Options

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the remote work policy?",
    "top_k": 3,
    "similarity_threshold": 0.8,
    "include_context": true,
    "metadata_filter": {"department": "HR"}
  }'
```

## Project Structure

```
RAG/
├── app/
│   ├── __init__.py
│   ├── config.py              # Pydantic Settings
│   ├── main.py                # FastAPI app
│   ├── api/
│   │   ├── dependencies.py    # DI singletons
│   │   └── routes/
│   │       ├── health.py      # GET /health
│   │       ├── ingest.py      # POST /ingest
│   │       └── query.py       # POST /query
│   ├── core/
│   │   ├── embeddings.py      # OpenAI embeddings + cache
│   │   ├── llm.py             # OpenAI chat + no-hallucination prompt
│   │   ├── retriever.py       # RAG orchestration
│   │   └── vector_store.py    # FAISS + persistence
│   ├── ingestion/
│   │   ├── loader.py          # PDF/DOCX/MD loaders
│   │   ├── chunker.py         # Semantic chunking
│   │   └── pipeline.py        # Ingestion orchestration
│   └── schemas/
│       ├── common.py          # HealthResponse, ErrorResponse
│       ├── documents.py       # IngestRequest/Response
│       └── query.py           # QueryRequest/Response, SourceCitation
├── data/
│   ├── documents/             # Metadata JSON
│   └── faiss_index/           # FAISS index files
├── .env.example
├── requirements.txt
└── README.md
```

## Data Persistence

- FAISS index: `./data/faiss_index/faiss.index`
- Chunk metadata: `./data/faiss_index/chunks.json`
- Document metadata: `./data/faiss_index/documents.json`
- Embedding cache: `./data/documents/embedding_cache/`

All data survives restarts. Delete `./data/` to reset.

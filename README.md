# AI & Data Science Learning Book

An interactive, full-stack educational platform that combines a nine-module AI/Data Science book with a strictly book-grounded RAG chatbot.

---

## What's Inside

| Layer | Technology |
|---|---|
| Book / Frontend | Docusaurus 3, React, TypeScript |
| Backend API | Python 3.11, FastAPI, Pydantic v2 |
| Embeddings | Google Gemini (`models/gemini-embedding-001`) |
| Generation | Google Gemini (`gemini-3.5-flash-lite`) |
| Vector Store | Qdrant |
| Containerisation | Docker, Docker Compose |

### Nine Curriculum Modules

1. Python Foundations
2. Python for Data Science — NumPy, Pandas & Visualization
3. Statistics and Mathematics for Machine Learning
4. Machine Learning
5. Deep Learning
6. MLOps
7. Big Data and Cloud Computing
8. Generative AI and Large Language Models
9. Agentic AI

---

## Project Structure

```
ai-ds-learning-book/
├── docs/                        # Source Markdown book content
│   ├── intro.md
│   ├── module-01-python-foundations.md
│   └── module-02 … module-09 …
├── frontend/                    # Docusaurus site
│   ├── docs/                    # Copy of docs/ served by Docusaurus
│   ├── chatbot/                 # Docusaurus route alias → src/pages/chat
│   │   ├── index.tsx
│   │   └── Chatbot.css
│   ├── src/
│   │   ├── pages/               # Custom React pages (home, chat, progress, projects)
│   │   ├── components/          # Reusable components (FloatingChatbot, Quiz, Progress)
│   │   ├── theme/               # Docusaurus theme overrides (Root.tsx)
│   │   └── css/custom.css
│   ├── static/img/              # Logo SVG, favicon
│   ├── docusaurus.config.ts
│   ├── sidebars.ts
│   ├── tsconfig.json
│   ├── package.json
│   └── Dockerfile
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + lifespan
│   │   ├── config.py            # Pydantic settings
│   │   ├── routes/
│   │   │   ├── chat.py          # POST /api/chat
│   │   │   ├── ingest.py        # POST /api/ingest
│   │   │   └── health.py        # GET  /health and /api/health
│   │   └── rag/
│   │       ├── pipeline.py      # Top-level RAGPipeline
│   │       ├── chunker.py       # Token-based chunker
│   │       ├── embeddings.py    # Gemini embeddings wrapper
│   │       ├── retrieve.py      # Vector retrieval + evidence check
│   │       ├── generator.py     # Grounded answer generation
│   │       ├── ingest.py        # Book content indexer
│   │       ├── store.py         # Qdrant integration
│   │       └── constants.py     # Shared RAG constants
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_api.py          # 20 endpoint tests
│   │   ├── test_chunker.py      # 23 chunker unit tests
│   │   ├── test_rag_pipeline.py # 22 RAG pipeline unit tests
│   │   └── test_evaluation.py   # 8 RAG precision/recall & refusal tests
│   ├── ingest_cli.py            # Repeatable CLI content ingestion script
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── .env.example
│   ├── pytest.ini
│   └── Dockerfile
├── .github/workflows/ci.yml     # Automated CI for tests and frontend build
└── docker-compose.yml
```

---

## Prerequisites

| Tool | Minimum Version |
|---|---|
| Python | 3.11 |
| Node.js | 18 |
| npm | 9 |
| Docker | 24 |
| Docker Compose | v2 |

You also need a **Google Gemini API key** — get one free at <https://aistudio.google.com/app/apikey>.

---

## Local Development Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/your-org/ai-ds-learning-book.git
cd ai-ds-learning-book
```

### 2. Configure environment variables

```bash
cp backend/.env.example backend/.env
```

Open `backend/.env` and fill in at minimum:

```
GEMINI_API_KEY=your_key_here
```

For the existing Cloud cluster, set `QDRANT_URL` to its HTTPS endpoint and
`QDRANT_API_KEY` in `backend/.env`. The backend does not inherit variables from
the Codex MCP process.

### 3. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.9.7
```

### 4. Set up and run the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API is now available at <http://localhost:8000>.  
Interactive docs: <http://localhost:8000/docs>

### 5. Ingest book content

You can index all Markdown files into Qdrant using the REST API or the standalone CLI script:

**Option A: Standalone CLI (Recommended)**
```bash
cd backend
python ingest_cli.py --dry-run          # inspect files & chunk counts locally
python ingest_cli.py --force-rebuild    # live index into Qdrant
```

**Option B: REST API**
```bash
curl -X POST http://localhost:8000/api/ingest/sync \
     -H "Content-Type: application/json" \
     -d '{"force_rebuild": true}'
```

Expected response:

```json
{
  "status": "completed",
  "message": "Ingestion completed successfully.",
  "files_discovered": 11,
  "chunks_indexed": 39,
  "errors": 0
}
```

These figures describe the current local dry-run; live totals can change with content and chunk settings.

### 6. Run the frontend

```bash
cd frontend
npm install
npm start
```

The site is now available at <http://localhost:3000>.

---

## Running Tests

```bash
cd backend
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                            # run the backend test suite
pytest tests/test_api.py -v       # endpoint tests only
pytest tests/test_chunker.py      # chunker unit tests
pytest tests/test_rag_pipeline.py # RAG pipeline tests
pytest tests/test_evaluation.py   # RAG grounded eval & refusal benchmark
```

Tests run without real API credentials — all external services are mocked.

## RAG and Qdrant Operations

The FastAPI service connects to Qdrant at runtime. Codex-to-Qdrant MCP is a separate management connection and does not provide the website with runtime access.

The default collection is `ai_ds_book`. The backend uses `models/gemini-embedding-001` with an explicitly requested 768 dimensional output and cosine distance. The model supports 128–3072 dimensions; 768 is one of Google's recommended sizes. Keep `EMBEDDING_DIMENSION` unchanged for an existing collection, or create a new collection and reindex when changing embedding models or dimensions.

Docker Compose passes the values from `backend/.env` into the backend. The bundled Qdrant service is optional local infrastructure; it is not used when `QDRANT_URL` points to Cloud. To use Cloud, set `QDRANT_URL=https://your-cluster-url` and `QDRANT_API_KEY` in `backend/.env`, then start Compose.

Check the local configuration and collection connection:

```bash
cd backend
python ingest_cli.py --check
```

Inspect discovered lessons and deterministic chunks without credentials:

```bash
python ingest_cli.py --dry-run
```

Run an incremental synchronization. Changed lessons are replaced by document id, deleted lessons have their stale chunks removed, and unrelated collections are untouched:

```bash
python ingest_cli.py
```

Run a full rebuild of points tagged for this book dataset in `ai_ds_book` (the collection itself is preserved):

```bash
python ingest_cli.py --force-rebuild
```

The API equivalents are `GET /api/health`, `POST /api/ingest/sync`, and `GET /api/ingest/status`. A live index run requires `GEMINI_API_KEY` and a reachable Qdrant instance. Run `python ingest_cli.py --check` before ingestion and `python ingest_cli.py --dry-run` to preview file and chunk totals. A successful API health response alone does not prove Qdrant is reachable. The current restricted execution environment cannot resolve the configured Cloud endpoint, so the collection must be checked from a network-enabled runtime before creation or ingestion.

## Codex Qdrant MCP

The configured Qdrant MCP tool responded to a read query in this session. Its available tools are Qdrant memory lookup and storage; they do not expose collection listing, schema inspection, point counts, or vector search for the application collection. The MCP check therefore does not verify the app's collection or replace its Python Qdrant client. In this environment, `python ingest_cli.py --check` could not reach the configured Qdrant endpoint (`Operation not permitted`), so live collection details and ingestion remain unverified.

For local stdio setup, the official server documents:

```bash
QDRANT_URL=http://localhost:6333 \
COLLECTION_NAME=ai_ds_book \
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
uvx mcp-server-qdrant
```

Use the FastAPI environment values for the website. The MCP server's embedding settings are independent of the Gemini vectors used by the website; do not use MCP to write vectors into the application collection unless its embedding model and dimensions match the backend configuration.

## Refusal Behavior

Greetings receive a short response. Questions without sufficient retrieved evidence receive: “I can only answer questions using the content available in this AI/ML book. I could not find enough relevant information in the book to answer your question.” Qdrant or embedding failures receive a separate temporary service message. The generator is instructed to use accepted book evidence only and to ignore instructions found inside retrieved text.

---

## Docker Compose (All-in-one)

Build and start all three services with one command:

```bash
cp backend/.env.example backend/.env
# Edit backend/.env — add GEMINI_API_KEY

docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | <http://localhost:3000> |
| Backend API | <http://localhost:8000> |
| API docs | <http://localhost:8000/docs> |
| Qdrant dashboard | <http://localhost:6333/dashboard> |

After containers start, run the ingestion:

```bash
curl -X POST http://localhost:8000/api/ingest/sync \
     -H "Content-Type: application/json" \
     -d '{"force_rebuild": true}'
```

To stop:

```bash
docker compose down           # stop containers, keep data
docker compose down -v        # stop and delete Qdrant data volume
```

---

## API Reference

### `GET /health` and `GET /api/health`

Returns service health status.

```json
{"status": "healthy", "service": "AI & Data Science Learning Book API"}
```

### `POST /api/chat`

Ask a question grounded in the book.

**Request:**
```json
{
  "question": "What is gradient descent?",
  "session_id": "optional-session-id",
  "include_citations": true
}
```

**Response:**
```json
{
  "answer": "Gradient descent is an optimisation algorithm...",
  "citations": [
    {
      "title": "Statistics and Mathematics for ML",
      "module": "Module 3",
      "section": "3.5 Calculus for ML",
      "file_path": "module-03-statistics-and-mathematics.md",
      "relevance_score": 0.94
    }
  ],
  "evidence_threshold_met": true
}
```

When the question is outside the book's coverage:
```json
{
  "answer": "I couldn't find sufficient information about this topic in the AI & Data Science Book. Please try asking about a topic covered in the book.",
  "citations": [],
  "evidence_threshold_met": false
}
```

### `POST /api/ingest`

Start background ingestion (non-blocking).

```json
{"force_rebuild": false}
```

### `POST /api/ingest/sync`

Run ingestion synchronously (blocking — use for scripts/CI).

### `GET /api/ingest/status`

Check the status of the running or last-completed ingestion job.

---

## Configuration Reference

All settings are loaded from `backend/.env`. See `backend/.env.example` for descriptions.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Your Gemini API key |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model |
| `GEMINI_GENERATION_MODEL` | `gemini-3.5-flash-lite` | Generation model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | — | Required for Qdrant Cloud |
| `CHUNK_SIZE` | `800` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap tokens between chunks |
| `EVIDENCE_THRESHOLD` | `0.7` | Minimum retrieval score (0–1) |
| `BOOK_DIR` | `../frontend/docs` | Path to Markdown content |

---

## Deployment

### Frontend (Netlify / Vercel)

```bash
cd frontend
npm run build
# Upload the build/ directory to Netlify or Vercel
# Set environment variable: REACT_APP_API_URL=https://your-backend-url.com
```

### Backend (any cloud VM or container service)

```bash
# On the server:
docker build -t ai-ds-backend ./backend
docker run -d \
  -p 8000:8000 \
  --env-file backend/.env \
  -e QDRANT_URL=https://your-qdrant-cloud-url \
  ai-ds-backend
```

### Qdrant Cloud

1. Create a free cluster at <https://cloud.qdrant.io>
2. Copy the cluster URL and API key
3. Set in `backend/.env`:
   ```
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your_qdrant_api_key
   ```

---

## Reindexing Content

Whenever you update or add Markdown files in `docs/`, sync to `frontend/docs/` and reindex:

```bash
cp -r docs/. frontend/docs/

# With local backend running:
curl -X POST http://localhost:8000/api/ingest/sync \
     -H "Content-Type: application/json" \
     -d '{"force_rebuild": true}'
```

---

## Security Notes

- API keys live **only** in `backend/.env` — never in frontend code
- `backend/.env` is listed in `.gitignore` — never commit it
- CORS is configured for `http://localhost:3000` in development; update `app/main.py` for production origins
- The `/api/ingest` endpoint should be protected (e.g., IP allowlist or API token) before exposing publicly

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run `pytest` — all tests must pass
5. Open a pull request

---

## License

MIT

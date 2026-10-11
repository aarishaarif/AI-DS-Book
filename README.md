# AI & Data Science Learning Book

An interactive, full-stack educational platform that combines a nine-module AI/Data Science book with a strictly book-grounded RAG chatbot.

> **Disclaimer:** This is an independent educational project and is **not** officially affiliated with Saylani AI & Data Science or any other organisation.

---

## What's Inside

| Layer | Technology |
|---|---|
| Book / Frontend | Docusaurus 3, React, TypeScript |
| Backend API | Python 3.11, FastAPI, Pydantic v2 |
| Embeddings | Google Gemini (`models/gemini-embedding-001`) |
| Generation | Google Gemini (`gemini-2.0-flash-exp`) |
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
│   ├── docs/                    # Symlink / copy of docs/
│   ├── src/
│   │   ├── pages/               # Custom React pages (home, chat)
│   │   ├── components/          # Reusable components (Quiz, Progress)
│   │   └── css/custom.css
│   ├── static/img/              # Logo SVG, favicon
│   ├── docusaurus.config.ts
│   ├── sidebars.ts
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
│   │       └── store.py         # Qdrant integration
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

Leave `QDRANT_URL=http://localhost:6333` for local development.

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
  "files_discovered": 10,
  "chunks_indexed": 35,
  "errors": 0
}
```

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
pytest                            # run all 73 tests
pytest tests/test_api.py -v       # endpoint tests only
pytest tests/test_chunker.py      # chunker unit tests
pytest tests/test_rag_pipeline.py # RAG pipeline tests
pytest tests/test_evaluation.py   # RAG grounded eval & refusal benchmark
```

Tests run without real API credentials — all external services are mocked.

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
| `GEMINI_GENERATION_MODEL` | `gemini-2.0-flash-exp` | Generation model |
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

MIT — see `LICENSE` for details.

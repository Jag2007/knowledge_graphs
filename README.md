# Knowledge Graph Studio

This project converts uploaded PDFs into a Neo4j knowledge graph and lets you ask questions from a clean FastAPI UI.

The current backend is intentionally organized into small folders so the code is easier to read and maintain, while the app still starts with the same command:

```bash
uvicorn app:app --reload
```

## What The App Does

- Reads a PDF and extracts text
- Splits the text into sentence-aware overlapping chunks
- Uses Groq to extract `(subject, relation, object)` triples from each chunk
- Cleans, normalizes, deduplicates, and stores those triples in Neo4j
- Tracks the currently uploaded document so answers stay scoped to that PDF
- Retrieves answers from the graph using entity/relation-aware search and multi-hop path lookup
- Serves a grey/white frontend where you can upload PDFs, ask questions, and inspect JSON

## Folder Structure

```text
Knowledge Graphs/
├── app.py
├── extractor.py
├── graph.py
├── query_engine.py
├── utils.py
├── document_store.py
├── kg_app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   ├── query_engine.py
│   │   └── utils.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── graph.py
│   └── state/
│       ├── __init__.py
│       └── document_store.py
├── static/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── tests/
│   └── test_smoke.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Why Root Files Still Exist

The root files `app.py`, `extractor.py`, `graph.py`, `query_engine.py`, `utils.py`, and `document_store.py` are tiny compatibility wrappers that import the real code from `kg_app/`.

That keeps the project simple to run and avoids breaking `uvicorn app:app --reload`, while still giving us a cleaner folder-based codebase.

## Module Guide

- `kg_app/api/server.py`
  - FastAPI routes, PDF upload flow, question endpoint, and frontend serving
- `kg_app/core/extractor.py`
  - Groq triple extraction, JSON cleanup, relation/entity validation, and context-preserving triple enrichment
- `kg_app/core/query_engine.py`
  - Graph retrieval, entity/relation ranking, multi-hop path answering, and answer formatting
- `kg_app/core/utils.py`
  - PDF text extraction, sentence splitting, chunking, JSON recovery helpers, and relation normalization
- `kg_app/db/graph.py`
  - Neo4j connection, constraints, document-scoped triple writes, and graph search helpers
- `kg_app/state/document_store.py`
  - Active document tracking so `/ask` uses the most recently uploaded PDF
- `static/`
  - Frontend UI
- `tests/test_smoke.py`
  - API and retrieval regression tests

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env`

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

For Neo4j Aura, use your Aura URI in `NEO4J_URI`.

## Run The App

From the project root:

```bash
uvicorn app:app --reload
```

Open:

- Frontend: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## How To Use

1. Upload a PDF in the left panel
2. Click **Build Knowledge Graph**
3. Ask a question in the right panel
4. Click **View JSON** if you want to inspect request/response payloads

## API Endpoints

### `POST /upload_pdf`

Uploads a PDF, extracts triples, stores them in Neo4j, and marks that PDF as the active document.

### `POST /ask`

Accepts:

```json
{
  "question": "What does Indian Constitution guarantee?"
}
```

Returns a document-scoped answer plus query/debug metadata.

## Local Testing

Run the test suite:

```bash
python -m unittest tests/test_smoke.py
```

Compile check:

```bash
python -m py_compile app.py extractor.py graph.py query_engine.py utils.py document_store.py kg_app/api/server.py kg_app/core/extractor.py kg_app/core/query_engine.py kg_app/core/utils.py kg_app/db/graph.py kg_app/state/document_store.py tests/test_smoke.py
```

## Useful Environment Knobs

You can tune upload chunking and worker behavior with environment variables:

```bash
export KG_TARGET_WORDS=550
export KG_MIN_WORDS=380
export KG_MAX_WORDS=700
export KG_OVERLAP_WORDS=80
export KG_UPLOAD_WORKERS=1
```

Groq extraction logging:

```bash
export KG_DEBUG_LLM_OUTPUT=1
```

Set it to `0` if you want quieter terminal logs.

## Troubleshooting

- **`GET /favicon.ico 404`**
  - Harmless. The browser is just requesting a favicon that the app does not provide.

- **Upload says no readable text was found**
  - The PDF may be image-only/scanned. This app currently expects extractable text PDFs.

- **Upload extracts too few triples**
  - Try smaller chunks or more overlap using the `KG_*` env vars above.
  - Also check whether the PDF text itself is being extracted cleanly.

- **Answers seem unrelated**
  - Re-upload the PDF after code changes so the active document graph is rebuilt.
  - If Neo4j still contains older test data, remember that `/ask` is scoped to the most recently uploaded document through the active document tracker.

- **Groq rate limit errors**
  - The extractor retries, but large PDFs can still take time on strict limits.
  - Reduce parallelism with `KG_UPLOAD_WORKERS=1` and use smaller chunks if needed.

## Git Notes

`.gitignore` already excludes:

- `.env`
- local active-document/cache files
- root-level test PDFs
- virtual environments
- Python cache folders
- logs and editor/OS files

So you can safely keep local test PDFs in the project root without committing them.

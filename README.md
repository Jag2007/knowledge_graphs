# Knowledge Graph Studio

A professional FastAPI-based Knowledge Graph application that converts PDFs into a Neo4j graph, then answers questions using a **hybrid retrieval pipeline**:

- **Graph retrieval** for precise entity and relation lookup
- **Semantic chunk retrieval** for broader context and fuzzy question understanding

The project is organized into small, maintainable folders while still keeping the entrypoint simple:

```bash
uvicorn app:app --reload
```

## Highlights

- PDF upload to knowledge graph pipeline
- Sentence-aware chunking with overlap
- Triple extraction using Groq
- Triple cleaning, normalization, and deduplication
- Neo4j-backed document-scoped graph storage
- Hybrid retrieval:
  - graph-first lookup
  - semantic chunk fallback
  - multi-hop path support
- Clean FastAPI frontend with JSON inspection
- Local smoke tests for regression coverage

## Tech Stack

### Backend

- **Python**
- **FastAPI**
- **Uvicorn**
- **python-dotenv**

### Knowledge Graph

- **Neo4j**
- **neo4j Python driver**

### LLM / Extraction

- **Groq API**

### PDF Processing

- **PyMuPDF**

### Testing

- **unittest**
- **FastAPI TestClient**

## Architecture

```text
PDF
 -> text extraction
 -> sentence-aware chunking
 -> Groq triple extraction
 -> cleaning + normalization + deduplication
 -> Neo4j document-scoped storage
 -> hybrid retrieval
    -> graph lookup
    -> semantic chunk lookup
    -> multi-hop fallback
 -> final natural-language answer
```

## Project Structure

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
│       ├── chunk_store.py
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

## Why The Root Files Still Exist

The root-level files:

- `app.py`
- `extractor.py`
- `graph.py`
- `query_engine.py`
- `utils.py`
- `document_store.py`

are intentionally kept as **small compatibility wrappers**.

That gives us two benefits:

1. `uvicorn app:app --reload` still works exactly as expected
2. the real implementation lives in a cleaner package layout under `kg_app/`

## What Each Folder Does

### `kg_app/api/`

API routes and frontend serving.

- [server.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/api/server.py)
  - upload flow
  - ask flow
  - active-document handling
  - frontend serving

### `kg_app/core/`

Core processing and retrieval logic.

- [extractor.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/core/extractor.py)
  - Groq triple extraction
  - JSON recovery
  - triple validation
  - contextual triple linking

- [query_engine.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/core/query_engine.py)
  - graph retrieval
  - semantic chunk retrieval
  - hybrid answer selection
  - multi-hop path formatting

- [utils.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/core/utils.py)
  - PDF text extraction
  - sentence splitting
  - chunking
  - relation normalization
  - JSON helpers

### `kg_app/db/`

Database integration.

- [graph.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/db/graph.py)
  - Neo4j connection
  - schema setup
  - document storage
  - triple insertion
  - graph search methods

### `kg_app/state/`

Local state used by the app runtime.

- [document_store.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/state/document_store.py)
  - remembers the currently active uploaded document

- [chunk_store.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/state/chunk_store.py)
  - stores document chunks used for semantic retrieval

### `static/`

Frontend files.

- [index.html](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/static/index.html)
- [style.css](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/static/style.css)
- [app.js](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/static/app.js)

### `tests/`

Automated regression tests.

- [test_smoke.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/tests/test_smoke.py)

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

If you use Neo4j Aura, put your Aura URI in `NEO4J_URI`.

## Run The App

```bash
uvicorn app:app --reload
```

Open:

- Frontend: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## How It Works

### 1. Upload a PDF

The upload flow:

- extracts PDF text
- builds overlapping sentence-aware chunks
- runs Groq triple extraction on each chunk
- cleans and deduplicates triples
- stores triples and summary in Neo4j
- stores chunks locally for semantic retrieval
- marks the uploaded file as the current active document

### 2. Ask a Question

The question flow:

- scopes everything to the active uploaded document
- tries graph retrieval first
- tries entity neighborhood lookup
- tries direct relation matching
- tries semantic triple matching
- tries multi-hop graph paths
- uses semantic chunk retrieval when chunk context matches the question better

## Hybrid Retrieval Strategy

This project now uses a **hybrid retrieval approach**.

### Graph Retrieval

Best for:

- direct factual questions
- relation-specific questions
- entity-to-entity queries
- multi-hop graph questions

Examples:

- `Who is the father of the constitution?`
- `What does Indian Constitution guarantee?`
- `What is the capital of Karnataka?`

### Semantic Chunk Retrieval

Best for:

- fuzzy phrasing
- broader context questions
- paragraph-level meaning
- questions that are phrased differently from the exact triple wording

Examples:

- `What is a good constitution?`
- `What is this PDF talking about?`

### Why This Helps

Graph retrieval gives **precision**.

Semantic chunk retrieval gives **flexibility**.

Together they reduce:

- brittle keyword-only matching
- false negatives on paraphrased questions
- poor answers when the graph is too sparse for a broad question

## API Endpoints

### `POST /upload_pdf`

Uploads a PDF and builds the graph.

Response includes:

- `chunks_processed`
- `triples_added`
- `document_id`
- `summary`
- `sample_triples`

### `POST /ask`

Accepts:

```json
{
  "question": "What does Indian Constitution guarantee?"
}
```

Returns:

- `answer`
- `results`
- `query`
- `steps`
- `debug`

## Example Questions

### Direct factual

- `Who is the father of the constitution?`
- `What does Indian Constitution guarantee?`
- `What does festival include?`

### Entity overview

- `Indian Culture`
- `What is Indian Constitution?`
- `What is separation of powers?`

### Broader/semantic

- `What is this PDF talking about?`
- `What is a good constitution?`
- `What do you know about India and Pakistan separation?`

### Multi-hop / indirect

- `Constituent Assembly adopted on`
- `When did Indian National Movement start?`

## Testing

Run the smoke suite:

```bash
python -m unittest tests/test_smoke.py
```

Compile check:

```bash
python -m py_compile app.py extractor.py graph.py query_engine.py utils.py document_store.py kg_app/api/server.py kg_app/core/extractor.py kg_app/core/query_engine.py kg_app/core/utils.py kg_app/db/graph.py kg_app/state/document_store.py kg_app/state/chunk_store.py tests/test_smoke.py
```

## Tunable Environment Variables

These help control chunking and upload throughput:

```bash
export KG_TARGET_WORDS=550
export KG_MIN_WORDS=380
export KG_MAX_WORDS=700
export KG_OVERLAP_WORDS=80
export KG_UPLOAD_WORKERS=1
```

Groq debug logging:

```bash
export KG_DEBUG_LLM_OUTPUT=1
```

Set it to `0` if you want quieter logs.

## Troubleshooting

### `GET /favicon.ico 404`

Harmless browser request. It does not affect the app.

### No readable text found

The PDF is likely scanned/image-only. The current pipeline expects extractable text.

### Too few triples

Try:

- smaller chunks
- more overlap
- cleaner PDFs
- lower upload worker count if rate limits are interfering

### Answers feel unrelated

Re-upload the PDF after code changes so:

- triples are rebuilt
- chunks are rebuilt
- the active document points to the newest upload

### Groq rate limits

Large PDFs may slow down under strict account limits. If needed:

- reduce worker count with `KG_UPLOAD_WORKERS=1`
- use smaller chunk sizes
- retry with a cleaner PDF

## Git Hygiene

The project already ignores non-essential and sensitive files in [.gitignore](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/.gitignore), including:

- `.env`
- active document state
- semantic chunk cache
- local triple cache
- test PDFs in the root
- virtual environments
- Python cache folders
- logs
- editor/OS files

That keeps the repository clean and safer to share.

## Professional Notes

This codebase is now structured to be:

- easier to navigate
- safer to run locally
- clearer to document
- simpler to extend

The hybrid retrieval layer is especially useful because it keeps the app practical for real user phrasing instead of relying only on exact graph matches.

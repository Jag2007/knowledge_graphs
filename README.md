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
- Hybrid chunking using headings, paragraphs, sentences, hierarchy, and overlap
- Rich JSON chunks with summaries, keywords, sections, and page numbers
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

### Semantic Retrieval

- Local normalized chunk embeddings
- `sentence-transformers` backend with BGE by default
- Safe local fallback if the transformer model cannot load
- Weighted hybrid scoring

### PDF Processing

- **PyMuPDF**

### Testing

- **unittest**
- **FastAPI TestClient**

## Architecture

```text
PDF
 -> text extraction
 -> page-aware text extraction
 -> hybrid chunking
    -> heading detection
    -> paragraph grouping
    -> sentence-safe sliding window
    -> 15-25% overlap
    -> chunk summary + keyword metadata
 -> Groq triple extraction
 -> cleaning + normalization + deduplication
 -> Neo4j document-scoped storage
 -> hybrid retrieval
    -> graph lookup
    -> query embedding
    -> semantic chunk lookup
    -> weighted re-ranking
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

- extracts page-aware PDF text
- builds rich hybrid chunks using headings, paragraphs, and sentence boundaries
- applies sliding-window overlap so context carries between chunks
- precomputes summary and keyword metadata for each chunk
- runs Groq triple extraction on each chunk
- cleans and deduplicates triples
- stores triples and summary in Neo4j
- stores chunks locally for semantic retrieval
- stores normalized chunk embeddings alongside the chunk JSON
- marks the uploaded file as the current active document

Each stored chunk has this shape:

```json
{
  "id": "chunk_1",
  "text": "...",
  "summary": "...",
  "keywords": ["queue", "hospital"],
  "section": "Introduction",
  "page": 2,
  "embedding": [0.123, 0.456]
}
```

The chunker is designed to avoid splitting mid-sentence. It uses the PDF page as the top-level hierarchy, section headings when they are visible, paragraph blocks as semantic boundaries, and sentence-level sliding windows for overlap.

### 2. Ask a Question

The question flow:

- scopes everything to the active uploaded document
- tries graph retrieval first
- extracts likely entities from the query
- expands the query using graph entities
- tries entity neighborhood lookup
- tries direct relation matching
- tries semantic triple matching
- tries multi-hop graph paths
- converts the question into an embedding
- scores chunks with embedding similarity, keyword score, fuzzy score, and graph boost
- re-ranks the best chunk candidates and keeps the strongest context
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

### Hybrid Scoring Formula

```text
final_score =
  0.5 * embedding_similarity +
  0.2 * keyword_score +
  0.1 * fuzzy_score +
  0.2 * graph_score
```

The graph score boosts chunks that contain entities discovered from the Neo4j graph. The app also uses a retrieval threshold so weak chunk matches do not become false answers.

By default, `KG_EMBEDDING_BACKEND=auto` tries `sentence-transformers` first with `BAAI/bge-small-en-v1.5`. If the model cannot load, the app falls back to a local deterministic normalized embedding so it still runs.

```bash
export KG_EMBEDDING_BACKEND=auto
export KG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

If you want to force the transformer model:

```bash
export KG_EMBEDDING_BACKEND=sentence-transformers
```

If you want the lightweight fallback:

```bash
export KG_EMBEDDING_BACKEND=local
```

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
export KG_METADATA_MAX_CHUNKS=30
export KG_EMBEDDING_BACKEND=auto
export KG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

`KG_OVERLAP_WORDS` is converted into an overlap ratio for the hybrid chunker. The chunker clamps overlap into the recommended 15-25% range so context is preserved without creating too much duplication.

`KG_METADATA_MAX_CHUNKS` controls how many chunk excerpts are sent in the single ingestion-time metadata enrichment call. If Groq is unavailable or rate-limited, the app keeps the local heuristic summaries and keywords instead of failing the upload.

`KG_EMBEDDING_BACKEND=auto` tries sentence-transformers first and safely falls back to the local embedding. `KG_EMBEDDING_BACKEND=local` forces the built-in dependency-free embedding fallback.

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

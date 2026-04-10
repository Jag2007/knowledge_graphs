# Knowledge Graph Studio

A FastAPI app that converts PDFs into a Neo4j knowledge graph and answers questions using graph retrieval plus chunk-based semantic retrieval.

## Tech Stack

- Python
- FastAPI
- Neo4j
- Groq API
- PyMuPDF
- sentence-transformers
- unittest

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
│   ├── core/
│   ├── db/
│   └── state/
├── static/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

The root files are lightweight compatibility wrappers, so this still works:

```bash
uvicorn app:app --reload
```

## Main Features

- Upload a PDF and build a document-scoped knowledge graph
- Extract triples using Groq
- Store graph data in Neo4j
- Store chunks locally for semantic retrieval
- Ask natural-language questions about the active uploaded PDF
- Use graph lookup, semantic chunk lookup, and multi-hop fallback
- View clean answers in the UI and inspect JSON when needed

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

### 3. Create `.env`

```env
GROQ_API_KEY=your_groq_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

## Run

```bash
uvicorn app:app --reload
```

Open:

- Frontend: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## How It Works

### Upload flow

- extract text from the PDF
- split it into chunks
- generate triples from chunks
- clean and deduplicate triples
- store the graph in Neo4j
- store chunk metadata locally
- mark the uploaded PDF as the active document

### Question flow

- search only within the active uploaded PDF
- retrieve relevant graph facts
- retrieve relevant semantic chunks
- combine the best evidence
- return a natural-language answer

## Tests

Run the smoke tests with:

```bash
python -m unittest tests/test_smoke.py
```

## Important Files

- [app.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/app.py)
- [kg_app/api/server.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/api/server.py)
- [kg_app/core/extractor.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/core/extractor.py)
- [kg_app/core/query_engine.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/core/query_engine.py)
- [kg_app/core/utils.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/core/utils.py)
- [kg_app/db/graph.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/db/graph.py)
- [kg_app/state/chunk_store.py](/Users/jagruthipulumati/Desktop/Knowledge%20Graphs/kg_app/state/chunk_store.py)

## Notes

- Use Neo4j Aura or local Neo4j.
- Re-upload a PDF after major extraction or retrieval changes so the graph is rebuilt with the latest logic.
- Large PDFs will take longer because triple extraction depends on LLM calls.

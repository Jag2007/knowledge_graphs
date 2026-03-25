# Knowledge Graph Builder from PDF

This is a beginner-friendly Python project that:
1. Extracts text from an uploaded PDF
2. Chunks the text
3. Uses the Groq LLM to extract `(subject, relation, object)` triples from each chunk
4. Stores triples in Neo4j
5. Uses Groq again to generate Cypher queries to answer questions over the graph

It exposes a simple **FastAPI** backend:
* `POST /upload_pdf` to ingest a PDF
* `POST /ask` to query the graph with natural language

## Prerequisites
- Python 3.9+
- Neo4j running (Neo4j Desktop or AuraDB)
- Groq API key

## Setup (Quickstart)
1. Open a terminal and `cd` into the repository root (where `app.py` lives)
2. (Optional) Create/activate a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configure environment variables
Create a `.env` file in the repository root (this file is ignored by git).

Use:
```bash
GROQ_API_KEY=your_groq_api_key_here
NEO4J_URI=your_neo4j_uri_here
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

Notes:
- `NEO4J_URI` should be `bolt://localhost:7687` for local Neo4j, or the full AuraDB/managed DB URI (often starts with `neo4j+s://`) for hosted databases.
- During PDF ingestion you may see `RAW LLM OUTPUT:` lines in the server logs (used for debugging triple extraction).

## Run the server
From the repository root:
```bash
uvicorn app:app --reload
```

Swagger UI:
[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)

## API Usage

### 1. Upload a PDF
Endpoint: `POST /upload_pdf`

In Swagger: choose `upload_pdf` and upload a file.

Example with `curl` (adjust the filename):
```bash
curl -X POST "http://127.0.0.1:8000/upload_pdf" \
  -H "accept: application/json" \
  -F "file=@./your_document.pdf"
```

Response includes:
* `chunks_processed`
* `triples_added`
* `sample_triples` (up to 10)

### 2. Ask a question
Endpoint: `POST /ask`

Body:
```json
{
  "question": "What companies are mentioned in the file?"
}
```

The response includes:
* `query` (the Cypher generated)
* `results` (matched entities/relations)
* `answer` (a concise synthesized response)

## Notes / Troubleshooting
- If ingestion fails with `No structured knowledge extracted from document`, the LLM didn’t return any valid triples for the chunks you provided. Try a different PDF or ensure the PDF text is readable.
- Neo4j relationship types are derived from the extracted triple `relation` values, normalized into `UPPERCASE_WITH_UNDERSCORES` format.

from pathlib import Path
import asyncio
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils import extract_text_from_pdf, chunk_text
from extractor import extract_triples_groq
from graph import Neo4jGraph
from query_engine import ask_question

app = FastAPI(title="Knowledge Graph Builder")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

UPLOAD_TARGET_WORDS = int(os.environ.get("KG_TARGET_WORDS", "550"))
UPLOAD_MIN_WORDS = int(os.environ.get("KG_MIN_WORDS", "380"))
UPLOAD_MAX_WORDS = int(os.environ.get("KG_MAX_WORDS", "700"))
UPLOAD_OVERLAP_WORDS = int(os.environ.get("KG_OVERLAP_WORDS", "80"))
UPLOAD_WORKERS = max(1, min(2, int(os.environ.get("KG_UPLOAD_WORKERS", "1"))))

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
async def index():
    """Serve the frontend."""
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract knowledge triples, and save to Neo4j."""
    graph = None
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            return {
                "error": "Please upload a PDF file.",
                "trace": "",
            }

        content = await file.read()
        if not content:
            return {
                "error": "The uploaded PDF is empty.",
                "trace": "",
            }
        
        # 1. Process PDF into text
        text = extract_text_from_pdf(content)
        if not text.strip():
            return {
                "error": "No readable text was found in the uploaded PDF.",
                "trace": "",
            }
        
        # 2. Split text into larger chunks to reduce total LLM calls on big PDFs.
        chunks = chunk_text(
            text,
            target_words=UPLOAD_TARGET_WORDS,
            min_words=UPLOAD_MIN_WORDS,
            max_words=UPLOAD_MAX_WORDS,
            overlap_words=UPLOAD_OVERLAP_WORDS,
        )
        if not chunks:
            return {
                "error": "The document could not be split into usable text chunks.",
                "trace": "",
            }
        
        graph = Neo4jGraph()
        extracted_triples_debug = []
        all_triples = []
        seen = set()
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
            tasks = [loop.run_in_executor(executor, extract_triples_groq, chunk) for chunk in chunks]
            raw_batches = await asyncio.gather(*tasks, return_exceptions=True)

        triple_batches: list[list[dict]] = []
        for batch in raw_batches:
            if isinstance(batch, Exception):
                print(f"Chunk extraction failed: {batch}")
                triple_batches.append([])
            else:
                triple_batches.append(batch)

        # 3. Accumulate validated triples (dedup across the whole document)
        for triples in triple_batches:
            for t in triples:
                key = (t["subject"].strip().lower(), t["relation"].strip().lower(), t["object"].strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                all_triples.append(t)
                if len(extracted_triples_debug) < 10:
                    extracted_triples_debug.append((t["subject"], t["relation"], t["object"]))

        triples_added = graph.insert_triples(all_triples)
        
        # Failsafe exactly as requested
        if triples_added == 0:
            return {
                "error": "No structured knowledge was extracted from the document.",
                "trace": "",
                "debug": {
                    "chunks_processed": len(chunks),
                    "sample_triples": extracted_triples_debug,
                    "triples_added": triples_added,
                    "workers_used": UPLOAD_WORKERS,
                },
            }
        
        return {
            "message": "PDF processed successfully",
            "file_name": file.filename,
            "chunks_processed": len(chunks),
            "triples_added": triples_added,
            "sample_triples": extracted_triples_debug,
            "debug": {
                "file_name": file.filename,
                "chunks_processed": len(chunks),
                "sample_triples": extracted_triples_debug,
                "triples_added": triples_added,
                "workers_used": UPLOAD_WORKERS,
            },
        }

    except Exception as e:
        print("ERROR IN UPLOAD_PDF:")
        print(traceback.format_exc())
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
    finally:
        if graph is not None:
            try:
                graph.close()
            except Exception:
                pass

@app.post("/ask")
async def ask(q: QuestionRequest):
    """Ask a natural language question against the Knowledge Graph."""
    question = q.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please provide a question.")

    try:
        response = ask_question(question)
        results = response.get("results", [])
        answer = response.get("answer", "").strip()

        if not results:
            answer = "It is not in the uploaded document. Please check the text."
        elif not answer:
            answer = "A result was found in the uploaded document."

        return {
            "triples_added": response.get("triples_added", 0),
            "query": response.get("query", ""),
            "results": results,
            "answer": answer,
            "steps": response.get("steps_taken", response.get("steps", [])),
            "debug": {
                "question": question,
                "query": response.get("query", ""),
                "results": results,
                "steps": response.get("steps_taken", response.get("steps", [])),
                "triples_added": response.get("triples_added", 0),
            },
        }
    except Exception as e:
        print("ERROR IN ASK:")
        print(traceback.format_exc())
        return {
            "query": "",
            "results": [],
            "answer": "It is not in the uploaded document. Please check the text.",
            "debug": {
                "question": question,
                "error": str(e),
                "trace": traceback.format_exc(),
            },
        }

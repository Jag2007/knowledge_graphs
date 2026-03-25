import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from utils import extract_text_from_pdf, chunk_text
from extractor import extract_triples_groq
from graph import Neo4jGraph
from query_engine import ask_question

app = FastAPI(title="Knowledge Graph Builder")

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract knowledge triples, and save to Neo4j."""
    try:
        content = await file.read()
        
        # 1. Process PDF into text
        text = extract_text_from_pdf(content)
        
        # 2. Split text into chunks (200-400 words)
        chunks = chunk_text(text, target_words=300, min_words=200, max_words=400)
        
        graph = Neo4jGraph()
        extracted_triples_debug = []
        all_triples = []
        seen = set()

        # 3. Process each chunk
        for chunk in chunks:
            # Safely extract explicitly validated triples
            triples = extract_triples_groq(chunk)
            
            # 4. Accumulate validated triples (dedup across the whole document)
            for t in triples:
                key = (t["subject"].strip().lower(), t["relation"].strip().lower(), t["object"].strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                all_triples.append(t)
                if len(extracted_triples_debug) < 10:
                    extracted_triples_debug.append((t["subject"], t["relation"], t["object"]))

        triples_added = graph.insert_triples(all_triples)
        graph.close()
        
        # Failsafe exactly as requested
        if triples_added == 0:
            raise HTTPException(status_code=400, detail="No structured knowledge extracted from document")
        
        return {
            "message": "PDF processed successfully",
            "chunks_processed": len(chunks),
            "triples_added": triples_added,
            "sample_triples": extracted_triples_debug
        }

    except Exception as e:
        err_msg = str(e)
        if isinstance(e, HTTPException):
            raise
        print("ERROR IN UPLOAD_PDF:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=err_msg)

@app.post("/ask")
async def ask(q: QuestionRequest):
    """Ask a natural language question against the Knowledge Graph."""
    response = ask_question(q.question)
    
    # Return directly formatted debug logic (pipeline may be multi-hop).
    return {
        "triples_added": response.get("triples_added", 0),
        "query": response.get("query", ""),
        "results": response.get("results", []),
        "answer": response.get("answer", ""),
        "steps": response.get("steps_taken", response.get("steps", []))
    }

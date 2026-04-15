from pathlib import Path
import asyncio
import json
import os
import requests
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kg_app.core.utils import (
    extract_text_from_pdf,
    extract_pdf_pages,
    build_hybrid_chunks,
    extract_keywords,
    summarize_text,
)
from kg_app.core.extractor import extract_triples_llm, extract_triples_fallback, precompute_chunk_metadata
from kg_app.db.graph import GraphStore
from kg_app.core.query_engine import ask_question
from kg_app.state.document_store import set_active_document, get_active_document
from kg_app.state.chunk_store import save_document_chunks, get_document_chunks

app = FastAPI(title="Knowledge Graph Builder")
BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

UPLOAD_TARGET_WORDS = int(os.environ.get("KG_TARGET_WORDS", "550"))
UPLOAD_MIN_WORDS = int(os.environ.get("KG_MIN_WORDS", "380"))
UPLOAD_MAX_WORDS = int(os.environ.get("KG_MAX_WORDS", "700"))
UPLOAD_OVERLAP_WORDS = int(os.environ.get("KG_OVERLAP_WORDS", "80"))
UPLOAD_MAX_CHUNKS = max(1, int(os.environ.get("KG_MAX_CHUNKS", "10")))
UPLOAD_WORKERS = max(1, min(2, int(os.environ.get("KG_UPLOAD_WORKERS", "1"))))


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


VERBOSE_BACKEND_LOGS = _env_flag("KG_VERBOSE_BACKEND_LOGS", "1")


def _backend_log(message: str) -> None:
    if VERBOSE_BACKEND_LOGS:
        print(f"[server] {message}", flush=True)


def _use_hebbrix_native() -> bool:
    return bool(os.environ.get("HEBBRIX_API_KEY")) and _env_flag("HEBBRIX_NATIVE_MODE", "1")


def _hebbrix_base_url() -> str:
    return os.environ.get("HEBBRIX_BASE_URL", "https://api.hebbrix.com/v1").rstrip("/")


def _hebbrix_timeout() -> int:
    return max(10, int(os.environ.get("HEBBRIX_HTTP_TIMEOUT_SECONDS", "120")))


def _encode_hebbrix_active_id(collection_id: str, document_id: str) -> str:
    return f"hebbrix:{collection_id}:{document_id}"


def _decode_hebbrix_active_id(value: str) -> dict:
    text = str(value or "").strip()
    if not text.startswith("hebbrix:"):
        return {}
    _, collection_id, document_id = text.split(":", 2)
    return {"collection_id": collection_id, "document_id": document_id}


def _hebbrix_headers(content_type: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {os.environ.get('HEBBRIX_API_KEY', '').strip()}",
        "Accept": "application/json",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _extract_hebbrix_id(payload: dict, *keys: str) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_key in ("id", "document_id", "collection_id"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    for container_key in ("document", "collection", "job", "result"):
        container = payload.get(container_key)
        if isinstance(container, dict):
            for key in keys or ("id",):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for nested_key in ("id", "document_id", "collection_id", "job_id"):
                nested_value = container.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        for key in keys or ("id",):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _extract_hebbrix_document(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    document = payload.get("document")
    if isinstance(document, dict):
        return document
    data = payload.get("data")
    if isinstance(data, dict):
        nested_document = data.get("document")
        if isinstance(nested_document, dict):
            return nested_document
        if any(key in data for key in ("id", "status", "collection_id", "original_filename")):
            return data
    if any(key in payload for key in ("id", "status", "collection_id", "original_filename")):
        return payload
    return {}


def _normalise_hebbrix_items(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "data", "chunks", "memories"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _hebbrix_request(method: str, path: str, *, payload=None, multipart: tuple[dict, tuple[str, bytes, str]] | None = None):
    url = f"{_hebbrix_base_url()}{path}"
    try:
        if multipart is not None:
            fields, file_tuple = multipart
            file_name, file_bytes, content_type = file_tuple
            response = requests.request(
                method.upper(),
                url,
                headers=_hebbrix_headers(),
                data={key: str(value) for key, value in fields.items()},
                files={"file": (file_name, file_bytes, content_type)},
                timeout=_hebbrix_timeout(),
            )
        else:
            response = requests.request(
                method.upper(),
                url,
                headers=_hebbrix_headers("application/json") if payload is not None else _hebbrix_headers(),
                json=payload,
                timeout=_hebbrix_timeout(),
            )

        if response.status_code >= 400:
            try:
                parsed = response.json()
                details = parsed.get("error", {}) if isinstance(parsed, dict) else {}
                message = details.get("message") or response.text or response.reason
            except Exception:
                message = response.text or response.reason
            allow_header = response.headers.get("Allow")
            if allow_header:
                message = f"{message} (Allow: {allow_header})"
            raise RuntimeError(f"Hebbrix API error ({response.status_code}) on {path}: {message}")

        if not response.text.strip():
            return {}
        return response.json()
    except Exception as error:
        raise RuntimeError(f"Hebbrix request failed for {path}: {error}") from error


def _create_hebbrix_collection(file_name: str) -> dict:
    name = f"{Path(file_name).stem}-{uuid.uuid4().hex[:8]}"
    payload = _hebbrix_request("POST", "/collections", payload={"name": name})
    collection_id = _extract_hebbrix_id(payload, "id", "collection_id")
    if not collection_id:
        raise RuntimeError("Hebbrix collection creation did not return an id.")
    _backend_log(f"Created Hebbrix collection {collection_id} for upload.")
    return payload


def _upload_hebbrix_document(file_name: str, content: bytes, collection_id: str) -> dict:
    upload_paths = ["/documents/upload", "/documents"]
    last_error = None
    for path in upload_paths:
        try:
            payload = _hebbrix_request(
                "POST",
                path,
                multipart=(
                    {"collection_id": collection_id, "extract_entities": "true"},
                    (file_name, content, "application/pdf"),
                ),
            )
            document_id = _extract_hebbrix_id(payload, "id", "document_id")
            if not document_id:
                raise RuntimeError(f"Hebbrix upload response from {path} did not return a document id.")
            _backend_log(f"Uploaded PDF to Hebbrix document {document_id} via {path}.")
            return payload
        except Exception as error:
            last_error = error
            message = str(error)
            retryable_path = any(token in message for token in ("(404)", "(405)", "Method Not Allowed", "Not Found"))
            if path != upload_paths[-1] and retryable_path:
                _backend_log(f"Hebbrix upload path {path} was rejected. Trying the next documented upload path.")
                continue
            raise
    raise RuntimeError(f"Hebbrix document upload failed: {last_error}")


def _wait_for_hebbrix_document(document_id: str) -> dict:
    max_wait = max(20, int(os.environ.get("HEBBRIX_DOCUMENT_WAIT_SECONDS", "180")))
    poll_seconds = max(2, int(os.environ.get("HEBBRIX_DOCUMENT_POLL_SECONDS", "4")))
    started = time.time()
    last_payload = {}
    while True:
        payload = _hebbrix_request("GET", f"/documents/{document_id}")
        last_payload = payload if isinstance(payload, dict) else {}
        document = _extract_hebbrix_document(last_payload)
        status = str(document.get("status", "") or last_payload.get("status", "")).strip().lower()
        if status in {"completed", "processed", "ready", "indexed", "searchable", "enriched"} or not status:
            if status:
                _backend_log(f"Hebbrix document {document_id} status: {status}.")
            return last_payload
        if status == "failed":
            raise RuntimeError(
                document.get("error_message")
                or last_payload.get("error_message")
                or f"Hebbrix document processing failed for {document_id}."
            )
        if time.time() - started > max_wait:
            raise RuntimeError(f"Timed out waiting for Hebbrix document {document_id} to finish processing.")
        _backend_log(f"Hebbrix document {document_id} status: {status}. Waiting {poll_seconds}s.")
        time.sleep(poll_seconds)


def _fetch_hebbrix_chunks(document_id: str) -> list[dict]:
    payload = _hebbrix_request("GET", f"/documents/{document_id}/chunks")
    items = _normalise_hebbrix_items(payload)
    chunks: list[dict] = []
    for index, item in enumerate(items, start=1):
        text = str(item.get("text") or item.get("content") or item.get("chunk_text") or "").strip()
        if not text:
            continue
        chunks.append(
            {
                "id": str(item.get("id") or f"chunk_{index}"),
                "text": text,
                "summary": str(item.get("summary", "")).strip(),
                "keywords": item.get("keywords", []) if isinstance(item.get("keywords", []), list) else [],
                "section": str(item.get("section", "Document")).strip() or "Document",
                "page": item.get("page"),
            }
        )
    return chunks


def _build_hebbrix_summary(file_name: str, chunks: list[dict], document_payload: dict) -> str:
    document = _extract_hebbrix_document(document_payload)
    explicit = str(document.get("summary", "") or document_payload.get("summary", "")).strip()
    if explicit:
        return explicit
    if chunks:
        return summarize_text(" ".join(chunk.get("text", "") for chunk in chunks[:3]))
    return f"The uploaded document {file_name} was indexed by Hebbrix."


def _search_hebbrix(question: str, collection_id: str) -> dict:
    return _hebbrix_request(
        "POST",
        "/search",
        payload={
            "query": question,
            "collection_id": collection_id,
            "limit": 8,
        },
    )


def _chat_hebbrix(question: str, collection_id: str) -> dict:
    model = os.environ.get("HEBBRIX_CHAT_MODEL", "gpt-5-nano")
    return _hebbrix_request(
        "POST",
        "/chat/completions",
        payload={
            "model": model,
            "collection_id": collection_id,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "stream": False,
        },
    )


def _query_hebbrix_graph(question: str, collection_id: str) -> dict:
    return _hebbrix_request(
        "POST",
        "/knowledge-graph/query",
        payload={
            "query": question,
            "collection_id": collection_id,
            "limit": 8,
        },
    )


def _extract_answer_from_hebbrix(payload: dict) -> str:
    if isinstance(payload.get("choices"), list):
        for choice in payload.get("choices", []):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    for key in ("answer", "response", "content", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    items = _normalise_hebbrix_items(payload)
    snippets: list[str] = []
    for item in items[:3]:
        for key in ("snippet", "summary", "content", "text"):
            value = str(item.get(key, "")).strip()
            if value:
                snippets.append(value)
                break
    return " ".join(snippets).strip()


def _format_hebbrix_results(payload: dict) -> list[dict]:
    items = _normalise_hebbrix_items(payload)
    results: list[dict] = []
    for item in items[:8]:
        text = str(item.get("snippet") or item.get("content") or item.get("text") or "").strip()
        score = item.get("score")
        results.append(
            {
                "text": text,
                "score": score,
                "source": item.get("source") or item.get("id") or "hebbrix",
            }
        )
    return results


def _merge_chunks_for_limit(chunks: list[dict], max_chunks: int) -> list[dict]:
    if len(chunks) <= max_chunks:
        return chunks

    merged_chunks: list[dict] = []
    total = len(chunks)
    groups = min(max_chunks, total)
    for index in range(groups):
        start = (index * total) // groups
        end = ((index + 1) * total) // groups
        group = chunks[start:end]
        if not group:
            continue

        text = " ".join(str(chunk.get("text", "")).strip() for chunk in group if str(chunk.get("text", "")).strip()).strip()
        if not text:
            continue

        section = next((str(chunk.get("section", "")).strip() for chunk in group if str(chunk.get("section", "")).strip()), "Document")
        pages = [chunk.get("page") for chunk in group if chunk.get("page") is not None]
        merged_chunks.append(
            {
                "id": f"chunk_{len(merged_chunks) + 1}",
                "text": text,
                "summary": summarize_text(text),
                "keywords": extract_keywords(text),
                "section": section or "Document",
                "page": min(pages) if pages else None,
            }
        )

    return merged_chunks

class QuestionRequest(BaseModel):
    question: str


def build_document_summary(file_name: str, triples: list[dict]) -> str:
    if not triples:
        return f"The uploaded document {file_name} was processed, but no structured summary could be created."

    topics: list[str] = []
    seen = set()
    for triple in triples:
        for value in (triple.get("subject", ""), triple.get("object", "")):
            text = str(value).strip().replace("_", " ")
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                topics.append(text)
            if len(topics) >= 6:
                break
        if len(topics) >= 6:
            break

    if not topics:
        return f"The uploaded document {file_name} contains structured facts."
    if len(topics) == 1:
        return f"The uploaded document mainly discusses {topics[0]}."
    return "The uploaded document mainly discusses " + ", ".join(topics[:-1]) + f", and {topics[-1]}."


@app.get("/")
async def index():
    """Serve the frontend."""
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract knowledge triples, and save to MongoDB."""
    graph = None
    try:
        _backend_log(f"Upload started for file '{file.filename or 'unknown'}'.")
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
        _backend_log(f"Read {len(content)} bytes from upload.")

        if _use_hebbrix_native():
            _backend_log("Using Hebbrix-native document pipeline for this upload.")
            collection_payload = _create_hebbrix_collection(file.filename)
            collection_id = _extract_hebbrix_id(collection_payload, "id", "collection_id")
            uploaded_document = _upload_hebbrix_document(file.filename, content, collection_id)
            hebbrix_document_id = _extract_hebbrix_id(uploaded_document, "id", "document_id")
            document_payload = _wait_for_hebbrix_document(hebbrix_document_id)
            document_record = _extract_hebbrix_document(document_payload)
            active_document_id = _encode_hebbrix_active_id(collection_id, hebbrix_document_id)
            hebbrix_chunks = _fetch_hebbrix_chunks(hebbrix_document_id)
            set_active_document(active_document_id, file.filename)
            summary = _build_hebbrix_summary(file.filename, hebbrix_chunks, document_payload)
            graph_items = int(
                document_record.get("relationships_count")
                or document_record.get("triples_count")
                or document_record.get("entities_count")
                or document_record.get("memory_count")
                or len(hebbrix_chunks)
            )
            _backend_log(
                f"Hebbrix-native upload complete with {len(hebbrix_chunks)} chunks and {graph_items} graph items."
            )
            return {
                "message": "PDF uploaded to Hebbrix and indexed successfully.",
                "file_name": file.filename,
                "chunks_processed": len(hebbrix_chunks),
                "triples_added": graph_items,
                "document_id": active_document_id,
                "summary": summary,
                "sample_triples": [],
                "debug": {
                    "provider": "hebbrix",
                    "collection_id": collection_id,
                    "hebbrix_document_id": hebbrix_document_id,
                    "hebbrix_status": document_record.get("status"),
                    "chunks_processed": len(hebbrix_chunks),
                    "graph_items": graph_items,
                    "summary": summary,
                },
            }
        
        # 1. Process PDF into page-aware text
        try:
            pages = extract_pdf_pages(content)
        except Exception:
            pages = [{"page": 1, "text": extract_text_from_pdf(content)}]
        _backend_log(f"Extracted readable text from {len(pages)} pages.")
        text = "\n".join(page["text"] for page in pages)
        if not text.strip():
            return {
                "error": "No readable text was found in the uploaded PDF.",
                "trace": "",
            }
        
        # 2. Build rich hybrid chunks: headings + paragraphs + sentence-safe sliding overlap.
        chunks = build_hybrid_chunks(
            pages,
            target_words=UPLOAD_TARGET_WORDS,
            min_words=UPLOAD_MIN_WORDS,
            max_words=UPLOAD_MAX_WORDS,
            overlap_ratio=UPLOAD_OVERLAP_WORDS / max(1, UPLOAD_TARGET_WORDS),
        )
        _backend_log(f"Built {len(chunks)} hybrid chunks for upload processing.")
        if len(chunks) > UPLOAD_MAX_CHUNKS:
            original_count = len(chunks)
            chunks = _merge_chunks_for_limit(chunks, UPLOAD_MAX_CHUNKS)
            _backend_log(
                f"Reduced upload workload from {original_count} chunks to {len(chunks)} chunks for Hebbrix-friendly processing."
            )
        if not chunks:
            return {
                "error": "The document could not be split into usable text chunks.",
                "trace": "",
            }

        chunks = precompute_chunk_metadata(chunks)
        _backend_log("Chunk metadata enrichment step finished.")
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        graph = GraphStore()
        document_id = str(uuid.uuid4())
        _backend_log(f"Created document id {document_id}.")
        extracted_triples_debug = []
        all_triples = []
        seen = set()
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
            _backend_log(
                f"Starting LLM triple extraction for {len(chunk_texts)} chunks with {UPLOAD_WORKERS} worker(s)."
            )
            tasks = [loop.run_in_executor(executor, extract_triples_llm, chunk_text) for chunk_text in chunk_texts]
            raw_batches = await asyncio.gather(*tasks, return_exceptions=True)

        triple_batches: list[list[dict]] = []
        for index, batch in enumerate(raw_batches, start=1):
            if isinstance(batch, Exception):
                print(f"[server] Chunk {index}/{len(raw_batches)} extraction failed: {batch}", flush=True)
                fallback_triples = extract_triples_fallback(chunk_texts[index - 1])
                _backend_log(
                    f"Chunk {index}/{len(raw_batches)} fallback extraction returned {len(fallback_triples)} triples."
                )
                triple_batches.append(fallback_triples)
            else:
                if not batch:
                    fallback_triples = extract_triples_fallback(chunk_texts[index - 1])
                    _backend_log(
                        f"Chunk {index}/{len(raw_batches)} fallback extraction returned {len(fallback_triples)} triples."
                    )
                    triple_batches.append(fallback_triples)
                    continue
                _backend_log(
                    f"Chunk {index}/{len(raw_batches)} extraction returned {len(batch)} triples."
                )
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
        _backend_log(f"Accumulated {len(all_triples)} unique triples across all chunks.")

        summary = build_document_summary(file.filename, all_triples)
        graph.store_document(document_id, file.filename, summary)
        triples_added = graph.insert_triples(all_triples, document_id)
        _backend_log(
            f"Stored document summary and inserted {triples_added} triples into MongoDB."
        )
        save_document_chunks(document_id, file.filename, chunks)
        set_active_document(document_id, file.filename)
        _backend_log("Saved chunk state and set the uploaded document as active.")
        
        # Failsafe exactly as requested
        if triples_added == 0:
            return {
                "error": "No structured knowledge was extracted from the document.",
                "note": "The PDF text chunks were still saved for semantic retrieval.",
                "trace": "",
                "debug": {
                    "chunks_processed": len(chunks),
                    "sample_triples": extracted_triples_debug,
                    "triples_added": triples_added,
                    "workers_used": UPLOAD_WORKERS,
                    "document_id": document_id,
                },
            }
        
        return {
            "message": "PDF processed successfully. The knowledge graph is ready.",
            "file_name": file.filename,
            "chunks_processed": len(chunks),
            "triples_added": triples_added,
            "document_id": document_id,
            "summary": summary,
            "sample_triples": extracted_triples_debug,
            "debug": {
                "file_name": file.filename,
                "chunks_processed": len(chunks),
                "sample_triples": extracted_triples_debug,
                "triples_added": triples_added,
                "workers_used": UPLOAD_WORKERS,
                "document_id": document_id,
                "summary": summary,
            },
        }

    except Exception as e:
        print("ERROR IN UPLOAD_PDF:", flush=True)
        print(traceback.format_exc(), flush=True)
        raise HTTPException(
            status_code=502,
            detail={
                "error": str(e),
                "trace": traceback.format_exc(),
            },
        ) from e
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
        _backend_log(f"Received question: {question}")
        active_document = get_active_document()
        document_id = str(active_document.get("document_id", "")).strip()
        if not document_id:
            return {
                "query": "",
                "results": [],
                "answer": "Please upload a PDF before asking a question.",
                "debug": {
                    "question": question,
                    "error": "No active document found.",
                },
            }

        hebbrix_ref = _decode_hebbrix_active_id(document_id)
        if hebbrix_ref and _use_hebbrix_native():
            collection_id = hebbrix_ref.get("collection_id", "")
            _backend_log(
                f"Using Hebbrix-native retrieval for collection {collection_id} and document {hebbrix_ref.get('document_id', '')}."
            )
            chat_payload = _chat_hebbrix(question, collection_id)
            answer = _extract_answer_from_hebbrix(chat_payload)
            search_payload = _search_hebbrix(question, collection_id)
            results = _format_hebbrix_results(search_payload)
            if not answer:
                graph_payload = _query_hebbrix_graph(question, collection_id)
                if graph_payload:
                    results = results or _format_hebbrix_results(graph_payload)
                    answer = _extract_answer_from_hebbrix(graph_payload)
            if not answer and results:
                answer = " ".join(result.get("text", "") for result in results[:3]).strip()
            if not answer:
                answer = "It is not in the uploaded document. Please check the text."
            return {
                "triples_added": 0,
                "query": question,
                "results": results,
                "answer": answer,
                "steps": ["Hebbrix chat completions", "Hebbrix search"],
                "debug": {
                    "provider": "hebbrix",
                    "mode": "native-chat-search",
                    "collection_id": collection_id,
                    "document_id": document_id,
                    "results_count": len(results),
                },
            }
        elif _use_hebbrix_native():
            return {
                "query": "",
                "results": [],
                "answer": "Please upload a PDF in Hebbrix mode before asking a question.",
                "debug": {
                    "question": question,
                    "error": "No Hebbrix-backed active document found.",
                    "document_id": document_id,
                },
            }

        saved_chunks = get_document_chunks(document_id)
        graph = GraphStore()
        triple_count = graph.count_triples(document_id)
        graph.close()
        if not saved_chunks and triple_count <= 0:
            return {
                "query": "",
                "results": [],
                "answer": "Please upload and finish processing a PDF before asking a question.",
                "debug": {
                    "question": question,
                    "error": "Active document has no saved chunks or graph triples.",
                    "document_id": document_id,
                },
            }
        _backend_log(
            f"Answering against document {document_id} with {len(saved_chunks)} saved chunks and {triple_count} triples."
        )

        response = ask_question(question, document_id=document_id)
        _backend_log(
            f"Question answered against document {document_id} with {len(response.get('results', []))} result rows."
        )
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
                "document_id": document_id,
                "active_file_name": active_document.get("file_name", ""),
            },
        }
    except Exception as e:
        print("ERROR IN ASK:", flush=True)
        print(traceback.format_exc(), flush=True)
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

import json
from pathlib import Path

STATE_PATH = Path(__file__).resolve().parent / ".document_chunks.json"


def _load_payload() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _save_payload(payload: dict) -> None:
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def save_document_chunks(document_id: str, file_name: str, chunks: list[str]) -> None:
    payload = _load_payload()
    payload[document_id] = {
        "file_name": file_name,
        "chunks": [str(chunk).strip() for chunk in chunks if str(chunk).strip()],
    }
    _save_payload(payload)


def get_document_chunks(document_id: str) -> list[str]:
    payload = _load_payload()
    document = payload.get(document_id, {})
    chunks = document.get("chunks", [])
    if isinstance(chunks, list):
        return [str(chunk).strip() for chunk in chunks if str(chunk).strip()]
    return []


def clear_document_chunks(document_id: str) -> None:
    payload = _load_payload()
    if document_id in payload:
        payload.pop(document_id, None)
        _save_payload(payload)


def clear_all_document_chunks() -> None:
    if STATE_PATH.exists():
        try:
            STATE_PATH.unlink()
        except Exception:
            pass

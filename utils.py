import json
import fitz  # PyMuPDF
import re

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    # PyMuPDF can open from memory (bytes)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text() + "\n"
    return text

def chunk_text(
    text: str,
    target_words: int = 180,
    min_words: int = 120,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[str]:
    """Split text into overlapping sentence-aware chunks for better extraction quality."""
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    def flush_chunk() -> None:
        nonlocal current_sentences, current_word_count
        chunk = " ".join(current_sentences).strip()
        if chunk:
            chunks.append(chunk)

        overlap_buffer: list[str] = []
        overlap_count = 0
        for previous_sentence in reversed(current_sentences):
            previous_count = len(previous_sentence.split())
            if overlap_buffer and overlap_count + previous_count > overlap_words:
                break
            overlap_buffer.insert(0, previous_sentence)
            overlap_count += previous_count

        current_sentences = overlap_buffer
        current_word_count = overlap_count

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if sentence_word_count == 0:
            continue

        if current_sentences and current_word_count + sentence_word_count > max_words:
            flush_chunk()

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

        if current_word_count >= target_words:
            flush_chunk()

    if current_sentences:
        chunk = " ".join(current_sentences).strip()
        if chunk:
            chunks.append(chunk)

    if len(chunks) >= 2:
        last_words = len(chunks[-1].split())
        if last_words < min_words:
            prev_words = len(chunks[-2].split())
            if prev_words + last_words <= max_words:
                chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
                chunks.pop()

    deduped_chunks: list[str] = []
    seen = set()
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_chunks.append(normalized)

    return deduped_chunks

def split_into_sentences(text: str) -> list[str]:
    """Split text into a list of sentences cleanly. (Legacy)"""
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
    # Clean and filter out very short results
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def strip_code_fences(text: str) -> str:
    """Remove common markdown code fences around JSON/Cypher outputs."""
    s = text.strip()
    if s.startswith("```"):
        # ```json\n...\n``` or ```\n...\n```
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        if s.endswith("```"):
            s = s[:-3].strip()
    return s.strip()


def extract_first_json(text: str) -> str:
    """
    Extract the first JSON array/object from a string.
    Returns the raw JSON substring (not parsed).
    """
    if not text:
        raise ValueError("Empty LLM response")

    s = strip_code_fences(text)

    # Prefer arrays, then objects.
    array_start = s.find("[")
    object_start = s.find("{")

    if array_start != -1 and (object_start == -1 or array_start < object_start):
        start = array_start
        end = s.rfind("]")
    else:
        start = object_start
        end = s.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON in LLM response")
    return s[start : end + 1]


def recover_json_objects(text: str) -> list[dict]:
    """
    Recover complete JSON objects from a possibly truncated JSON array.
    This is a safety net for LLM responses that stop mid-array.
    """
    if not text:
        return []

    s = strip_code_fences(text)
    objects: list[dict] = []

    for match in re.finditer(r"\{[^{}]*\}", s, flags=re.DOTALL):
        try:
            item = json.loads(match.group(0))
        except Exception:
            continue
        if isinstance(item, dict):
            objects.append(item)

    return objects


def normalise_relation_for_llm(relation: str) -> str:
    """Normalize relation to UPPERCASE_WITH_UNDERSCORES without Neo4j-specific constraints."""
    rel = str(relation or "").strip().upper()
    rel = rel.replace(" ", "_").replace("-", "_")
    rel = re.sub(r"[^A-Z0-9_]", "", rel)
    rel = re.sub(r"_+", "_", rel).strip("_")
    return rel


def normalise_relation_for_neo4j(rel_type: str) -> str:
    """
    Neo4j relationship type must be a valid token.
    This function guarantees a safe uppercase underscore format and prefixes if needed.
    """
    rel = normalise_relation_for_llm(rel_type)
    if not rel:
        return "RELATED"
    # Relationship type cannot start with a digit.
    if re.match(r"^[0-9]", rel):
        rel = f"R_{rel}"
    # Keep it reasonably short.
    return rel[:60]


def load_env() -> None:
    """Backwards-compatible helper for older modules."""
    from dotenv import load_dotenv
    load_dotenv()


def normalise_cypher_response(text: str) -> str:
    """Strip markdown fences and keep the raw Cypher text."""
    return strip_code_fences(text)

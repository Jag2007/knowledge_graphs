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

def chunk_text(text: str, target_words: int = 300, min_words: int = 200, max_words: int = 400) -> list[str]:
    """Split text into chunks (200–400 words each when possible)."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    i = 0
    while i < len(words):
        # Create a chunk near the target, but never exceed max_words.
        j = min(i + target_words, len(words))
        if j - i > max_words:
            j = i + max_words
        chunk_words = words[i:j]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        i = j

    # If the last chunk is too small, merge it into the previous one.
    if len(chunks) >= 2:
        last_words = len(chunks[-1].split())
        if last_words < min_words:
            prev_words = len(chunks[-2].split())
            # Only merge if we stay within the maximum chunk size.
            if prev_words + last_words <= max_words:
                chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
                chunks.pop()

    # Final safeguard: ensure no chunk becomes empty.
    return [c for c in chunks if c.strip()]

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

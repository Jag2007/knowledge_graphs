import hashlib
import json
import os
import re
import threading
import time

from groq import Groq
from dotenv import load_dotenv

from utils import extract_first_json, normalise_relation_for_llm, recover_json_objects

load_dotenv()

NOISY_RELATIONS = {
    "IS",
    "HAS",
    "ARE",
    "ARE_COMMON",
    "OLDEST",
    "MOST_DIVERSE",
    "VARY_BY",
    "VARY_WIDELY",
    "SPREADS",
    "EXPRESSES",
    "EXPRESS",
    "BASED_ON",
    "CONTINUES_TO_EVOLVE",
    "IMPORTANT",
    "POPULAR",
    "FAMOUS",
    "KNOWN_FOR",
}

INVALID_OBJECTS = {
    "world",
    "company",
    "organization",
    "person",
    "culture",
    "society",
}

INVALID_SUBJECTS = {
    "it",
    "they",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "we",
    "i",
    "you",
}

_EXTRACTION_CACHE: dict[str, list[dict]] = {}
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
DEBUG_LLM_OUTPUT = os.environ.get("KG_DEBUG_LLM_OUTPUT", "0").strip().lower() in {"1", "true", "yes"}


def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")
    return Groq(api_key=api_key)


def _extract_retry_delay(error: Exception) -> float:
    message = str(error)
    match = re.search(r"try again in ([0-9.]+)s", message, re.IGNORECASE)
    if match:
        try:
            return max(1.0, float(match.group(1)) + 1.0)
        except ValueError:
            pass
    return float(os.environ.get("GROQ_RETRY_DELAY_SECONDS", "12"))


def _call_groq_with_retry(client: Groq, *, model: str, prompt: str):
    global _LAST_REQUEST_AT

    max_retries = max(1, int(os.environ.get("GROQ_MAX_RETRIES", "4")))
    min_interval = max(0.0, float(os.environ.get("GROQ_MIN_INTERVAL_SECONDS", "1.5")))
    max_tokens = max(256, int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "900")))

    last_error = None
    for attempt in range(max_retries):
        with _REQUEST_LOCK:
            now = time.monotonic()
            wait_time = min_interval - (now - _LAST_REQUEST_AT)
            if wait_time > 0:
                time.sleep(wait_time)

            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                _LAST_REQUEST_AT = time.monotonic()
                return completion
            except Exception as error:
                _LAST_REQUEST_AT = time.monotonic()
                last_error = error
                delay = _extract_retry_delay(error)
                print(
                    f"Groq request failed (attempt {attempt + 1}/{max_retries}). "
                    f"Waiting {delay:.1f}s before retrying."
                )

        if attempt < max_retries - 1:
            time.sleep(delay)

    raise last_error if last_error else RuntimeError("Groq extraction failed.")

def clean_and_validate_triples(triples: list) -> list:
    """Validate and clean the LLM-extracted triples."""
    valid_triples = []

    seen = set()
    
    if not isinstance(triples, list):
        return []
        
    for t in triples:
        if not isinstance(t, dict):
            continue
            
        # Normalize minimal fields as requested.
        subj = str(t.get("subject", "")).strip()
        rel = str(t.get("relation", "")).strip()
        obj = str(t.get("object", "")).strip()

        # Only accept triples when required fields exist and are non-empty.
        if not subj or not rel or not obj:
            continue

        if subj.strip().lower() in INVALID_SUBJECTS:
            continue
            
        # Reject subject == object
        if subj.lower() == obj.lower():
            continue

        # Normalize: strip spaces + uppercase + underscore format.
        rel_clean = normalise_relation_for_llm(rel)
        if not rel_clean:
            continue

        if rel_clean in NOISY_RELATIONS:
            continue

        # Enforce "1–3 words" relations (heuristic: underscore-separated parts).
        parts = [p for p in rel_clean.split("_") if p]
        if len(parts) < 1 or len(parts) > 3:
            continue

        if obj.strip().lower() in INVALID_OBJECTS:
            continue

        if len(obj.split()) > 8:
            continue

        if obj.strip().lower().endswith(("its", "their", "his", "her", "the", "a", "an")):
            continue
            
        key = (subj.strip().lower(), rel_clean.lower(), obj.strip().lower())
        if key in seen:
            continue
        seen.add(key)

        valid_triples.append(
            {
            "subject": subj,
            "relation": rel_clean,
            "object": obj
            }
        )
        
    return valid_triples


def extract_triples_groq(chunk: str) -> list:
    """Extract triples from one chunk with strict JSON parsing and per-chunk caching."""
    text = chunk.strip()
    if not text:
        return []

    cache_key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    if cache_key in _EXTRACTION_CACHE:
        return list(_EXTRACTION_CACHE[cache_key])

    model = os.environ.get("GROQ_TRIPLE_MODEL", "llama-3.1-8b-instant")
    client = _get_client()

    def _attempt(prompt: str) -> tuple[list, str]:
        completion = _call_groq_with_retry(client, model=model, prompt=prompt)

        response = completion.choices[0].message.content or ""
        if DEBUG_LLM_OUTPUT:
            print("RAW LLM OUTPUT:", response[:500])

        # Fallback trigger: empty response.
        if not response.strip():
            return [], "empty"

        try:
            json_blob = extract_first_json(response)
            triples = json.loads(json_blob)
        except Exception:
            triples = recover_json_objects(response)
            if not triples:
                # If parsing still fails -> skip this chunk without crashing the full upload.
                return [], "parse_failed"

        cleaned = clean_and_validate_triples(triples)
        if not cleaned:
            return [], "empty"

        return cleaned, "ok"

    base_prompt = f"""You are an expert knowledge extraction system.

Extract factual knowledge graph triples from the text.

Return ONLY valid JSON.

Format:
[
  {{"subject": "...", "relation": "...", "object": "..."}}
]

Rules:
- Extract multiple factual triples when they are clearly stated.
- Return as many clearly supported triples as possible, up to 20 triples.
- Keep subject and object short and clean.
- Relation must be 1-3 words.
- Use UPPERCASE_WITH_UNDERSCORES.
- Prefer useful factual relations such as LOCATED_IN, WEARS, CELEBRATES, SPEAKS, INCLUDES, PRACTICES, USES, REFLECTS, GUARANTEES, PROTECTS, DRAFTED, MEMBER_OF, ADOPTED, ADOPTED_ON, CONVENED_IN.
- Avoid vague predicate wording like PLAYS, HAS, IS, ARE, IMPORTANT_ROLE, or very long relation names.
- Skip weak descriptive relations like OLDEST, MOST_DIVERSE, IMPORTANT, KNOWN_FOR.
- Skip incomplete objects or trailing phrases.
- Preserve meaningful document facts, but do not invent relations that are not present in the text.
- Do not explain anything.
- If there are no clear factual triples, return [].

Text:
{text}
"""

    fallback_prompt = f"""Extract knowledge graph triples from the text.

Return ONLY valid JSON.

Format:
[
  {{"subject": "...", "relation": "...", "object": "..."}}
]

Text:
{text}
"""

    triples, status = _attempt(base_prompt)
    if status == "empty":
        triples, _ = _attempt(fallback_prompt)

    _EXTRACTION_CACHE[cache_key] = list(triples)
    return list(triples)

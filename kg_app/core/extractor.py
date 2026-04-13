import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

from groq import Groq
from dotenv import load_dotenv

from kg_app.core.utils import (
    extract_first_json,
    normalise_relation_for_llm,
    recover_json_objects,
    split_into_sentences,
)

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

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
    "not specified",
    "not_specified",
    "unknown",
    "none",
    "null",
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
DEBUG_LLM_OUTPUT = os.environ.get("KG_DEBUG_LLM_OUTPUT", "1").strip().lower() in {"1", "true", "yes"}


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
    max_tokens = max(128, int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "400")))

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


def precompute_chunk_metadata(chunks: list[dict]) -> list[dict]:
    """
    Enrich chunk JSON with LLM-generated summaries and keywords in one ingestion-time call.
    If the LLM call fails, the heuristic metadata already on each chunk is kept.
    """
    if not chunks:
        return []

    max_chunks = max(1, int(os.environ.get("KG_METADATA_MAX_CHUNKS", "30")))
    model = os.environ.get("GROQ_METADATA_MODEL", os.environ.get("GROQ_TRIPLE_MODEL", "llama-3.1-8b-instant"))
    payload = [
        {
            "id": chunk.get("id", f"chunk_{index + 1}"),
            "text": str(chunk.get("text", ""))[:900],
        }
        for index, chunk in enumerate(chunks[:max_chunks])
        if str(chunk.get("text", "")).strip()
    ]
    if not payload:
        return chunks

    prompt = f"""
Create retrieval metadata for these PDF chunks.

STRICT RULES:
- Output ONLY valid JSON
- Return a JSON list
- Each item must be:
  {{"id": "chunk_1", "summary": "...", "keywords": ["...", "..."]}}
- Summary must be one short sentence
- Keywords must be 3 to 8 important topic words or phrases
- Do not invent facts

Chunks:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    try:
        client = _get_client()
        completion = _call_groq_with_retry(client, model=model, prompt=prompt)
        content = completion.choices[0].message.content or ""
        metadata = json.loads(extract_first_json(content))
        if not isinstance(metadata, list):
            return chunks

        by_id = {
            str(item.get("id", "")).strip(): item
            for item in metadata
            if isinstance(item, dict) and item.get("id")
        }
        enriched: list[dict] = []
        for chunk in chunks:
            updated = dict(chunk)
            item = by_id.get(str(chunk.get("id", "")).strip())
            if item:
                summary = str(item.get("summary", "")).strip()
                keywords = item.get("keywords", [])
                if summary:
                    updated["summary"] = summary
                if isinstance(keywords, list):
                    cleaned_keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
                    if cleaned_keywords:
                        updated["keywords"] = cleaned_keywords[:8]
            enriched.append(updated)
        return enriched
    except Exception as error:
        print(f"Chunk metadata enrichment skipped: {error}")
        return chunks


def _split_compound_entity(value: str) -> list[str]:
    """Split simple conjunction/list mentions into separate entity values."""
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return []

    if not re.search(r",|\band\b|\bor\b", text, flags=re.IGNORECASE):
        return [text]

    parts = [
        item.strip(" .;:")
        for item in re.split(r"\s*,\s*|\s+\band\b\s+|\s+\bor\b\s+", text, flags=re.IGNORECASE)
        if item.strip(" .;:")
    ]
    return parts or [text]


def _clean_context_phrase(value: str) -> str:
    """Clean a phrase before reusing it as a subject/object in context-linked triples."""
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^(the|a|an|these|those|this|that)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^for\s+(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(is|are|was|were|has|have|had)$", "", text, flags=re.IGNORECASE)
    return text.strip(" .;:")


def _infer_relation_from_phrase(phrase: str) -> str:
    """Map simple natural-language predicates to stable relation labels."""
    text = re.sub(r"\s+", " ", str(phrase or "").strip().lower())
    relation_map = {
        "adopted on": "ADOPTED_ON",
        "was adopted on": "ADOPTED_ON",
        "does not allow for": "DOES_NOT_ALLOW",
        "does not allow": "DOES_NOT_ALLOW",
        "do not allow": "DOES_NOT_ALLOW",
        "helps to protect": "PROTECTS",
        "helps protect": "PROTECTS",
        "adopted": "ADOPTED",
        "drafted": "DRAFTED",
        "includes": "INCLUDES",
        "include": "INCLUDES",
        "contains": "INCLUDES",
        "consists of": "INCLUDES",
        "refers to": "REFERS_TO",
        "guarantees": "GUARANTEES",
        "protects": "PROTECTS",
        "reflects": "REFLECTS",
        "celebrates": "CELEBRATES",
        "practices": "PRACTICES",
        "speaks": "SPEAKS",
        "uses": "USES",
        "located in": "LOCATED_IN",
        "based in": "BASED_IN",
        "convened in": "CONVENED_IN",
        "member of": "MEMBER_OF",
    }
    for key, relation in relation_map.items():
        if key in text:
            return relation
    return ""


def _extract_direct_statement_triples(sentence: str, fallback_subject: str = "") -> list[dict]:
    """Extract direct one-sentence triples and resolve nearby pronoun subjects."""
    text = re.sub(r"\s+", " ", str(sentence or "").strip())
    if not text:
        return []

    patterns = [
        r"^(?P<subject>.+?)\s+(?P<relation>does not allow for|does not allow|do not allow|helps to protect|helps protect|was adopted on|adopted on|convened in|located in|based in|member of|refers to|consists of|includes|include|contains|drafted|adopted|guarantees|protects|reflects|celebrates|practices|speaks|uses)\s+(?P<object>.+?)\.?$",
        r"^(?P<subject>.+?)\s+(?:is|are|was|were)\s+(?P<relation>located in|based in|member of)\s+(?P<object>.+?)\.?$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        subject = _clean_context_phrase(match.group("subject"))
        relation = _infer_relation_from_phrase(match.group("relation"))
        object_text = _clean_context_phrase(match.group("object"))

        if subject.lower() in INVALID_SUBJECTS and fallback_subject:
            subject = fallback_subject

        if not subject or not relation or not object_text:
            return []

        return [
            {
                "subject": subject,
                "relation": relation,
                "object": _clean_context_phrase(object_part),
            }
            for object_part in _split_compound_entity(object_text)
            if _clean_context_phrase(object_part)
        ]

    return []


def _extract_interlinked_context_triples(text: str) -> list[dict]:
    """Recover short-range context links such as 'There are X. These are A, B and C.'"""
    context_subject = ""
    list_subject = ""
    triples: list[dict] = []

    for sentence in split_into_sentences(text):
        normalized = re.sub(r"\s+", " ", sentence).strip()
        if not normalized:
            continue

        intro_match = re.search(
            r"(?:there\s+are|there\s+is)\s+(?P<subject>.+?)\.?$",
            normalized,
            flags=re.IGNORECASE,
        )
        if intro_match:
            list_subject = _clean_context_phrase(intro_match.group("subject"))
            context_subject = list_subject
            continue

        followup_match = re.match(
            r"^(?:these|those|they)(?:\s+are|\s+include)?\s+(?P<object>.+?)\.?$",
            normalized,
            flags=re.IGNORECASE,
        )
        if followup_match and list_subject:
            for object_part in _split_compound_entity(followup_match.group("object")):
                cleaned_object = _clean_context_phrase(object_part)
                if cleaned_object:
                    triples.append(
                        {
                            "subject": list_subject,
                            "relation": "INCLUDES",
                            "object": cleaned_object,
                        }
                    )
            context_subject = list_subject
            continue

        direct_triples = _extract_direct_statement_triples(normalized, context_subject)
        if direct_triples:
            triples.extend(direct_triples)
            context_subject = direct_triples[-1]["subject"]
            list_subject = context_subject
            continue

        if not re.match(r"^(it|they|this|that|these|those)\b", normalized, flags=re.IGNORECASE):
            context_subject = ""
            list_subject = ""

    return triples

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

        if len(obj.split()) > 14:
            continue

        if obj.strip().lower().endswith(("its", "their", "his", "her", "the", "a", "an")):
            continue
            
        for subject_part in _split_compound_entity(subj):
            subject_clean = _clean_context_phrase(subject_part)
            if not subject_clean or subject_clean.lower() in INVALID_SUBJECTS:
                continue

            for object_part in _split_compound_entity(obj):
                object_clean = _clean_context_phrase(object_part)
                if not object_clean or object_clean.lower() in INVALID_OBJECTS:
                    continue

                key = (subject_clean.lower(), rel_clean.lower(), object_clean.lower())
                if key in seen:
                    continue
                seen.add(key)

                valid_triples.append(
                    {
                        "subject": subject_clean,
                        "relation": rel_clean,
                        "object": object_clean,
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
- Return at most 20 triples.
- Keep subject and object short and clean.
- Relation must be 1-3 words.
- Use UPPERCASE_WITH_UNDERSCORES.
- Preserve paragraph context. If a later sentence uses it, they, this, these, or those, resolve it from the nearest previous entity in the same paragraph.
- Preserve list/conjunction structure. If one subject relates to A, B, and C, output one triple per object while repeating the same subject and relation.
- Prefer useful factual relations such as LOCATED_IN, WEARS, CELEBRATES, SPEAKS, INCLUDES, PRACTICES, USES, REFLECTS, GUARANTEES, PROTECTS, DRAFTED, ADOPTED_ON, CONVENED_IN, MEMBER_OF.
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

    context_triples = clean_and_validate_triples(_extract_interlinked_context_triples(text))
    if context_triples:
        merged_triples: list[dict] = []
        seen = set()
        for triple in list(triples) + context_triples:
            key = (
                triple["subject"].strip().lower(),
                triple["relation"].strip().lower(),
                triple["object"].strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged_triples.append(triple)
        triples = merged_triples

    if DEBUG_LLM_OUTPUT:
        print("CLEANED TRIPLES JSON:", json.dumps(triples, ensure_ascii=False))

    _EXTRACTION_CACHE[cache_key] = list(triples)
    return list(triples)

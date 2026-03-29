import json
import os
from groq import Groq
from dotenv import load_dotenv
from utils import (
    extract_first_json,
    normalise_relation_for_llm,
    split_into_sentences,
)

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

NOISY_RELATIONS = {
    "OLDEST",
    "MOST_DIVERSE",
    "REFLECTS",
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


def _group_sentences_for_extraction(text: str, group_size: int = 2, max_words: int = 120) -> list[str]:
    """Split a chunk into smaller sentence groups for more accurate extraction."""
    sentences = split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    groups = []
    current = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and (len(current) >= group_size or current_words + sentence_words > max_words):
            groups.append(" ".join(current).strip())
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words

    if current:
        groups.append(" ".join(current).strip())

    return [group for group in groups if group]

def extract_triples_groq(chunk: str) -> list:
    """Extract (subject, relation, object) triples using smaller text windows."""
    if not chunk.strip():
        return []

    model = os.environ.get("GROQ_TRIPLE_MODEL", "llama-3.1-8b-instant")

    def _attempt(prompt: str) -> tuple[list, str]:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )

        response = completion.choices[0].message.content or ""
        # Print raw output BEFORE parsing (debug).
        print("RAW LLM OUTPUT:", response)

        # Fallback trigger: empty response.
        if not response.strip():
            return [], "empty"

        try:
            json_blob = extract_first_json(response)
            triples = json.loads(json_blob)
        except Exception:
            # If parsing fails -> skip this chunk (do not crash).
            return [], "parse_failed"

        cleaned = clean_and_validate_triples(triples)
        if not cleaned:
            return [], "empty"

        return cleaned, "ok"

    all_triples = []
    seen = set()
    windows = _group_sentences_for_extraction(chunk)

    for window in windows:
        base_prompt = f"""You are an expert knowledge extraction system.

Extract factual knowledge graph triples from the text.

Return ONLY valid JSON.

Format:
[
  {{"subject": "...", "relation": "...", "object": "..."}}
]

Rules:
- Extract multiple factual triples when they are clearly stated.
- Keep subject and object short and clean.
- Relation must be 1-3 words.
- Use UPPERCASE_WITH_UNDERSCORES.
- Prefer useful factual relations such as LOCATED_IN, WEARS, CELEBRATES, SPEAKS, INCLUDES, PRACTICES, USES, PLAYS, PERFORMS.
- Skip vague or descriptive relations like OLDEST, MOST_DIVERSE, IMPORTANT, KNOWN_FOR.
- Skip incomplete objects or trailing phrases.
- Do not explain anything.
- If there are no clear factual triples, return [].

Text:
{window}
"""

        fallback_prompt = f"Extract simple subject-relation-object triples from the text. Return JSON only.\n\nText:\n{window}"

        triples, status = _attempt(base_prompt)

        if status == "empty":
            triples, _ = _attempt(fallback_prompt)

        for triple in triples:
            key = (
                triple["subject"].strip().lower(),
                triple["relation"].strip().lower(),
                triple["object"].strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            all_triples.append(triple)

    return all_triples

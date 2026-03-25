import json
import os
from groq import Groq
from dotenv import load_dotenv
from utils import (
    extract_first_json,
    normalise_relation_for_llm,
)

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

        # Enforce "1–3 words" relations (heuristic: underscore-separated parts).
        parts = [p for p in rel_clean.split("_") if p]
        if len(parts) < 1 or len(parts) > 3:
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
    """Extract (subject, relation, object) triples using Groq LLM and a strict structured prompt."""
    if not chunk.strip():
        return []

    model = os.environ.get("GROQ_TRIPLE_MODEL", "llama-3.1-8b-instant")

    # USE THIS EXACT PROMPT (as provided by the user).
    base_prompt = f"""You are an expert knowledge extraction system.

Extract factual relationships from the text.

Return ONLY valid JSON.

Format:
[
{{"subject": "...", "relation": "...", "object": "..."}}
]

Rules:

Extract ONLY clear factual relationships
Keep subject and object short and clean
Relation must be 1–3 words
Use UPPERCASE_WITH_UNDERSCORES (e.g., CEO_OF, FOUNDED, BASED_IN)
DO NOT return explanations
DO NOT return text outside JSON
If no relationships exist, return []

Text:
{chunk}
"""

    fallback_prompt = f"Extract simple subject-relation-object triples from the text. Return JSON only.\n\nText:\n{chunk}"

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

    triples, status = _attempt(base_prompt)

    # Fallback (critical): if LLM returns empty, retry ONCE with simplified prompt.
    if status == "empty":
        triples, _ = _attempt(fallback_prompt)

    # If parsing fails -> skip chunk (return []).
    return triples

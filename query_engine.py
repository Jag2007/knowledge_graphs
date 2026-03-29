import re

from graph import Neo4jGraph

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
}


def _relation_to_text(relation: str) -> str:
    return relation.replace("_", " ").strip().lower()


def _clean_entity_text(value: str) -> str:
    text = str(value or "").strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    if text.isupper():
        return text.title()
    return text


def _sentence_from_relation(subject: str, relation: str, object_: str, question: str, terms: list[str]) -> str:
    relation_upper = relation.replace(" ", "_").upper()
    term_hits_subject = sum(1 for term in terms if term.lower() in subject.lower())
    term_hits_object = sum(1 for term in terms if term.lower() in object_.lower())

    if relation_upper.startswith("HAS_"):
        role = relation_upper[4:].replace("_", " ").lower()
        return f"{object_} is the {role} of {subject}."

    if relation_upper.endswith("_OF"):
        role = relation_upper[:-3].replace("_", " ").lower()
        return f"{subject} is the {role} of {object_}."

    if question.strip().lower().startswith("who is") and term_hits_object > term_hits_subject:
        return f"{object_} is related to {subject} through {relation}."

    if question.strip().lower().startswith("where"):
        return f"{subject} is {relation} {object_}."

    return f"{subject} {relation} {object_}."


def _extract_terms(question: str) -> list[str]:
    """Extract useful search terms without hardcoding any domain entities."""
    if not question.strip():
        return []

    terms: list[str] = []
    seen = set()

    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
    for left, right in quoted:
        value = (left or right).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            terms.append(value)

    title_case = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)
    for value in title_case:
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in STOP_WORDS and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            terms.append(cleaned)

    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question):
        lowered = token.lower()
        if lowered in STOP_WORDS or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(token)

    return terms[:12]


def _score_direct_row(row: dict, terms: list[str]) -> int:
    subject = _clean_entity_text(row.get("from_name", "")).lower()
    object_ = _clean_entity_text(row.get("to_name", "")).lower()
    relation = _relation_to_text(str(row.get("relation_type", "")).strip())
    score = 0

    for term in terms:
        token = term.lower()
        if token in subject:
            score += 3
        if token in object_:
            score += 3
        if token in relation:
            score += 1

    if subject and object_:
        score += 1
    return score


def _format_direct_results(question: str, results: list[dict], terms: list[str]) -> str:
    cleaned_rows = []
    for row in results:
        subject = _clean_entity_text(row.get("from_name", ""))
        relation = _relation_to_text(str(row.get("relation_type", "")).strip())
        object_ = _clean_entity_text(row.get("to_name", ""))
        if not subject or not relation or not object_:
            continue
        cleaned_rows.append(
            {
                "subject": subject,
                "relation": relation,
                "object": object_,
                "score": _score_direct_row(row, terms),
            }
        )

    if not cleaned_rows:
        return ""

    best = sorted(
        cleaned_rows,
        key=lambda row: (row["score"], len(row["subject"]) + len(row["object"])),
        reverse=True,
    )[0]

    lower_question = question.strip().lower()
    if lower_question.startswith("who is"):
        return _sentence_from_relation(best["subject"], best["relation"], best["object"], question, terms)
    if lower_question.startswith("what is"):
        return f"{best['subject']} is related to {best['object']} through {best['relation']}."
    if lower_question.startswith("where"):
        return f"{best['subject']} is {best['relation']} {best['object']}."
    return _sentence_from_relation(best["subject"], best["relation"], best["object"], question, terms)


def _format_path_results(results: list[dict]) -> str:
    lines: list[str] = []
    seen = set()

    for row in results:
        nodes = [_clean_entity_text(name) for name in row.get("path_nodes", []) if _clean_entity_text(name)]
        relations = [str(rel).strip() for rel in row.get("path_relationships", []) if str(rel).strip()]
        if len(nodes) < 2 or not relations:
            continue

        parts = [nodes[0]]
        for idx, relation in enumerate(relations):
            if idx + 1 >= len(nodes):
                break
            parts.append(_relation_to_text(relation))
            parts.append(nodes[idx + 1])

        line = " ".join(parts)
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if len(lines) >= 6:
            break

    if not lines:
        return ""
    return "Based on the uploaded document: " + "; ".join(lines) + "."


def ask_question(question: str) -> dict:
    """Fast graph-first question answering with one-hop and two-hop fallback."""
    graph = Neo4jGraph()
    try:
        total_triples = graph.count_triples()
        terms = _extract_terms(question)

        direct_cypher, direct_results = graph.search_related(terms, limit=25)
        if direct_results:
            return {
                "triples_added": total_triples,
                "query": direct_cypher,
                "results": direct_results,
                "answer": _format_direct_results(question, direct_results, terms),
                "steps_taken": [
                    {
                        "step": "one_hop_lookup",
                        "terms": terms,
                        "matches": len(direct_results),
                    }
                ],
            }

        path_cypher, path_results = graph.search_paths(terms, max_hops=2, limit=20)
        if path_results:
            return {
                "triples_added": total_triples,
                "query": path_cypher,
                "results": path_results,
                "answer": _format_path_results(path_results),
                "steps_taken": [
                    {
                        "step": "two_hop_lookup",
                        "terms": terms,
                        "matches": len(path_results),
                    }
                ],
            }

        return {
            "triples_added": total_triples,
            "query": path_cypher if terms else direct_cypher,
            "results": [],
            "answer": "No data found in knowledge graph",
            "steps_taken": [
                {
                    "step": "no_match",
                    "terms": terms,
                    "matches": 0,
                }
            ],
        }
    finally:
        graph.close()

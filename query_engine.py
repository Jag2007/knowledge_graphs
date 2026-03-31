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
    "pdf",
    "document",
    "file",
}

QUESTION_SYNONYMS = {
    "clothing": ["wear", "wears", "attire", "dress", "garment"],
    "food": ["cuisine", "dish", "dishes", "meal"],
    "festival": ["festivals", "celebrates", "celebration"],
    "religion": ["spirituality", "belief", "beliefs"],
    "culture": ["tradition", "traditions", "heritage"],
}

GENERIC_NODE_PATTERNS = [
    "what do you know about",
    "tell me about",
    "what are",
    "who is",
]

RELATION_HINTS = {
    "include": {"INCLUDES"},
    "includes": {"INCLUDES"},
    "capital": {"HAS_CAPITAL", "CAPITAL", "CAPITAL_OF"},
    "located": {"LOCATED_IN", "BASED_IN"},
    "based": {"LOCATED_IN", "BASED_IN"},
    "celebrate": {"CELEBRATES"},
    "celebrates": {"CELEBRATES"},
    "wear": {"WEARS"},
    "wears": {"WEARS"},
    "practice": {"PRACTICES"},
    "practices": {"PRACTICES"},
    "found": {"FOUNDED", "FOUNDED_BY"},
    "founded": {"FOUNDED", "FOUNDED_BY"},
}

SUMMARY_RELATION_ORDER = [
    "REFLECTS",
    "CELEBRATES",
    "INCLUDES",
    "PRACTICES",
    "WEARS",
    "LOCATED_IN",
    "BASED_IN",
]


def _relation_to_text(relation: str) -> str:
    return relation.replace("_", " ").strip().lower()


def _verb_from_relation(relation: str) -> str:
    relation_upper = relation.replace(" ", "_").upper()
    mapping = {
        "HAS_CAPITAL": "has capital",
        "CAPITAL_OF": "is the capital of",
        "LOCATED_IN": "is located in",
        "BASED_IN": "is based in",
        "INCLUDES": "includes",
        "WEARS": "wears",
        "PRACTICES": "practices",
        "CELEBRATES": "celebrates",
        "FOUNDED": "founded",
        "FOUNDED_BY": "was founded by",
        "CHAIRMAN_OF": "is the chairman of",
        "CEO_OF": "is the CEO of",
        "REFLECTS": "reflects",
        "REPRESENTS": "represents",
        "KNOWN_FOR": "is known for",
    }
    if relation_upper in mapping:
        return mapping[relation_upper]
    return _relation_to_text(relation)


def _adjust_verb_for_subject(subject: str, verb: str) -> str:
    lowered_subject = subject.strip().lower()
    if lowered_subject.endswith("s") and verb.endswith("s"):
        if verb.endswith("ies"):
            return verb[:-3] + "y"
        if verb.endswith("es"):
            return verb[:-2]
        return verb[:-1]
    return verb


def _clean_entity_text(value: str) -> str:
    text = str(value or "").strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    if text.isupper():
        return text.title()
    return text


def _normalize_token(value: str) -> str:
    token = str(value or "").strip().lower()
    if token.endswith("s") and len(token) > 4:
        token = token[:-1]
    return token


def _sentence_from_relation(subject: str, relation: str, object_: str, question: str, terms: list[str]) -> str:
    relation_upper = relation.replace(" ", "_").upper()
    term_hits_subject = sum(1 for term in terms if term.lower() in subject.lower())
    term_hits_object = sum(1 for term in terms if term.lower() in object_.lower())

    if relation_upper in {"CAPITAL", "HAS_CAPITAL"}:
        return f"The capital of {subject} is {object_}."

    if relation_upper == "CAPITAL_OF":
        return f"The capital of {object_} is {subject}."

    if relation_upper.startswith("HAS_"):
        role = relation_upper[4:].replace("_", " ").lower()
        return f"{object_} is the {role} of {subject}."

    if relation_upper.endswith("_OF"):
        role = relation_upper[:-3].replace("_", " ").lower()
        return f"{subject} is the {role} of {object_}."

    if relation_upper in {"LOCATED_IN", "BASED_IN"}:
        return f"{subject} is located in {object_}."

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


def _extract_entity_phrases(question: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9\s_-]", " ", question)
    tokens = [token for token in cleaned.split() if token.strip()]
    relation_words = {word.lower() for word in RELATION_HINTS.keys()}
    meaningful = [
        token for token in tokens
        if token.lower() not in STOP_WORDS and token.lower() not in relation_words
    ]
    phrases: list[str] = []

    if len(meaningful) >= 2:
        phrases.append(" ".join(meaningful))

    title_case = re.findall(r"\b(?:[A-Z][A-Za-z0-9_-]*)(?:\s+[A-Z][A-Za-z0-9_-]*)+\b", question)
    for value in title_case:
        phrase = value.strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    return phrases[:4]


def _expand_terms(terms: list[str]) -> list[str]:
    expanded: list[str] = []
    seen = set()
    for term in terms:
        base = term.strip().lower()
        if not base:
            continue
        candidates = [base]
        if base.endswith("s") and len(base) > 4:
            candidates.append(base[:-1])
        if base.endswith("ing") and len(base) > 5:
            candidates.append(base[:-3])
        candidates.extend(QUESTION_SYNONYMS.get(base, []))

        for candidate in candidates:
            cleaned = candidate.strip().lower()
            if not cleaned or cleaned in seen or cleaned in STOP_WORDS:
                continue
            seen.add(cleaned)
            expanded.append(cleaned)
    return expanded[:20]


def _extract_relation_hints(question: str) -> set[str]:
    hints: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", question.lower()):
        hints.update(RELATION_HINTS.get(token, set()))
    return hints


def _is_overview_question(question: str) -> bool:
    text = question.strip().lower()
    patterns = [
        "what does this pdf talk about",
        "what does this document talk about",
        "what is this pdf about",
        "what is this document about",
        "give me a summary",
        "summarize",
        "summary",
        "main topics",
        "key topics",
        "overview",
    ]
    return any(pattern in text for pattern in patterns)


def _is_node_overview_question(question: str) -> bool:
    text = question.strip().lower()
    return any(pattern in text for pattern in GENERIC_NODE_PATTERNS)


def _is_plain_entity_query(question: str) -> bool:
    tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", question.lower()) if token not in STOP_WORDS]
    return bool(tokens) and len(tokens) <= 4 and "?" not in question


def _score_direct_row(row: dict, terms: list[str]) -> int:
    subject = _clean_entity_text(row.get("from_name", "")).lower()
    object_ = _clean_entity_text(row.get("to_name", "")).lower()
    relation = _relation_to_text(str(row.get("relation_type", "")).strip())
    searchable_text = str(row.get("searchable_text", "")).lower()
    score = 0

    for term in terms:
        token = term.lower()
        if token in subject:
            score += 3
        if token in object_:
            score += 3
        if token in relation:
            score += 2
        if token in searchable_text:
            score += 1

    if subject and object_:
        score += 1
    return score


def _format_direct_results(question: str, results: list[dict], terms: list[str], relation_hints: set[str]) -> str:
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

    if relation_hints:
        grouped: dict[tuple[str, str], list[str]] = {}
        for row in sorted(cleaned_rows, key=lambda item: item["score"], reverse=True):
            relation_upper = row["relation"].replace(" ", "_").upper()
            if relation_upper not in relation_hints:
                continue
            key = (row["subject"], relation_upper)
            grouped.setdefault(key, [])
            if row["object"] not in grouped[key]:
                grouped[key].append(row["object"])

        if grouped:
            best_key = max(grouped.keys(), key=lambda key: len(grouped[key]))
            subject, relation_upper = best_key
            values = grouped[best_key]
            if relation_upper in {"HAS_CAPITAL", "CAPITAL"} and values:
                return f"The capital of {subject} is {values[0]}."
            if relation_upper == "CAPITAL_OF" and values:
                return f"The capital of {values[0]} is {subject}."
            if relation_upper == "CELEBRATES":
                return f"{subject} celebrates {_join_values(values)}."
            if relation_upper == "INCLUDES":
                return f"{subject} includes {_join_values(values)}."
            if relation_upper in {"LOCATED_IN", "BASED_IN"} and values:
                return f"{subject} is located in {values[0]}."
            verb = _adjust_verb_for_subject(subject, _verb_from_relation(relation_upper))
            return f"{subject} {verb} {_join_values(values)}."

    best = sorted(
        cleaned_rows,
        key=lambda row: (row["score"], len(row["subject"]) + len(row["object"])),
        reverse=True,
    )[0]

    lower_question = question.strip().lower()
    if lower_question.startswith("who is"):
        return _sentence_from_relation(best["subject"], best["relation"], best["object"], question, terms)
    if lower_question.startswith("what is"):
        return _sentence_from_relation(best["subject"], best["relation"], best["object"], question, terms)
    if lower_question.startswith("where"):
        return _sentence_from_relation(best["subject"], best["relation"], best["object"], question, terms)
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


def _format_overview(results: list[dict]) -> str:
    topics: list[str] = []
    seen = set()

    for row in results:
        subject = _clean_entity_text(row.get("from_name", ""))
        object_ = _clean_entity_text(row.get("to_name", ""))
        for item in (subject, object_):
            key = item.lower()
            if item and key not in seen:
                seen.add(key)
                topics.append(item)
            if len(topics) >= 6:
                break
        if len(topics) >= 6:
            break

    if not topics:
        return ""
    if len(topics) == 1:
        return f"The uploaded document mainly discusses {topics[0]}."
    return "The uploaded document mainly discusses " + ", ".join(topics[:-1]) + f", and {topics[-1]}."


def _join_values(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _rank_anchor_entities(terms: list[str], phrases: list[str], candidates: list[dict]) -> list[str]:
    scored: list[tuple[int, str]] = []
    for candidate in candidates:
        name = _clean_entity_text(candidate.get("entity_name", ""))
        lowered = name.lower()
        normalized_name = _normalize_token(name)
        score = 0
        for phrase in phrases:
            phrase_lower = phrase.lower()
            normalized_phrase = _normalize_token(phrase)
            if phrase_lower == lowered:
                score += 12
            elif normalized_phrase == normalized_name:
                score += 10
            elif phrase_lower in lowered:
                score += 6
        for term in terms:
            token = term.lower()
            if token == lowered:
                score += 6
            elif _normalize_token(token) == normalized_name:
                score += 5
            elif token in lowered:
                score += 3
        score += len(name.split())
        scored.append((score, name))

    ranked = sorted(scored, key=lambda item: (item[0], -len(item[1])), reverse=True)
    ordered_names: list[str] = []
    seen = set()
    for _, name in ranked:
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered_names.append(name)
    return ordered_names


def _score_neighborhood_row(row: dict, anchor: str, terms: list[str], relation_hints: set[str]) -> int:
    subject = _clean_entity_text(row.get("from_name", ""))
    object_ = _clean_entity_text(row.get("to_name", ""))
    relation = str(row.get("relation_type", "")).strip().upper()
    score = 0

    if subject.lower() == anchor.lower() or object_.lower() == anchor.lower():
        score += 5
    if relation in relation_hints:
        score += 8

    subject_lower = subject.lower()
    object_lower = object_.lower()
    relation_text = _relation_to_text(relation).lower()
    for term in terms:
        token = term.lower()
        if token in subject_lower:
            score += 2
        if token in object_lower:
            score += 2
        if token in relation_text:
            score += 2
    return score


def _format_entity_neighborhood(question: str, anchor: str, rows: list[dict], terms: list[str], relation_hints: set[str]) -> str:
    facts: list[str] = []
    seen = set()
    lowered_anchor = anchor.lower()
    grouped_objects: dict[str, list[str]] = {}

    ranked_rows = sorted(
        rows,
        key=lambda row: _score_neighborhood_row(row, anchor, terms, relation_hints),
        reverse=True,
    )

    for row in ranked_rows:
        subject = _clean_entity_text(row.get("from_name", ""))
        relation = str(row.get("relation_type", "")).strip()
        object_ = _clean_entity_text(row.get("to_name", ""))
        relation_upper = relation.replace(" ", "_").upper()
        if not subject or not relation or not object_:
            continue

        if subject.lower() == lowered_anchor:
            grouped_objects.setdefault(relation_upper, [])
            if object_ not in grouped_objects[relation_upper]:
                grouped_objects[relation_upper].append(object_)

    lower_question = question.strip().lower()
    if relation_hints & {"HAS_CAPITAL", "CAPITAL", "CAPITAL_OF"}:
        if "HAS_CAPITAL" in grouped_objects:
            value = grouped_objects["HAS_CAPITAL"][0]
            return f"The capital of {anchor} is {value}."
        if "CAPITAL" in grouped_objects:
            value = grouped_objects["CAPITAL"][0]
            return f"The capital of {anchor} is {value}."
        for row in ranked_rows:
            relation_upper = str(row.get("relation_type", "")).strip().replace(" ", "_").upper()
            subject = _clean_entity_text(row.get("from_name", ""))
            object_ = _clean_entity_text(row.get("to_name", ""))
            if relation_upper == "CAPITAL_OF" and object_.lower() == lowered_anchor:
                return f"The capital of {anchor} is {subject}."

    if relation_hints:
        if "CELEBRATES" in relation_hints and "CELEBRATES" in grouped_objects:
            values = grouped_objects["CELEBRATES"]
            return f"{anchor} celebrates {_join_values(values)}."
        if "INCLUDES" in relation_hints and "INCLUDES" in grouped_objects:
            values = grouped_objects["INCLUDES"]
            return f"{anchor} includes {_join_values(values)}."
        if relation_hints & {"LOCATED_IN", "BASED_IN"}:
            for relation_key in ("LOCATED_IN", "BASED_IN"):
                if relation_key in grouped_objects and grouped_objects[relation_key]:
                    return f"{anchor} is located in {grouped_objects[relation_key][0]}."
        if relation_hints & {"PRACTICES"} and "PRACTICES" in grouped_objects:
            return f"{anchor} practices {_join_values(grouped_objects['PRACTICES'])}."
        if relation_hints & {"WEARS"} and "WEARS" in grouped_objects:
            return f"{anchor} wears {_join_values(grouped_objects['WEARS'])}."
        if relation_hints & {"FOUNDED", "FOUNDED_BY"}:
            if "FOUNDED" in grouped_objects:
                return f"{anchor} founded {_join_values(grouped_objects['FOUNDED'])}."
            if "FOUNDED_BY" in grouped_objects:
                return f"{anchor} was founded by {_join_values(grouped_objects['FOUNDED_BY'])}."
        return ""

    if lower_question.startswith("what does") and "INCLUDES" in grouped_objects:
        values = grouped_objects["INCLUDES"]
        if len(values) == 1:
            return f"{anchor} includes {values[0]}."
        return f"{anchor} includes {_join_values(values)}."

    if not relation_hints and (
        lower_question.startswith("what is")
        or lower_question.startswith("what are")
        or _is_node_overview_question(question)
        or _is_plain_entity_query(question)
    ):
        grouped_by_verb: dict[str, list[str]] = {}
        ordered_relations = [rel for rel in SUMMARY_RELATION_ORDER if rel in grouped_objects]
        ordered_relations.extend(rel for rel in grouped_objects.keys() if rel not in ordered_relations)
        for relation_upper in ordered_relations:
            values = grouped_objects.get(relation_upper, [])
            if not values:
                continue
            verb = _adjust_verb_for_subject(anchor, _verb_from_relation(relation_upper))
            grouped_by_verb.setdefault(verb, [])
            for value in values:
                if value not in grouped_by_verb[verb]:
                    grouped_by_verb[verb].append(value)
        summary_lines = [f"{anchor} {verb} {_join_values(values)}" for verb, values in grouped_by_verb.items() if values]
        if summary_lines:
            if len(summary_lines) == 1:
                return summary_lines[0] + "."
            return "Based on the uploaded document: " + "; ".join(summary_lines) + "."

    for row in ranked_rows:
        subject = _clean_entity_text(row.get("from_name", ""))
        relation = str(row.get("relation_type", "")).strip()
        object_ = _clean_entity_text(row.get("to_name", ""))
        if not subject or not relation or not object_:
            continue

        if subject.lower() == lowered_anchor:
            verb = _adjust_verb_for_subject(anchor, _verb_from_relation(relation))
            line = f"{anchor} {verb} {object_}"
        elif object_.lower() == lowered_anchor:
            relation_upper = relation.replace(" ", "_").upper()
            if relation_upper.startswith("HAS_"):
                role = relation_upper[4:].replace("_", " ").lower()
                line = f"{anchor} is the {role} of {subject}"
            elif relation_upper.endswith("_OF"):
                role = relation_upper[:-3].replace("_", " ").lower()
                line = f"{subject} is the {role} of {anchor}"
            else:
                line = f"{anchor} is related to {subject} through {_verb_from_relation(relation)}"
        else:
            continue

        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(line)
        if len(facts) >= 4:
            break

    if not facts:
        return ""

    if question.strip().lower().startswith("what are"):
        return facts[0] + "."
    if question.strip().lower().startswith("who is") or question.strip().lower().startswith("what is"):
        return facts[0] + "."
    return "Based on the uploaded document: " + "; ".join(facts) + "."


def ask_question(question: str, document_id: str) -> dict:
    """Graph-first question answering scoped to the active uploaded document."""
    graph = Neo4jGraph()
    try:
        total_triples = graph.count_triples(document_id)
        terms = _extract_terms(question)
        entity_phrases = _extract_entity_phrases(question)
        expanded_terms = _expand_terms(terms)
        relation_hints = _extract_relation_hints(question)

        if _is_overview_question(question):
            summary = graph.get_document_summary(document_id)
            if summary:
                return {
                    "triples_added": total_triples,
                    "query": "DOCUMENT_SUMMARY_LOOKUP",
                    "results": [{"summary": summary}],
                    "answer": summary,
                    "steps_taken": [
                        {
                            "step": "document_summary",
                            "terms": expanded_terms,
                            "matches": 1,
                        }
                    ],
                }

            overview_cypher, overview_results = graph.get_graph_overview(document_id, limit=12)
            if overview_results:
                return {
                    "triples_added": total_triples,
                    "query": overview_cypher,
                    "results": overview_results,
                    "answer": _format_overview(overview_results),
                    "steps_taken": [
                        {
                            "step": "graph_overview",
                            "terms": expanded_terms,
                            "matches": len(overview_results),
                        }
                    ],
                }

        entity_search_terms = entity_phrases + (terms or expanded_terms)
        entity_cypher, entity_candidates = graph.find_relevant_entities(entity_search_terms, document_id, limit=12)
        anchor_entities = _rank_anchor_entities(terms or expanded_terms, entity_phrases, entity_candidates)
        for anchor_entity in anchor_entities[:5]:
            neighborhood_cypher, neighborhood_rows = graph.get_entity_neighborhood(anchor_entity, document_id, limit=20)
            if not neighborhood_rows:
                continue
            neighborhood_answer = _format_entity_neighborhood(
                question,
                anchor_entity,
                neighborhood_rows,
                expanded_terms,
                relation_hints,
            )
            if not neighborhood_answer:
                continue

            step_name = "entity_neighborhood"
            if _is_node_overview_question(question):
                step_name = "entity_overview"

            return {
                "triples_added": total_triples,
                "query": neighborhood_cypher,
                "results": neighborhood_rows,
                "answer": neighborhood_answer,
                "steps_taken": [
                    {
                        "step": step_name,
                        "anchor": anchor_entity,
                        "terms": expanded_terms,
                        "relation_hints": sorted(relation_hints),
                        "matches": len(neighborhood_rows),
                    }
                ],
            }

        direct_cypher, direct_results = graph.search_related(expanded_terms, document_id, limit=25)
        if direct_results:
            return {
                "triples_added": total_triples,
                "query": direct_cypher,
                "results": direct_results,
                "answer": _format_direct_results(question, direct_results, expanded_terms, relation_hints),
                "steps_taken": [
                    {
                        "step": "one_hop_lookup",
                        "terms": expanded_terms,
                        "matches": len(direct_results),
                    }
                ],
            }

        semantic_cypher, semantic_results = graph.search_semantic(expanded_terms, document_id, limit=40)
        if semantic_results:
            return {
                "triples_added": total_triples,
                "query": semantic_cypher,
                "results": semantic_results,
                "answer": _format_direct_results(question, semantic_results, expanded_terms, relation_hints),
                "steps_taken": [
                    {
                        "step": "semantic_lookup",
                        "terms": expanded_terms,
                        "matches": len(semantic_results),
                    }
                ],
            }

        path_cypher, path_results = graph.search_paths(expanded_terms, document_id, max_hops=2, limit=20)
        if path_results:
            return {
                "triples_added": total_triples,
                "query": path_cypher,
                "results": path_results,
                "answer": _format_path_results(path_results),
                "steps_taken": [
                    {
                        "step": "two_hop_lookup",
                        "terms": expanded_terms,
                        "matches": len(path_results),
                    }
                ],
            }

        return {
            "triples_added": total_triples,
            "query": path_cypher if expanded_terms else direct_cypher,
            "results": [],
            "answer": "No data found in knowledge graph",
            "steps_taken": [
                {
                    "step": "no_match",
                    "terms": expanded_terms,
                    "matches": 0,
                }
            ],
        }
    finally:
        graph.close()

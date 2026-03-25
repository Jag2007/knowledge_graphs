import os
import json
import re
from typing import Any, Dict, List, Tuple
from groq import Groq
from graph import Neo4jGraph
from dotenv import load_dotenv
from utils import extract_first_json, normalise_cypher_response

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def decompose_question(question: str) -> list:
    """Break a complex question into single-hop logical steps dynamically."""
    prompt = f"""You are a reasoning agent.
Break down the complex question into a JSON list of simpler, step-by-step questions to query a knowledge graph.
If the question is simple, return a list with just the question.
ONLY output a valid JSON list of strings. No explanations.

Example Q: "Where is the company founded by the CEO of X based?"
Output:
[
  "Who is the CEO of X?",
  "What company was founded by that person?",
  "Where is that company based?"
]

Now decompose:
Q: "{question}"
Output:"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        res = completion.choices[0].message.content.strip()
        if res.startswith("```json"): res = res[7:-3]
        elif res.startswith("```"): res = res[3:-3]
        steps = json.loads(res.strip())
        if isinstance(steps, list): return steps
    except:
        pass
    return [question]

def generate_cypher_query(question: str, context: str = "") -> str:
    """Convert question to Cypher generics."""
    ctx_prompt = f"\nPrevious findings context to use for reference: {context}\n" if context else ""
    prompt = f"""You are a Neo4j Cypher generator.

Convert a natural language question into a Cypher query.

Schema assumptions:

* Nodes: have property `name`
* Relationships: dynamic types created from extracted triples

Rules:

* Use ONLY:
  MATCH (a)-[r]->(b) or MATCH (a)-[r]-(b)

* Use case-insensitive matching:
  toLower(a.name) CONTAINS toLower("entity")

* Return relevant names

* DO NOT assume specific entities from a particular domain

* Work for ANY domain
{ctx_prompt}

---

Examples:

Q: Who founded X?
Cypher:
MATCH (a)-[r]->(b)
WHERE toLower(b.name) CONTAINS "x"
RETURN a.name

---

Q: Where is X based?
Cypher:
MATCH (a)-[r]->(b)
WHERE toLower(a.name) CONTAINS "x"
RETURN b.name

---

Now generate Cypher for:
{question}
"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        query = completion.choices[0].message.content.strip()
        if query.startswith("```cypher"): query = query[9:-3]
        elif query.startswith("```"): query = query[3:-3]
        return query.strip()
    except Exception as e:
        print(f"Error generating cypher: {e}")
        return ""

def generate_final_answer(question: str, results: list) -> str:
    """Synthesize results into a human friendly generic answer."""
    if not results or all(not r for r in results):
        return "No data found in knowledge graph"
    
    prompt = f"""Answer the question concisely based ONLY on the provided graph database results.
If the results don't contain the answer, say "No data found in knowledge graph"

Question: {question}
Results: {json.dumps(results)}
Answer:"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return "Failed to generate answer"

def ask_question(question: str):
    """Execute dynamic multi-hop logic and return formatted results."""
    steps = decompose_question(question)
    
    graph = Neo4jGraph()
    combined_results = []
    context = ""
    last_cypher = ""
    total_triples = 0
    
    try:
        with graph.driver.session() as session:
            # Count overall triples so it's globally available
            count_res = session.run("MATCH ()-[r]->() RETURN count(r) as c")
            total_triples = count_res.single()["c"] if count_res else 0

            for step in steps:
                cypher_query = generate_cypher_query(step, context)
                if not cypher_query:
                    continue
                last_cypher = cypher_query
                
                try:
                    result = session.run(cypher_query)
                    records = [record.data() for record in result]
                    if records:
                        combined_results.extend(records)
                        # Add results to context for the next hop
                        context += f"For step '{step}', found: {json.dumps(records)}\n"
                except Exception as e:
                    print(f"Cypher execution error: {e}")
                    
    finally:
        graph.close()

    if not last_cypher:
        return {
            "triples_added": total_triples,
            "query": "",
            "results": [],
            "answer": "Failed to generate Cypher query"
        }

    if not combined_results:
        return {
            "triples_added": total_triples,
            "query": last_cypher,
            "results": [],
            "answer": "No data found in knowledge graph"
        }
        
    final_answer = generate_final_answer(question, combined_results)
    
    return {
        "triples_added": total_triples, # Total system triples
        "query": last_cypher,          # Executed query (last hop or single query)
        "results": combined_results,   # Array of all matched items
        "answer": final_answer,        # Direct concise text answer
        "steps_taken": steps           # Included for debug visibility
    }


# --- Upgraded Generalized Pipeline (overrides ask_question) ---

def _llm_chat(prompt: str, *, model: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


def _safe_parse_json_from_llm(text: str) -> Any:
    """Parse the first JSON array/object from an LLM response."""
    json_blob = extract_first_json(text)
    return json.loads(json_blob)


def _extract_question_entities(question: str) -> list[str]:
    model = os.environ.get("GROQ_ENTITY_MODEL", os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))
    prompt = f"""
You are an information extraction system.

Extract the most relevant entity strings mentioned in the question.
Entities can be names of people, organizations, products, locations, etc.

Rules:
* Output ONLY valid JSON.
* Output ONLY a JSON array of strings.
* Each string must be a span exactly as it appears or closely as it is written in the question.
* Output [] if you cannot find any entities.

Question:
{question}
"""
    try:
        text = _llm_chat(prompt, model=model, max_tokens=256, temperature=0.0)
        parsed = _safe_parse_json_from_llm(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            # De-dup while preserving order
            seen = set()
            out: list[str] = []
            for x in parsed:
                s = x.strip()
                if not s:
                    continue
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(s)
            return out[:12]
    except Exception:
        pass
    return []


def _decompose_question_steps(question: str, max_steps: int = 3) -> list[str]:
    model = os.environ.get("GROQ_DECOMP_MODEL", os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))
    prompt = f"""
You are a reasoning agent.

Decompose the question into up to {max_steps} sequential graph steps.
Each step should describe what the next traversal should achieve (in natural language),
but you must still keep it aligned with step-by-step graph navigation.

Rules:
* Output ONLY valid JSON.
* Output ONLY a JSON object with this shape:
  {{"steps": ["...", "..."]}}
* The steps array length must be between 1 and {max_steps}.

Question:
{question}
"""
    try:
        text = _llm_chat(prompt, model=model, max_tokens=256, temperature=0.0)
        parsed = _safe_parse_json_from_llm(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
            steps = [str(s).strip() for s in parsed["steps"] if str(s).strip()]
            return steps[:max_steps] if steps else [question]
    except Exception:
        pass
    return [question]


def _generate_direct_cypher(question: str) -> str:
    model = os.environ.get("GROQ_CYPHER_MODEL", os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))
    prompt = f"""
You are a Neo4j Cypher generator.
Convert a natural language question into a Cypher query for a graph built from extracted triples.

Schema:
* Nodes: (:Entity {{name}})
* Relationships: dynamic relationship types (but in Cypher you should match with a relationship variable `r`)

Rules:
* Output ONLY Cypher text (no markdown, no code fences).
* Use flexible matching:
  toLower(a.name) CONTAINS toLower("<entity span from the question>")
  and/or toLower(b.name) CONTAINS toLower("<entity span from the question>")
* Do NOT hardcode entity names that are not present in the question.
* Infer relationships dynamically:
  MATCH (a)-[r]->(b)
* Return relevant names and relationship:
  RETURN a.name AS from_name, coalesce(r.relType, type(r)) AS relation_type, b.name AS to_name
* LIMIT 50

Question:
{question}
"""
    text = _llm_chat(prompt, model=model, max_tokens=256, temperature=0.0)
    cypher = normalise_cypher_response(text)
    cypher = cypher.strip()

    # Heuristic cleanup: keep from the first MATCH to the last RETURN.
    match_idx = cypher.upper().find("MATCH")
    return_idx = cypher.upper().rfind("RETURN")
    if match_idx != -1 and return_idx != -1:
        cypher = cypher[match_idx:]
    cypher = cypher.split("```")[0].strip().rstrip(";")

    # Basic safety checks to avoid obviously invalid outputs.
    upper = cypher.upper()
    if "MATCH" not in upper or "RETURN" not in upper:
        return ""
    if "DELETE" in upper or "DETACH" in upper or "CREATE " in upper or "DROP " in upper:
        return ""
    return cypher


def _execute_cypher(graph: Neo4jGraph, cypher: str) -> list[dict]:
    with graph.driver.session() as session:
        result = session.run(cypher)
        return [r.data() for r in result]


def _one_hop_traversal(graph: Neo4jGraph, start_entities: list[str], limit: int = 200) -> Tuple[str, list[dict]]:
    cypher = """
MATCH (a:Entity)
WHERE ANY(n IN $names WHERE toLower(a.name) CONTAINS toLower(n))
MATCH (a)-[r]-(b:Entity)
RETURN a.name AS from_name,
       coalesce(r.relType, type(r)) AS relation_type,
       b.name AS to_name
LIMIT $limit
""".strip()
    if not start_entities:
        return cypher, []
    names = [s for s in start_entities if s.strip()]
    with graph.driver.session() as session:
        result = session.run(cypher, names=names[:12], limit=limit)
        return cypher, [r.data() for r in result]


def _select_next_entities(question: str, step_description: str, candidates: list[dict], *, max_pick: int = 8) -> list[str]:
    model = os.environ.get("GROQ_SELECT_MODEL", os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))

    # Provide only the candidate to_name list (LLM must choose from it; no hallucination).
    to_names: list[str] = []
    seen = set()
    for c in candidates:
        tn = str(c.get("to_name", "")).strip()
        if not tn:
            continue
        key = tn.lower()
        if key in seen:
            continue
        seen.add(key)
        to_names.append(tn)
        if len(to_names) >= 60:
            break

    prompt = f"""
You are selecting intermediate entities from provided knowledge-graph traversal candidates.

Rules:
* Output ONLY valid JSON.
* Output ONLY a JSON array of strings: ["entity1", "entity2", ...]
* Every selected entity MUST be taken from the provided to_name candidate list.
* If none fit the step goal, return [].

Question:
{question}

Step goal:
{step_description}

Candidates (to_name only):
{json.dumps(to_names)}
"""
    try:
        text = _llm_chat(prompt, model=model, max_tokens=128, temperature=0.0)
        parsed = _safe_parse_json_from_llm(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            # Filter to candidates only.
            allowed = {x.lower(): x for x in to_names}
            out: list[str] = []
            for x in parsed:
                s = x.strip()
                if not s:
                    continue
                if s.lower() in allowed and allowed[s.lower()] not in out:
                    out.append(allowed[s.lower()])
                if len(out) >= max_pick:
                    break
            return out
    except Exception:
        pass

    # Fallback: pick up to 3 candidates arbitrarily (still from candidates).
    return to_names[:3]


def _generate_final_answer(question: str, results: list[dict]) -> str:
    model = os.environ.get("GROQ_ANSWER_MODEL", os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))
    # Truncate results payload to keep prompts small.
    payload = json.dumps(results, ensure_ascii=False)
    if len(payload) > 6000:
        payload = payload[:6000] + "...(truncated)"

    prompt = f"""
You are an answer generator.
Answer the question clearly using ONLY the provided results.
If the results do not contain the answer, reply exactly:
"No data found in knowledge graph"

Question:
{question}

Results:
{payload}

Answer:
"""
    try:
        return _llm_chat(prompt, model=model, max_tokens=200, temperature=0.0)
    except Exception:
        return "Failed to generate answer"


def ask_question(question: str) -> dict:
    """
    Generalized KG question answering:
    1) Try direct Cypher via LLM (with retry once).
    2) If direct query yields nothing, do multi-hop traversal step-by-step with LLM-guided entity selection.
    """
    graph = Neo4jGraph()
    max_steps = 3
    try:
        with graph.driver.session() as session:
            count_res = session.run("MATCH ()-[r]->() RETURN count(r) AS c")
            total_triples = count_res.single().get("c", 0) if count_res else 0

        # 1) Direct cypher attempt
        direct_steps: list[str] = []
        cypher_attempts = 2
        direct_cypher = ""
        direct_results: list[dict] = []
        for i in range(cypher_attempts):
            direct_cypher = _generate_direct_cypher(question) if i == 0 else ""
            if not direct_cypher and i == 1:
                # Ask again with a correction instruction (simple heuristic).
                model = os.environ.get("GROQ_CYPHER_MODEL", os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))
                correction_prompt = f"""
Previous Cypher output was invalid or not executable.
Generate a corrected Cypher query.

Rules (same as before):
* Output ONLY Cypher text (no markdown).
* Use MATCH (a)-[r]->(b)
* RETURN a.name AS from_name, coalesce(r.relType, type(r)) AS relation_type, b.name AS to_name
* LIMIT 50

Question:
{question}
"""
                text = _llm_chat(correction_prompt, model=model, max_tokens=256, temperature=0.0)
                direct_cypher = normalise_cypher_response(text).strip().rstrip(";")

            if not direct_cypher:
                continue

            try:
                direct_results = _execute_cypher(graph, direct_cypher)
                if direct_results:
                    answer = _generate_final_answer(question, direct_results)
                    return {
                        "triples_added": total_triples,
                        "query": direct_cypher,
                        "results": direct_results,
                        "answer": answer,
                        "steps_taken": ["direct_cypher"]
                    }
            except Exception:
                direct_results = []

        # 2) Multi-hop traversal fallback
        start_entities = _extract_question_entities(question)
        if not start_entities:
            # Fallback: pick high-degree entities from the graph (helps when question doesn't contain exact names).
            with graph.driver.session() as session:
                r = session.run(
                    """
MATCH (e:Entity)
RETURN e.name AS name, size((e)--()) AS degree
ORDER BY degree DESC
LIMIT 15
"""
                )
                start_entities = [x["name"] for x in r if x.get("name")]

        steps = _decompose_question_steps(question, max_steps=max_steps)
        current_entities = start_entities[:8]
        hop_details: list[dict] = []
        last_traversal: list[dict] = []
        last_cypher = ""

        for idx, step_description in enumerate(steps[:max_steps]):
            if not current_entities:
                break

            traversal_cypher, candidates = _one_hop_traversal(graph, current_entities, limit=250)
            last_cypher = traversal_cypher

            # Avoid huge prompts; keep a reasonable candidate set.
            candidates_sorted = candidates[:200]
            last_traversal = candidates_sorted

            selected = _select_next_entities(
                question,
                step_description,
                candidates_sorted,
                max_pick=8,
            )

            hop_details.append(
                {
                    "step_idx": idx + 1,
                    "step_description": step_description,
                    "start_entities": current_entities,
                    "selected_entities": selected,
                    "traversal_candidates": candidates_sorted[:50],
                }
            )

            current_entities = selected

        if not last_traversal:
            return {
                "triples_added": total_triples,
                "query": last_cypher,
                "results": [],
                "answer": "No data found in knowledge graph",
                "steps_taken": hop_details,
            }

        # If we selected some entities, focus the final results on those.
        selected_set = {s.strip().lower() for s in hop_details[-1].get("selected_entities", [])} if hop_details else set()
        final_results = last_traversal
        if selected_set:
            final_results = [r for r in last_traversal if str(r.get("to_name", "")).strip().lower() in selected_set]

        answer = _generate_final_answer(question, final_results)
        return {
            "triples_added": total_triples,
            "query": last_cypher,
            "results": final_results,
            "answer": answer,
            "steps_taken": hop_details,
        }
    finally:
        # Close driver to avoid leaking connections.
        graph.close()

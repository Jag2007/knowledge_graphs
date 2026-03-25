from groq import Groq

from utils import load_env, normalise_cypher_response


class ReasoningEngine:
    """
    Simple query runner: question -> Cypher -> Neo4j result.
    """

    def __init__(self, query_engine):
        load_env()
        self.client = Groq()
        self.graph = query_engine.graph

    def _build_prompt(self, question: str) -> str:
        """Build the simple Cypher generation prompt."""
        return f"""
Convert the question into a Neo4j Cypher query.

Rules:
- Use (a)-[r]-(b) (ignore direction)
- Use exact entity names from question
- Return a.name, b.name, type(r)

Question:
{question}
""".strip()

    def generate_query(self, question: str) -> str:
        """Generate a Cypher query from the user's question."""
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Return only Neo4j Cypher query text.",
                },
                {"role": "user", "content": self._build_prompt(question)},
            ],
        )

        content = response.choices[0].message.content or ""
        return normalise_cypher_response(content)

    def answer_question(self, question: str) -> dict:
        """Convert the question, run the query, and return the raw result."""
        cypher = self.generate_query(question)
        result = self.graph.run_query(cypher)

        return {
            "question": question,
            "cypher": cypher,
            "results": result,
        }

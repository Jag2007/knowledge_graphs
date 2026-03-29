import os
from collections import defaultdict
from typing import Iterable

from dotenv import load_dotenv
from neo4j import GraphDatabase

from utils import normalise_relation_for_neo4j

load_dotenv()

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


class Neo4jGraph:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            )
            self.driver.verify_connectivity()
            self._ensure_schema()
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized. Check NEO4J_URI/USERNAME/PASSWORD.")

    def _ensure_schema(self) -> None:
        with self.driver.session() as session:
            session.run(
                """
                CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.name IS UNIQUE
                """
            )

    def close(self):
        if self.driver:
            self.driver.close()

    def count_triples(self) -> int:
        with self.driver.session() as session:
            record = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
            return int(record["c"]) if record and record.get("c") is not None else 0

    @staticmethod
    def _normalise_rows(triples: Iterable[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            relation = normalise_relation_for_neo4j(triple.get("relation", ""))
            object_ = str(triple.get("object", "")).strip()
            if not subject or not relation or not object_:
                continue
            grouped[relation].append(
                {
                    "subject": subject,
                    "object": object_,
                }
            )
        return grouped

    def insert_triples(self, triples: list[dict]) -> int:
        """Insert triples into Neo4j in grouped UNWIND batches."""
        if not triples:
            return 0

        grouped = self._normalise_rows(triples)
        if not grouped:
            return 0

        inserted = 0
        with self.driver.session() as session:
            for rel_type, rows in grouped.items():
                query = f"""
                UNWIND $rows AS row
                MERGE (s:Entity {{name: row.subject}})
                MERGE (o:Entity {{name: row.object}})
                MERGE (s)-[r:{rel_type}]->(o)
                SET r.relType = $rel_type
                """
                session.run(query, rows=rows, rel_type=rel_type)
                inserted += len(rows)

        return inserted

    def search_related(self, terms: list[str], limit: int = 25) -> tuple[str, list[dict]]:
        """Fast one-hop lookup for the most relevant terms from the question."""
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        cypher = """
        MATCH (a:Entity)-[r]-(b:Entity)
        WHERE ANY(term IN $terms WHERE toLower(a.name) CONTAINS term OR toLower(b.name) CONTAINS term)
        RETURN DISTINCT a.name AS from_name,
               coalesce(r.relType, type(r)) AS relation_type,
               b.name AS to_name
        LIMIT $limit
        """.strip()
        if not cleaned_terms:
            return cypher, []

        with self.driver.session() as session:
            rows = session.run(cypher, terms=cleaned_terms[:12], limit=limit)
            return cypher, [row.data() for row in rows]

    def search_paths(self, terms: list[str], max_hops: int = 2, limit: int = 20) -> tuple[str, list[dict]]:
        """Fallback multi-hop traversal for questions that need linked facts."""
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        cypher = f"""
        MATCH (anchor:Entity)
        WHERE ANY(term IN $terms WHERE toLower(anchor.name) CONTAINS term)
        MATCH p=(anchor)-[rels*1..{max_hops}]-(target:Entity)
        RETURN DISTINCT [node IN nodes(p) | node.name] AS path_nodes,
               [rel IN relationships(p) | coalesce(rel.relType, type(rel))] AS path_relationships
        LIMIT $limit
        """.strip()
        if not cleaned_terms:
            return cypher, []

        with self.driver.session() as session:
            rows = session.run(cypher, terms=cleaned_terms[:12], limit=limit)
            return cypher, [row.data() for row in rows]

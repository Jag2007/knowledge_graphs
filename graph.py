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
            session.run(
                """
                CREATE CONSTRAINT document_id_unique IF NOT EXISTS
                FOR (d:Document) REQUIRE d.document_id IS UNIQUE
                """
            )

    def close(self):
        if self.driver:
            self.driver.close()

    def store_document(self, document_id: str, file_name: str, summary: str) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {document_id: $document_id})
                SET d.file_name = $file_name,
                    d.summary = $summary,
                    d.updated_at = timestamp()
                """,
                document_id=document_id,
                file_name=file_name,
                summary=summary,
            )

    def count_triples(self, document_id: str | None = None) -> int:
        query = "MATCH ()-[r]->()"
        params = {}
        if document_id:
            query += " WHERE r.document_id = $document_id"
            params["document_id"] = document_id
        query += " RETURN count(r) AS c"
        with self.driver.session() as session:
            record = session.run(query, **params).single()
            return int(record["c"]) if record and record.get("c") is not None else 0

    @staticmethod
    def _normalise_rows(triples: Iterable[dict], document_id: str) -> dict[str, list[dict]]:
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
                    "document_id": document_id,
                }
            )
        return grouped

    def insert_triples(self, triples: list[dict], document_id: str) -> int:
        """Insert triples into Neo4j in grouped UNWIND batches scoped to one document."""
        if not triples or not document_id:
            return 0

        grouped = self._normalise_rows(triples, document_id)
        if not grouped:
            return 0

        inserted = 0
        with self.driver.session() as session:
            for rel_type, rows in grouped.items():
                query = f"""
                UNWIND $rows AS row
                MERGE (s:Entity {{name: row.subject}})
                MERGE (o:Entity {{name: row.object}})
                MERGE (d:Document {{document_id: row.document_id}})
                MERGE (s)-[r:{rel_type} {{document_id: row.document_id}}]->(o)
                SET r.relType = $rel_type
                MERGE (d)-[:MENTIONS]->(s)
                MERGE (d)-[:MENTIONS]->(o)
                """
                session.run(query, rows=rows, rel_type=rel_type)
                inserted += len(rows)

        return inserted

    def get_document_summary(self, document_id: str) -> str:
        with self.driver.session() as session:
            record = session.run(
                """
                MATCH (d:Document {document_id: $document_id})
                RETURN d.summary AS summary
                LIMIT 1
                """,
                document_id=document_id,
            ).single()
            return str(record["summary"]).strip() if record and record.get("summary") else ""

    def find_relevant_entities(self, terms: list[str], document_id: str, limit: int = 10) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        cypher = """
        MATCH (d:Document {document_id: $document_id})-[:MENTIONS]->(e:Entity)
        WITH DISTINCT e
        WITH e, toLower(e.name) AS lowered_name
        WHERE ANY(term IN $terms WHERE lowered_name CONTAINS term)
        RETURN e.name AS entity_name
        ORDER BY size(e.name) DESC
        LIMIT $limit
        """.strip()
        if not cleaned_terms or not document_id:
            return cypher, []

        with self.driver.session() as session:
            rows = session.run(cypher, document_id=document_id, terms=cleaned_terms[:20], limit=limit)
            return cypher, [row.data() for row in rows]

    def get_entity_neighborhood(self, entity_name: str, document_id: str, limit: int = 20) -> tuple[str, list[dict]]:
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower($entity_name)
        MATCH (e)-[r]-(other:Entity)
        WHERE r.document_id = $document_id
        RETURN e.name AS center_name,
               startNode(r).name AS from_name,
               coalesce(r.relType, type(r)) AS relation_type,
               endNode(r).name AS to_name,
               other.name AS neighbor_name
        LIMIT $limit
        """.strip()
        if not entity_name or not document_id:
            return cypher, []

        with self.driver.session() as session:
            rows = session.run(cypher, entity_name=entity_name, document_id=document_id, limit=limit)
            return cypher, [row.data() for row in rows]

    def search_related(self, terms: list[str], document_id: str, limit: int = 25) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        cypher = """
        MATCH (a:Entity)-[r]-(b:Entity)
        WHERE r.document_id = $document_id
          AND ANY(term IN $terms WHERE toLower(a.name) CONTAINS term OR toLower(b.name) CONTAINS term)
        RETURN DISTINCT a.name AS from_name,
               coalesce(r.relType, type(r)) AS relation_type,
               b.name AS to_name
        LIMIT $limit
        """.strip()
        if not cleaned_terms or not document_id:
            return cypher, []

        with self.driver.session() as session:
            rows = session.run(cypher, document_id=document_id, terms=cleaned_terms[:12], limit=limit)
            return cypher, [row.data() for row in rows]

    def search_semantic(self, terms: list[str], document_id: str, limit: int = 40) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        cypher = """
        MATCH (a:Entity)-[r]-(b:Entity)
        WHERE r.document_id = $document_id
        WITH a,
             b,
             coalesce(r.relType, type(r)) AS relation_type,
             toLower(a.name + " " + coalesce(r.relType, type(r)) + " " + b.name) AS searchable_text
        WHERE ANY(term IN $terms WHERE searchable_text CONTAINS term)
        RETURN DISTINCT a.name AS from_name,
               relation_type AS relation_type,
               b.name AS to_name,
               searchable_text AS searchable_text
        LIMIT $limit
        """.strip()
        if not cleaned_terms or not document_id:
            return cypher, []

        with self.driver.session() as session:
            rows = session.run(cypher, document_id=document_id, terms=cleaned_terms[:20], limit=limit)
            return cypher, [row.data() for row in rows]

    def search_paths(self, terms: list[str], document_id: str, max_hops: int = 2, limit: int = 20) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        cypher = f"""
        MATCH (anchor:Entity)-[seed]-()
        WHERE seed.document_id = $document_id
          AND ANY(term IN $terms WHERE toLower(anchor.name) CONTAINS term)
        MATCH p=(anchor)-[rels*1..{max_hops}]-(target:Entity)
        WHERE ALL(rel IN relationships(p) WHERE rel.document_id = $document_id)
        RETURN DISTINCT [node IN nodes(p) | node.name] AS path_nodes,
               [rel IN relationships(p) | coalesce(rel.relType, type(rel))] AS path_relationships
        LIMIT $limit
        """.strip()
        if not cleaned_terms or not document_id:
            return cypher, []

        with self.driver.session() as session:
            rows = session.run(cypher, document_id=document_id, terms=cleaned_terms[:12], limit=limit)
            return cypher, [row.data() for row in rows]

    def get_graph_overview(self, document_id: str, limit: int = 12) -> tuple[str, list[dict]]:
        cypher = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE r.document_id = $document_id
        WITH a, b, coalesce(r.relType, type(r)) AS relation_type
        RETURN a.name AS from_name,
               relation_type AS relation_type,
               b.name AS to_name
        LIMIT $limit
        """.strip()
        if not document_id:
            return cypher, []
        with self.driver.session() as session:
            rows = session.run(cypher, document_id=document_id, limit=limit)
            return cypher, [row.data() for row in rows]

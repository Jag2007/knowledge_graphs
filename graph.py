import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from utils import normalise_relation_for_neo4j

# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

class Neo4jGraph:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized. Check NEO4J_URI/USERNAME/PASSWORD.")

    def close(self):
        if self.driver:
            self.driver.close()

    def insert_triple(self, subject: str, relation: str, object_: str) -> None:
        """Insert a single triple into Neo4j."""
        rel_type = normalise_relation_for_neo4j(relation)

        query = f"""
        MERGE (s:Entity {{name: $subject}})
        MERGE (o:Entity {{name: $object}})
        MERGE (s)-[r:{rel_type}]->(o)
        SET r.relType = $rel_type
        """
        with self.driver.session() as session:
            session.run(
                query,
                subject=subject,
                object=object_,
                rel_type=rel_type,
            )

    @staticmethod
    def _insert_triple_tx(tx, subject: str, relation: str, object_: str) -> None:
        """Insert a triple using an existing Neo4j transaction."""
        rel_type = normalise_relation_for_neo4j(relation)
        query = f"""
        MERGE (s:Entity {{name: $subject}})
        MERGE (o:Entity {{name: $object}})
        MERGE (s)-[r:{rel_type}]->(o)
        SET r.relType = $rel_type
        """
        tx.run(
            query,
            subject=subject,
            object=object_,
            rel_type=rel_type,
        )

    def insert_triples(self, triples: list[dict]) -> int:
        """
        Insert triples into the Neo4j graph.
        Returns the number of triples attempted (validated triples passed in).
        """
        if not triples:
            return 0

        # Use a transaction for speed/reliability.
        with self.driver.session() as session:
            tx = session.begin_transaction()
            try:
                for t in triples:
                    self._insert_triple_tx(tx, t["subject"], t["relation"], t["object"])
                tx.commit()
            except Exception as e:
                tx.rollback()
                raise RuntimeError(f"Neo4j error while inserting triples: {e}") from e

        return len(triples)

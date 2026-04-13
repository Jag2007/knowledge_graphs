import os
import unittest

from fastapi.testclient import TestClient

import app
from kg_app.api import server as app_server
from kg_app.core import extractor, query_engine, utils
from kg_app.state import chunk_store

os.environ.setdefault("KG_EMBEDDING_BACKEND", "local")


class FakeGraph:
    stored = {}
    summaries = {}

    def __init__(self):
        self.driver = object()

    def store_document(self, document_id, file_name, summary):
        FakeGraph.summaries[document_id] = {
            "file_name": file_name,
            "summary": summary,
        }

    def insert_triples(self, triples, document_id):
        FakeGraph.stored[document_id] = list(triples)
        return len(triples)

    def count_triples(self, document_id=None):
        if not document_id:
            return sum(len(value) for value in FakeGraph.stored.values())
        return len(FakeGraph.stored.get(document_id, []))

    def search_related(self, terms, document_id, limit=25):
        rows = []
        for triple in FakeGraph.stored.get(document_id, []):
            if any(
                term.lower() in triple["subject"].lower() or term.lower() in triple["object"].lower()
                for term in terms
            ):
                rows.append(
                    {
                        "from_name": triple["subject"],
                        "relation_type": triple["relation"],
                        "to_name": triple["object"],
                    }
                )
        return "MATCH (a)-[r]-(b)", rows[:limit]

    def search_paths(self, terms, document_id, max_hops=2, limit=20):
        triples = FakeGraph.stored.get(document_id, [])
        paths = []
        for first in triples:
            if terms and not any(
                term.lower() in first["subject"].lower() or term.lower() in first["object"].lower()
                for term in terms
            ):
                continue
            for second in triples:
                if first["object"].lower() != second["subject"].lower():
                    continue
                paths.append(
                    {
                        "path_nodes": [
                            first["subject"],
                            first["object"],
                            second["object"],
                        ],
                        "path_relationships": [
                            first["relation"],
                            second["relation"],
                        ],
                    }
                )
        if paths:
            return "MATCH p=(a)-[r*1..2]-(b)", paths[:limit]
        return "MATCH p=(a)-[r*1..2]-(b)", []

    def search_semantic(self, terms, document_id, limit=40):
        rows = []
        for triple in FakeGraph.stored.get(document_id, []):
            searchable = f"{triple['subject']} {triple['relation']} {triple['object']}".lower()
            if any(term.lower() in searchable for term in terms):
                rows.append(
                    {
                        "from_name": triple["subject"],
                        "relation_type": triple["relation"],
                        "to_name": triple["object"],
                        "searchable_text": searchable,
                    }
                )
        return "MATCH (a)-[r]-(b) semantic", rows[:limit]

    def get_graph_overview(self, document_id, limit=12):
        rows = [
            {
                "from_name": triple["subject"],
                "relation_type": triple["relation"],
                "to_name": triple["object"],
            }
            for triple in FakeGraph.stored.get(document_id, [])[:limit]
        ]
        return "MATCH (a)-[r]->(b) overview", rows

    def get_document_summary(self, document_id):
        payload = FakeGraph.summaries.get(document_id, {})
        return payload.get("summary", "")

    def find_relevant_entities(self, terms, document_id, limit=10):
        candidates = []
        seen = set()
        for triple in FakeGraph.stored.get(document_id, []):
            for value in (triple["subject"], triple["object"]):
                lowered = value.lower()
                if lowered in seen:
                    continue
                if any(term.lower() in lowered for term in terms):
                    seen.add(lowered)
                    candidates.append({"entity_name": value})
        if not candidates:
            for triple in FakeGraph.stored.get(document_id, []):
                for value in (triple["subject"], triple["object"]):
                    lowered = value.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    candidates.append({"entity_name": value})
        return "MATCH (d)-[:MENTIONS]->(e)", candidates[:limit]

    def get_entity_neighborhood(self, entity_name, document_id, limit=20):
        rows = []
        lowered = entity_name.lower()
        for triple in FakeGraph.stored.get(document_id, []):
            if triple["subject"].lower() == lowered or triple["object"].lower() == lowered:
                rows.append(
                    {
                        "center_name": entity_name,
                        "from_name": triple["subject"],
                        "relation_type": triple["relation"],
                        "to_name": triple["object"],
                        "neighbor_name": triple["object"] if triple["subject"].lower() == lowered else triple["subject"],
                    }
                )
        return "MATCH (e)-[r]-(other)", rows[:limit]

    def close(self):
        pass


class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        self.original_graph_app = app_server.GraphStore
        self.original_graph_query = query_engine.GraphStore
        self.original_extract_text = app_server.extract_text_from_pdf
        self.original_extract_triples = app_server.extract_triples_groq
        self.original_precompute_metadata = app_server.precompute_chunk_metadata
        self.original_set_active_document = app_server.set_active_document
        self.original_get_active_document = app_server.get_active_document

        app_server.GraphStore = FakeGraph
        query_engine.GraphStore = FakeGraph
        app_server.precompute_chunk_metadata = lambda chunks: chunks
        FakeGraph.stored = {}
        FakeGraph.summaries = {}
        chunk_store.clear_all_document_chunks()
        self.active_document = {}
        app_server.set_active_document = self._set_active_document
        app_server.get_active_document = self._get_active_document
        self.client = TestClient(app.app)

    def tearDown(self):
        app_server.GraphStore = self.original_graph_app
        query_engine.GraphStore = self.original_graph_query
        app_server.extract_text_from_pdf = self.original_extract_text
        app_server.extract_triples_groq = self.original_extract_triples
        app_server.precompute_chunk_metadata = self.original_precompute_metadata
        app_server.set_active_document = self.original_set_active_document
        app_server.get_active_document = self.original_get_active_document
        chunk_store.clear_all_document_chunks()

    def _set_active_document(self, document_id, file_name):
        self.active_document = {"document_id": document_id, "file_name": file_name}

    def _get_active_document(self):
        return dict(self.active_document)

    def test_rejects_non_pdf_upload(self):
        response = self.client.post("/upload_pdf", files={"file": ("notes.txt", b"hello", "text/plain")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"], "Please upload a PDF file.")

    def test_rejects_empty_pdf(self):
        response = self.client.post("/upload_pdf", files={"file": ("empty.pdf", b"", "application/pdf")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"], "The uploaded PDF is empty.")

    def test_handles_missing_text(self):
        app_server.extract_text_from_pdf = lambda _: "   "
        response = self.client.post("/upload_pdf", files={"file": ("empty.pdf", b"%PDF", "application/pdf")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"], "No readable text was found in the uploaded PDF.")

    def test_handles_documents_with_no_triples(self):
        app_server.extract_text_from_pdf = lambda _: "Just plain text without useful relations."
        app_server.extract_triples_groq = lambda chunk: []
        response = self.client.post("/upload_pdf", files={"file": ("plain.pdf", b"%PDF", "application/pdf")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"], "No structured knowledge was extracted from the document.")

    def test_upload_and_ask_flow(self):
        app_server.extract_text_from_pdf = lambda _: "Mukesh Ambani chairs Reliance Industries. Reliance Industries is based in India."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Mukesh Ambani", "relation": "CHAIRMAN_OF", "object": "Reliance Industries"},
            {"subject": "Reliance Industries", "relation": "BASED_IN", "object": "India"},
        ]

        upload = self.client.post("/upload_pdf", files={"file": ("sample.pdf", b"%PDF", "application/pdf")})
        self.assertEqual(upload.status_code, 200)
        self.assertEqual(upload.json()["triples_added"], 2)

        ask = self.client.post("/ask", json={"question": "who is mukesh ambani"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(ask.json()["answer"], "Mukesh Ambani is the chairman of Reliance Industries.")

    def test_rejects_empty_question(self):
        response = self.client.post("/ask", json={"question": "   "})
        self.assertEqual(response.status_code, 400)

    def test_overview_question_returns_summary(self):
        app_server.extract_text_from_pdf = lambda _: "Indian culture includes festivals, cuisine, and dance."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Culture", "relation": "INCLUDES", "object": "Festivals"},
            {"subject": "Indian Culture", "relation": "INCLUDES", "object": "Cuisine"},
            {"subject": "Indian Culture", "relation": "INCLUDES", "object": "Dance"},
        ]
        self.client.post("/upload_pdf", files={"file": ("sample.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what does this pdf talk about"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("The uploaded document mainly discusses", ask.json()["answer"])

    def test_overview_question_accepts_talking_about_variant(self):
        app_server.extract_text_from_pdf = lambda _: "The Indian Constitution discusses fundamental rights and duties."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Constitution", "relation": "INCLUDES", "object": "Fundamental Rights"},
            {"subject": "Indian Constitution", "relation": "INCLUDES", "object": "Fundamental Duties"},
        ]
        self.client.post("/upload_pdf", files={"file": ("constitution.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what is this pdf talking about"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("The uploaded document mainly discusses", ask.json()["answer"])
        self.assertIn("Indian Constitution", ask.json()["answer"])

    def test_semantic_retrieval_matches_relation_text(self):
        app_server.extract_text_from_pdf = lambda _: "Indian culture includes festivals."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Culture", "relation": "INCLUDES", "object": "Festivals"},
        ]
        self.client.post("/upload_pdf", files={"file": ("sample.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what festivals are mentioned"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("Festivals", ask.json()["answer"])

    def test_semantic_chunk_retrieval_answers_when_graph_is_too_weak(self):
        app_server.extract_text_from_pdf = lambda _: (
            "A good Constitution does not allow whims to change its basic structure. "
            "It does not allow easy overthrow of provisions that guarantee rights of citizens."
        )
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Constitution", "relation": "INCLUDES", "object": "Rights"},
        ]
        self.client.post("/upload_pdf", files={"file": ("constitution.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what is a good constitution"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("good Constitution", ask.json()["answer"])
        self.assertIn("basic structure", ask.json()["answer"])

    def test_hybrid_chunk_results_include_embedding_score_and_source_id(self):
        app_server.extract_text_from_pdf = lambda _: (
            "The legislature, the executive, and the judiciary are the three organs of government. "
            "Each organ exercises different powers."
        )
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Government", "relation": "INCLUDES", "object": "Legislature"},
        ]
        self.client.post("/upload_pdf", files={"file": ("government.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what are the three organs of government"})
        self.assertEqual(ask.status_code, 200)
        chunk_results = [row for row in ask.json()["results"] if "score_breakdown" in row]
        self.assertTrue(chunk_results)
        self.assertTrue(chunk_results[0]["chunk_id"])
        self.assertIn("embedding", chunk_results[0]["score_breakdown"])

    def test_current_document_scope_prevents_cross_document_leakage(self):
        app_server.extract_text_from_pdf = lambda _: "Karnataka has capital Bengaluru."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Karnataka", "relation": "HAS_CAPITAL", "object": "Bengaluru"},
        ]
        first_upload = self.client.post("/upload_pdf", files={"file": ("karnataka.pdf", b"%PDF", "application/pdf")})
        self.assertEqual(first_upload.status_code, 200)

        app_server.extract_text_from_pdf = lambda _: "Indian culture includes festivals and cuisine."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Culture", "relation": "INCLUDES", "object": "Festivals"},
            {"subject": "Indian Culture", "relation": "INCLUDES", "object": "Cuisine"},
        ]
        second_upload = self.client.post("/upload_pdf", files={"file": ("culture.pdf", b"%PDF", "application/pdf")})
        self.assertEqual(second_upload.status_code, 200)

        ask = self.client.post("/ask", json={"question": "what does this pdf talk about"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("Indian Culture", ask.json()["answer"])
        self.assertNotIn("Karnataka", ask.json()["answer"])

    def test_capital_question_gets_natural_answer(self):
        app_server.extract_text_from_pdf = lambda _: "Karnataka has capital Bengaluru."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Karnataka", "relation": "HAS_CAPITAL", "object": "Bengaluru"},
        ]
        self.client.post("/upload_pdf", files={"file": ("karnataka.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what is the capital of karnataka"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(ask.json()["answer"], "The capital of Karnataka is Bengaluru.")

    def test_node_overview_uses_anchor_entity_context(self):
        app_server.extract_text_from_pdf = lambda _: "Indian tradition includes spirituality, hospitality, and family values."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Tradition", "relation": "INCLUDES", "object": "Spirituality"},
            {"subject": "Indian Tradition", "relation": "INCLUDES", "object": "Hospitality"},
            {"subject": "Indian Tradition", "relation": "INCLUDES", "object": "Family Values"},
        ]
        self.client.post("/upload_pdf", files={"file": ("tradition.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what do you know about indian tradition"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("Indian Tradition includes Spirituality", ask.json()["answer"])
        self.assertIn("Hospitality", ask.json()["answer"])

    def test_what_are_question_stays_on_requested_node(self):
        app_server.extract_text_from_pdf = lambda _: "Saris reflect Indian tradition."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Saris", "relation": "REFLECTS", "object": "Indian Tradition"},
        ]
        self.client.post("/upload_pdf", files={"file": ("saris.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what are saris"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(ask.json()["answer"], "Saris reflect Indian Tradition.")

    def test_include_question_prefers_include_neighbors(self):
        app_server.extract_text_from_pdf = lambda _: "Festivals include rituals, food, and gatherings. India celebrates festivals."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Festival", "relation": "INCLUDES", "object": "Rituals"},
            {"subject": "Festival", "relation": "INCLUDES", "object": "Food"},
            {"subject": "Festival", "relation": "INCLUDES", "object": "Gatherings"},
            {"subject": "India", "relation": "CELEBRATES", "object": "Festivals"},
        ]
        self.client.post("/upload_pdf", files={"file": ("festival.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what does festival include?"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(ask.json()["answer"], "Festival includes Rituals, Food, and Gatherings.")

    def test_what_is_question_returns_node_summary(self):
        app_server.extract_text_from_pdf = lambda _: "Indian tradition includes spirituality, hospitality, and family values."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Tradition", "relation": "INCLUDES", "object": "Spirituality"},
            {"subject": "Indian Tradition", "relation": "INCLUDES", "object": "Hospitality"},
            {"subject": "Indian Tradition", "relation": "INCLUDES", "object": "Family Values"},
        ]
        self.client.post("/upload_pdf", files={"file": ("tradition.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what is indian tradition"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(
            ask.json()["answer"],
            "Indian Tradition includes Spirituality, Hospitality, and Family Values.",
        )

    def test_celebrate_question_aggregates_related_values(self):
        app_server.extract_text_from_pdf = lambda _: "Indian culture celebrates family, respect, spirituality, and hospitality."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Family"},
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Respect"},
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Spirituality"},
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Hospitality"},
            {"subject": "Tradition", "relation": "VARY_BY", "object": "Clothing"},
        ]
        self.client.post("/upload_pdf", files={"file": ("culture.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what does indian culture celebrate"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(
            ask.json()["answer"],
            "Indian Culture celebrates Family, Respect, Spirituality, and Hospitality.",
        )

    def test_general_entity_question_returns_multi_fact_summary(self):
        app_server.extract_text_from_pdf = lambda _: "Indian culture reflects unity in diversity and celebrates family, respect, spirituality, and hospitality."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Culture", "relation": "REFLECTS", "object": "Unity In Diversity"},
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Family"},
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Respect"},
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Spirituality"},
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Hospitality"},
            {"subject": "Culture", "relation": "INFLUENCES", "object": "Global Trends"},
        ]
        self.client.post("/upload_pdf", files={"file": ("culture.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what is a INDIAN CULTURE"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(
            ask.json()["answer"],
            "Based on the uploaded document: Indian Culture reflects Unity In Diversity; Indian Culture celebrates Family, Respect, Spirituality, and Hospitality.",
        )

    def test_relation_specific_query_does_not_fall_back_to_unrelated_edge(self):
        app_server.extract_text_from_pdf = lambda _: "Indian culture celebrates family and respect. Culture influences global trends."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Family"},
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Respect"},
            {"subject": "Culture", "relation": "INFLUENCES", "object": "Global Trends"},
        ]
        self.client.post("/upload_pdf", files={"file": ("culture.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what does indian culture CELEBRATES"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(
            ask.json()["answer"],
            "Indian Culture celebrates Family and Respect.",
        )

    def test_relation_specific_question_uses_requested_relation_family_generically(self):
        app_server.extract_text_from_pdf = lambda _: "The Indian Constitution guarantees justice, equality, and liberty."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Constitution", "relation": "GUARANTEES", "object": "Justice"},
            {"subject": "Indian Constitution", "relation": "GUARANTEES", "object": "Equality"},
            {"subject": "Indian Constitution", "relation": "GUARANTEES", "object": "Liberty"},
            {"subject": "Indian Constitution", "relation": "INCLUDES", "object": "Constituent Assembly"},
        ]
        self.client.post("/upload_pdf", files={"file": ("constitution.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what does Indian Constitution guarantees"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(
            ask.json()["answer"],
            "Indian Constitution guarantees Justice, Equality, and Liberty.",
        )

    def test_relation_question_prefers_edges_covering_multiple_query_terms(self):
        app_server.extract_text_from_pdf = lambda _: "India was partitioned from Pakistan. India practices federalism."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "India", "relation": "PARTITIONED_FROM", "object": "Pakistan"},
            {"subject": "India", "relation": "PRACTICES", "object": "Federalism"},
        ]
        self.client.post("/upload_pdf", files={"file": ("partition.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what do you know about india and pakistan separation"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("India partitioned from Pakistan", ask.json()["answer"])
        self.assertNotIn("federalism", ask.json()["answer"].lower())

    def test_direct_formatter_aggregates_all_relation_values(self):
        results = [
            {"from_name": "Indian Culture", "relation_type": "CELEBRATES", "to_name": "Family"},
            {"from_name": "Indian Culture", "relation_type": "CELEBRATES", "to_name": "Respect"},
            {"from_name": "Indian Culture", "relation_type": "CELEBRATES", "to_name": "Hospitality"},
        ]
        answer = query_engine._format_direct_results(
            "what does indian culture celebrate",
            results,
            ["indian", "culture", "celebrate"],
            {"CELEBRATES"},
        )
        self.assertEqual(
            answer,
            "Indian Culture celebrates Family, Respect, and Hospitality.",
        )

    def test_entity_only_question_returns_all_relation_groups(self):
        app_server.extract_text_from_pdf = lambda _: "Indian culture reflects unity in diversity, celebrates family and hospitality, and practices yoga."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Culture", "relation": "REFLECTS", "object": "Unity In Diversity"},
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Family"},
            {"subject": "Indian Culture", "relation": "CELEBRATES", "object": "Hospitality"},
            {"subject": "Indian Culture", "relation": "PRACTICES", "object": "Yoga"},
        ]
        self.client.post("/upload_pdf", files={"file": ("culture.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "indian culture"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(
            ask.json()["answer"],
            "Based on the uploaded document: Indian Culture reflects Unity In Diversity; Indian Culture celebrates Family and Hospitality; Indian Culture practices Yoga.",
        )

    def test_indirect_multi_hop_query_uses_path_chain(self):
        app_server.extract_text_from_pdf = lambda _: "Elon Musk founded SpaceX. SpaceX is based in USA."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Elon Musk", "relation": "FOUNDED", "object": "SpaceX"},
            {"subject": "SpaceX", "relation": "BASED_IN", "object": "USA"},
        ]
        self.client.post("/upload_pdf", files={"file": ("spacex.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post(
            "/ask",
            json={"question": "where is the company founded by elon musk based"},
        )
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(
            ask.json()["answer"],
            "Elon Musk founded SpaceX, and SpaceX is located in USA.",
        )

    def test_inverse_relation_answer_is_grammatically_natural(self):
        app_server.extract_text_from_pdf = lambda _: "Indian cuisine includes vegetables."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Cuisine", "relation": "INCLUDES", "object": "Vegetables"},
        ]
        self.client.post("/upload_pdf", files={"file": ("food.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what are vegetables"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(
            ask.json()["answer"],
            "Vegetables are included in Indian Cuisine.",
        )

    def test_unrelated_question_does_not_return_false_positive(self):
        app_server.extract_text_from_pdf = lambda _: "Baba Saheb Dr Ambedkar is known as Father of the Indian Constitution."
        app_server.extract_triples_groq = lambda chunk: [
            {
                "subject": "Baba Saheb Dr Ambedkar",
                "relation": "KNOWN_AS",
                "object": "Father of the Indian Constitution",
            },
        ]
        self.client.post("/upload_pdf", files={"file": ("constitution.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what do you know about second world war"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(
            ask.json()["answer"],
            "It is not in the uploaded document. Please check the text.",
        )

    def test_adopted_query_uses_path_when_relation_is_on_neighbor_node(self):
        app_server.extract_text_from_pdf = lambda _: (
            "The Constituent Assembly drafted the Indian Constitution. "
            "The Indian Constitution was adopted on 26 November 1949."
        )
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Constituent Assembly", "relation": "DRAFTED", "object": "Indian Constitution"},
            {"subject": "Constituent Assembly", "relation": "CONVENED_IN", "object": "December 1946"},
            {"subject": "Indian Constitution", "relation": "ADOPTED_ON", "object": "26 November 1949"},
        ]
        self.client.post("/upload_pdf", files={"file": ("constitution.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "Constituent Assembly adopted on"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("Indian Constitution was adopted on 26 November 1949", ask.json()["answer"])

    def test_chunking_preserves_sentence_boundaries(self):
        text = (
            "The Indian Constitution was drafted by the Constituent Assembly. "
            "These are the legislature, the executive and the judiciary. "
            "The judiciary refers to the system of courts."
        )
        chunks = utils.chunk_text(
            text,
            target_words=8,
            min_words=5,
            max_words=12,
            overlap_words=4,
        )
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.strip().endswith((".", "!", "?")) for chunk in chunks))
        self.assertTrue(any("These are the legislature" in chunk for chunk in chunks))

    def test_hybrid_chunks_include_metadata_and_sentence_safe_text(self):
        pages = [
            {
                "page": 2,
                "text": (
                    "Introduction\n"
                    "The hospital queue system manages patient flow. "
                    "It reduces waiting time and improves service quality.\n\n"
                    "Implementation\n"
                    "The queue uses tokens and priority rules. "
                    "The staff dashboard tracks every patient request."
                ),
            }
        ]
        chunks = utils.build_hybrid_chunks(
            pages,
            target_words=10,
            min_words=5,
            max_words=18,
            overlap_ratio=0.20,
        )
        self.assertTrue(chunks)
        first = chunks[0]
        self.assertEqual(first["id"], "chunk_1")
        self.assertIn("text", first)
        self.assertIn("summary", first)
        self.assertIn("keywords", first)
        self.assertIn("section", first)
        self.assertEqual(first["page"], 2)
        self.assertTrue(first["text"].endswith((".", "!", "?")))
        self.assertTrue(first["keywords"])

    def test_context_extraction_links_followup_list_sentence(self):
        triples = extractor._extract_interlinked_context_triples(
            "According to the Constitution, there are three organs of government. "
            "These are the legislature, the executive and the judiciary."
        )
        cleaned = extractor.clean_and_validate_triples(triples)
        self.assertIn(
            {
                "subject": "three organs of government",
                "relation": "INCLUDES",
                "object": "legislature",
            },
            cleaned,
        )
        self.assertIn(
            {
                "subject": "three organs of government",
                "relation": "INCLUDES",
                "object": "executive",
            },
            cleaned,
        )
        self.assertIn(
            {
                "subject": "three organs of government",
                "relation": "INCLUDES",
                "object": "judiciary",
            },
            cleaned,
        )

    def test_context_extraction_keeps_adjective_qualified_entity_and_pronoun_followup(self):
        triples = extractor._extract_interlinked_context_triples(
            "A good Constitution does not allow these whims to change its basic structure. "
            "It does not allow for the easy overthrow of provisions that guarantee rights of citizens."
        )
        cleaned = extractor.clean_and_validate_triples(triples)
        self.assertIn(
            {
                "subject": "good Constitution",
                "relation": "DOES_NOT_ALLOW",
                "object": "whims to change its basic structure",
            },
            cleaned,
        )
        self.assertIn(
            {
                "subject": "good Constitution",
                "relation": "DOES_NOT_ALLOW",
                "object": "easy overthrow of provisions that guarantee rights of citizens",
            },
            cleaned,
        )

    def test_clean_and_validate_triples_drops_placeholder_objects(self):
        cleaned = extractor.clean_and_validate_triples(
            [
                {"subject": "Indian Constitution", "relation": "CONVENED_IN", "object": "Not Specified"},
                {"subject": "Indian Constitution", "relation": "INCLUDES", "object": "Fundamental Rights"},
            ]
        )
        self.assertEqual(
            cleaned,
            [
                {
                    "subject": "Indian Constitution",
                    "relation": "INCLUDES",
                    "object": "Fundamental Rights",
                }
            ],
        )

    def test_entity_only_answer_includes_incoming_and_outgoing_facts(self):
        app_server.extract_text_from_pdf = lambda _: "Indian Constitution includes Fundamental Rights. Constituent Assembly drafted Indian Constitution."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Constitution", "relation": "INCLUDES", "object": "Fundamental Rights"},
            {"subject": "Constituent Assembly", "relation": "DRAFTED", "object": "Indian Constitution"},
        ]
        self.client.post("/upload_pdf", files={"file": ("constitution.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "Indian Constitution"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("Indian Constitution includes Fundamental Rights", ask.json()["answer"])
        self.assertIn("Constituent Assembly drafted Indian Constitution", ask.json()["answer"])

    def test_adjective_qualified_entity_question_prefers_specific_node(self):
        app_server.extract_text_from_pdf = lambda _: "A good Constitution protects basic structure."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Constitution", "relation": "INCLUDES", "object": "Lists"},
            {"subject": "Good Constitution", "relation": "DOES_NOT_ALLOW", "object": "Easy Overthrow of Provisions"},
            {"subject": "Good Constitution", "relation": "PROTECTS", "object": "Rights of Citizens"},
        ]
        self.client.post("/upload_pdf", files={"file": ("constitution.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what is a good constitution"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("Good Constitution does not allow Easy Overthrow of Provisions", ask.json()["answer"])
        self.assertIn("Good Constitution protects Rights of Citizens", ask.json()["answer"])
        self.assertNotIn("Constitution includes Lists", ask.json()["answer"])

    def test_anchor_ranking_prefers_base_entity_when_modifier_is_not_requested(self):
        ranked = query_engine._rank_anchor_entities(
            terms=["forest"],
            phrases=[],
            candidates=[
                {"entity_name": "Green Forest"},
                {"entity_name": "Forest"},
            ],
        )
        self.assertEqual(ranked[0], "Forest")

    def test_anchor_ranking_prefers_qualified_entity_when_modifier_is_requested(self):
        ranked = query_engine._rank_anchor_entities(
            terms=["green", "forest"],
            phrases=["green forest"],
            candidates=[
                {"entity_name": "Forest"},
                {"entity_name": "Green Forest"},
            ],
        )
        self.assertEqual(ranked[0], "Green Forest")

    def test_what_is_constitution_prefers_clean_anchor_over_long_noisy_node(self):
        app_server.extract_text_from_pdf = lambda _: "The Constitution contains fundamental rights."
        app_server.extract_triples_groq = lambda chunk: [
            {
                "subject": "various minority communities also expressed the need for the Constitution to",
                "relation": "INCLUDES",
                "object": "rights that would protect their groups",
            },
            {
                "subject": "Indian Constitution",
                "relation": "INCLUDES",
                "object": "Fundamental Rights",
            },
            {
                "subject": "Constituent Assembly",
                "relation": "DRAFTED",
                "object": "Indian Constitution",
            },
        ]
        self.client.post("/upload_pdf", files={"file": ("constitution.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "what is constitution"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("Indian Constitution includes Fundamental Rights", ask.json()["answer"])
        self.assertNotIn("various minority communities", ask.json()["answer"].lower())

    def test_who_is_father_of_constitution_returns_ambedkar_fact(self):
        app_server.extract_text_from_pdf = lambda _: "Baba Saheb Dr Ambedkar is known as Father of the Indian Constitution."
        app_server.extract_triples_groq = lambda chunk: [
            {
                "subject": "Baba Saheb Dr Ambedkar",
                "relation": "KNOWN_AS",
                "object": "Father of the Indian Constitution",
            },
            {
                "subject": "Indian Constitution",
                "relation": "INCLUDES",
                "object": "Fundamental Rights",
            },
        ]
        self.client.post("/upload_pdf", files={"file": ("constitution.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "who is the father of constitution"})
        self.assertEqual(ask.status_code, 200)
        self.assertIn("Baba Saheb Dr Ambedkar", ask.json()["answer"])
        self.assertIn("Father of the Indian Constitution", ask.json()["answer"])

    def test_when_question_uses_temporal_relation_and_fuzzy_entity_match(self):
        app_server.extract_text_from_pdf = lambda _: "The Indian National Movement began in 1885. Indian National Congress made demand for Constituent Assembly."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian National Movement", "relation": "BEGAN_IN", "object": "1885"},
            {"subject": "Indian National Congress", "relation": "MADE_DEMAND", "object": "Constituent Assembly"},
        ]
        self.client.post("/upload_pdf", files={"file": ("movement.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "when did indian national moment start"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(ask.json()["answer"], "Indian National Movement began in 1885.")

    def test_plain_topic_phrase_with_relation_like_noun_is_treated_as_entity_lookup(self):
        app_server.extract_text_from_pdf = lambda _: "The Indian Constitution features Separation of Powers. Separation Between Religion defines Secularism."
        app_server.extract_triples_groq = lambda chunk: [
            {"subject": "Indian Constitution", "relation": "FEATURES", "object": "Separation of Powers"},
            {"subject": "Separation Between Religion", "relation": "DEFINES", "object": "Secularism"},
        ]
        self.client.post("/upload_pdf", files={"file": ("constitution.pdf", b"%PDF", "application/pdf")})
        ask = self.client.post("/ask", json={"question": "Separation of Powers"})
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(ask.json()["answer"], "Indian Constitution features Separation of Powers.")

    def test_sentence_formatter_renders_unknown_relations_in_readable_lowercase(self):
        self.assertEqual(
            query_engine._sentence_from_relation(
                "Indian Constitution",
                "FEATURES",
                "Separation of Powers",
                "",
                [],
            ),
            "Indian Constitution features Separation of Powers.",
        )


if __name__ == "__main__":
    unittest.main()

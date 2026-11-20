from unittest.mock import Mock

import mongomock
from qdrant_client import QdrantClient

from backend.agent.graph import build_graph
from backend.graph.neo4j import GraphStore
from backend.ingestion.indexer import index_document
from backend.ingestion.loader import Document
from backend.memory.history import ChatHistory
from backend.retrieval.vector import VectorStore


def test_ingestion_and_langgraph_use_separate_stores(fake_gemini):
    vectors = VectorStore(QdrantClient(":memory:"), "workflow")
    driver = Mock()
    driver.execute_query.return_value = ([], None, None)
    graph = GraphStore(driver)
    history = ChatHistory(mongomock.MongoClient())
    history.initialize()
    session_id = history.create_session("alice")
    embed = lambda texts: [[1.0, 0.0] for text in texts]
    document = Document("Handbook", "handbook.md", "md", [(3, "## Leave\n20 days leave")])
    index_document(document, vectors, graph, embed)
    assert driver.execute_query.call_args.kwargs["chunks"][0]["section"] == "Leave"
    workflow = build_graph(vectors, graph, history, embed)
    result = workflow.invoke({"user_id": "alice", "roles": ["employee"],
                              "session_id": session_id, "question": "Leave policy?", "mode": "hybrid"})
    assert "20 days leave [1]" in result["response"]["answer"]
    assert result["response"]["citations"][0]["page"] == 3
    assert history.recent("alice", session_id)[0]["question"] == "Leave policy?"
    assert driver.execute_query.call_args.kwargs["roles"] == ["public", "employee"]
    follow_up = workflow.invoke({"user_id": "alice", "roles": ["employee"],
                                "session_id": session_id, "question": "What about California?", "mode": "vector"})
    assert "Leave policy?" in follow_up["query"]
    vectors.close()
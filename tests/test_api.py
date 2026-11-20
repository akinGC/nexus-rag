from unittest.mock import Mock

import mongomock
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from backend.agent.graph import build_graph
from backend.app import app
from backend.dependencies import services
from backend.graph.neo4j import GraphStore
from backend.ingestion.indexer import index_document
from backend.ingestion.loader import Document
from backend.memory.history import ChatHistory
from backend.retrieval.vector import VectorStore


def test_chat_api_creates_and_restores_mongodb_conversation(fake_gemini):
    vectors = VectorStore(QdrantClient(":memory:"), "api")
    driver = Mock()
    driver.execute_query.return_value = ([], None, None)
    graph = GraphStore(driver)
    history = ChatHistory(mongomock.MongoClient())
    history.initialize()
    embed = lambda texts: [[1.0, 0.0] for text in texts]
    index_document(Document("Handbook", "handbook.md", "md", [(1, "20 vacation days.")]),
                   vectors, graph, embed)
    stores = {"vectors": vectors, "graph": graph, "history": history,
              "workflow": build_graph(vectors, graph, history, embed)}
    app.dependency_overrides[services] = lambda: stores
    client = TestClient(app)
    try:
        response = client.post("/api/chat", json={"question": "Vacation?", "mode": "hybrid"})
        assert response.status_code == 200, response.text
        session_id = response.json()["session_id"]
        assert "20 vacation days" in response.json()["answer"]
        assert response.json()["provider"] == "gemini"
        assert response.json()["confidence"] == 1.0
        assert response.json()["trace"]["verification"]["supported"] is True
        assert client.get(f"/api/conversations/{session_id}").json()["turns"][0]["question"] == "Vacation?"
        assert client.get(f"/api/conversations/{session_id}", headers={"X-Demo-User": "executive"}).status_code == 404
        followup = client.post("/api/chat", json={"question": "What about California?",
                                                   "session_id": session_id, "mode": "vector"})
        assert followup.status_code == 200
        assert "Vacation?" in followup.json()["trace"]["rewritten_query"]
        assert len(client.get(f"/api/conversations/{session_id}").json()["turns"]) == 2
        assert client.get("/api/documents").json()["documents"][0]["title"] == "Handbook"
    finally:
        app.dependency_overrides.pop(services, None)
        client.close()
        vectors.close()
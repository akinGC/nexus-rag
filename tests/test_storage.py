import mongomock
from qdrant_client import QdrantClient

from backend.memory.history import ChatHistory
from backend.retrieval.vector import VectorStore


def test_qdrant_filters_permissions_and_keeps_citation_metadata():
    store = VectorStore(QdrantClient(":memory:"), "test_chunks")
    chunks = [
        {"chunk_id": "public", "document_id": "handbook", "text": "Leave policy",
         "allowed_roles": ["public"], "section": "Leave", "page": 3},
        {"chunk_id": "private", "document_id": "payroll", "text": "CEO pay",
         "allowed_roles": ["executive"], "section": "Pay", "page": 1},
    ]
    store.index(chunks, [[1.0, 0.0], [1.0, 0.0]])
    results = store.search([1.0, 0.0], ["employee"])
    assert [item["chunk_id"] for item in results] == ["public"]
    assert results[0]["page"] == 3
    assert store.chunks(["employee"], "payroll") == []
    assert len(store.search([1.0, 0.0], ["executive"])) == 2
    store.delete_document("handbook", ["employee"])
    assert store.chunks(["employee"]) == []
    store.close()


def test_mongodb_history_is_persistent_and_user_scoped():
    client = mongomock.MongoClient()
    history = ChatHistory(client)
    history.initialize()
    session_id = history.create_session("alice")
    history.append("alice", session_id, "Leave?", {"answer": "20 days"})
    restored = ChatHistory(client)
    assert restored.recent("alice", session_id)[0]["response"]["answer"] == "20 days"
    assert restored.sessions("bob") == []
    try:
        restored.recent("bob", session_id)
    except KeyError:
        pass
    else:
        raise AssertionError("Another user accessed Alice's conversation")
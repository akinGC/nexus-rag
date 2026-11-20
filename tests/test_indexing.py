import logging
from unittest.mock import Mock

from backend.ingestion.embedder import EmbeddingUnavailable
from backend.ingestion.indexer import index_directory, index_document
from backend.ingestion.loader import Document


def test_index_directory_skips_unchanged_pdfs_and_reports_failures(tmp_path, monkeypatch):
    (tmp_path / "policy.pdf").write_bytes(b"policy")
    (tmp_path / "broken.pdf").write_bytes(b"broken")
    (tmp_path / "ignored.txt").write_text("ignored")
    vectors = Mock()
    graph = Mock()
    stored = []
    vectors.chunks.side_effect = lambda roles: stored.copy()
    vectors.index.side_effect = lambda chunks, embeddings: stored.extend(chunks)

    def load(path, roles):
        if path.name == "broken.pdf":
            raise ValueError("Unreadable PDF")
        return Document("Policy", path.name, "pdf", [(1, "20 leave days")], roles)

    monkeypatch.setattr("backend.ingestion.indexer.load_document", load)
    embed = lambda texts: [[1.0, 0.0] for text in texts]
    first = index_directory(tmp_path, vectors, graph, embed)
    assert [item["source"] for item in first["indexed"]] == ["policy.pdf"]
    assert [item["source"] for item in first["failed"]] == ["broken.pdf"]
    second = index_directory(tmp_path, vectors, graph, embed)
    assert second["indexed"] == []
    assert second["skipped"] == [{"source": "policy.pdf"}]
    assert vectors.index.call_count == 1
    (tmp_path / "policy.pdf").write_bytes(b"updated policy")
    third = index_directory(tmp_path, vectors, graph, embed)
    assert len(third["indexed"]) == 1
    graph.delete_document.assert_called_once()
    vectors.delete_document.assert_called_once()


def test_document_embeddings_and_qdrant_writes_are_batched(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    chunks = [{"chunk_id": f"doc:{index}", "text": "policy"} for index in range(65)]
    monkeypatch.setattr("backend.ingestion.indexer.chunk_document", lambda document: chunks)
    vectors = Mock()
    vectors.client.collection_exists.return_value = True
    graph = Mock()
    batch_sizes = []
    def embed(texts):
        batch_sizes.append(len(texts))
        return [[1.0, 0.0] for _ in texts]
    document = Document("Policy", "policy.pdf", "pdf", [(1, "policy")], document_id="doc")
    index_document(document, vectors, graph, embed)
    assert batch_sizes == [32, 32, 1]
    assert [len(call.args[0]) for call in vectors.index.call_args_list] == [32, 32, 1]
    graph.index.assert_called_once()
    assert "embedding batch 1/3" in caplog.text
    assert "stored vector batch 3/3" in caplog.text


def test_quota_exhaustion_stops_remaining_pdfs(tmp_path, monkeypatch):
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr("backend.ingestion.indexer.load_document", lambda path, roles: Document(
        path.stem, path.name, "pdf", [(1, "policy")], roles,
    ))
    vectors = Mock()
    vectors.chunks.return_value = []
    graph = Mock()
    def exhausted(*args):
        raise EmbeddingUnavailable("wait 60 seconds, then click Index PDFs again", quota_exhausted=True, retry_after=60)
    report = index_directory(tmp_path, vectors, graph, exhausted)
    assert [item["source"] for item in report["failed"]] == ["a.pdf", "b.pdf", "c.pdf"]
    assert "Not attempted" in report["failed"][1]["error"]
    assert "Index PDFs again" in report["failed"][0]["error"]
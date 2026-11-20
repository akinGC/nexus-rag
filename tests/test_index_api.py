import asyncio
from unittest.mock import Mock

from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from backend.app import app
from backend.dependencies import services
from backend.routes.upload import index_documents, index_lock


def test_index_endpoint_uses_server_folder_and_has_no_upload(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.routes.upload.settings.data_dir", tmp_path)
    report = {"indexed": [{"source": "policy.pdf", "chunk_count": 1}], "skipped": [], "failed": []}
    index = Mock(return_value=report)
    monkeypatch.setattr("backend.routes.upload.index_directory", index)
    stores = {"vectors": Mock(), "graph": Mock()}
    app.dependency_overrides[services] = lambda: stores
    client = TestClient(app)
    try:
        response = client.post("/api/index-documents")
        assert response.status_code == 202
        receipt = response.json()
        assert receipt["status"] == "running"
        assert receipt["message"] == "PDF indexing started in the background."
        index.assert_called_once_with(tmp_path / "documents", stores["vectors"], stores["graph"])
        status = client.get(f"/api/index-documents/{receipt['job_id']}")
        assert status.status_code == 200
        assert status.json()["status"] == "complete"
        assert status.json()["report"] == report
        assert client.get(f"/api/index-documents/{receipt['job_id']}",
                  headers={"X-Demo-User": "executive"}).status_code == 404
        assert "/api/upload" not in app.openapi()["paths"]
        index_lock.acquire()
        try:
            assert client.post("/api/index-documents").status_code == 409
        finally:
            index_lock.release()
    finally:
        app.dependency_overrides.pop(services, None)
        client.close()


def test_post_handler_queues_background_work_and_returns_immediately(monkeypatch, tmp_path, caplog):
    index = Mock(return_value={"indexed": [], "skipped": [], "failed": []})
    monkeypatch.setattr("backend.routes.upload.index_directory", index)
    monkeypatch.setattr("backend.routes.upload.settings.data_dir", tmp_path)
    caplog.set_level("INFO")
    background = BackgroundTasks()
    stores = {"vectors": Mock(), "graph": Mock()}
    try:
        receipt = index_documents(background, {"user_id": "demo:employee"}, stores)
        assert receipt["status"] == "running"
        index.assert_not_called()
        asyncio.run(background())
        index.assert_called_once_with(tmp_path / "documents", stores["vectors"], stores["graph"])
        assert "Accepted PDF indexing job" in caplog.text
        assert "PDF indexing job" in caplog.text and "complete" in caplog.text
    finally:
        if index_lock.locked():
            index_lock.release()
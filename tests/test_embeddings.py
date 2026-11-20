from unittest.mock import Mock, patch

import pytest

from backend.ingestion import embedder
from backend.ingestion.embedder import embed_query, embed_texts


def test_gemini_embedding_tasks_and_batching():
    model = Mock()
    model.embed_documents.side_effect = lambda texts, **kwargs: [[1.0] for text in texts]
    model.embed_query.return_value = [2.0]
    with patch("backend.ingestion.embedder.embedding_model", return_value=model), \
        patch("backend.ingestion.embedder.time.sleep") as sleep:
        embedder._next_request_at = 0
        assert embed_texts([]) == []
        assert len(embed_texts(["policy"] * 33)) == 33
        assert model.embed_documents.call_count == 2
        assert model.embed_documents.call_args.kwargs["task_type"] == "RETRIEVAL_DOCUMENT"
        assert embed_query("policy?") == [2.0]
        assert model.embed_query.call_args.kwargs["task_type"] == "RETRIEVAL_QUERY"
        assert sleep.call_args_list


def test_embedding_retries_quota_using_server_retry_delay(monkeypatch):
    model = Mock()
    model.embed_query.side_effect = [RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 7s."), [2.0]]
    monkeypatch.setattr(embedder, "_next_request_at", 0)
    with patch("backend.ingestion.embedder.embedding_model", return_value=model), \
        patch("backend.ingestion.embedder.time.sleep") as sleep:
        assert embed_query("policy?") == [2.0]
        assert model.embed_query.call_count == 2
        assert sleep.call_args.args[0] >= 6.9


def test_persistent_quota_error_is_reported_for_indexing_retry(monkeypatch):
    model = Mock()
    model.embed_documents.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 5s.")
    monkeypatch.setattr(embedder, "_next_request_at", 0)
    with patch("backend.ingestion.embedder.embedding_model", return_value=model), \
        patch("backend.ingestion.embedder.time.sleep"):
        with pytest.raises(embedder.EmbeddingUnavailable, match="click Index PDFs again") as error:
            embed_texts(["policy"])
    assert error.value.quota_exhausted is True
    assert model.embed_documents.call_count == 2
import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from backend.retrieval.reranker import rerank


def test_reranker_uses_worker_scores():
    chunks = [{"chunk_id": "low", "text": "router"}, {"chunk_id": "high", "text": "leave"}]
    with patch("backend.retrieval.reranker.subprocess.run", return_value=Mock(returncode=0, stdout=json.dumps([-2, 4]))) as worker:
        assert rerank("leave", chunks, 1)[0]["chunk_id"] == "high"
        assert worker.call_args.kwargs["env"]["OMP_NUM_THREADS"] == "1"


def test_reranker_native_crash_is_a_python_error():
    with patch("backend.retrieval.reranker.subprocess.run", return_value=Mock(returncode=-11)):
        with pytest.raises(ValueError, match="worker failed"):
            rerank("leave", [{"chunk_id": "first", "text": "leave"}])


def test_reranker_timeout_is_bounded():
    with patch("backend.retrieval.reranker.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 240)):
        with pytest.raises(ValueError, match="timed out"):
            rerank("leave", [{"chunk_id": "first", "text": "leave"}])
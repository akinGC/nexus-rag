from functools import lru_cache
import re
from threading import Lock
import time

from backend.config import settings

class EmbeddingUnavailable(RuntimeError):
    def __init__(self, message: str, *, quota_exhausted: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.quota_exhausted = quota_exhausted
        self.retry_after = retry_after


_request_lock = Lock()
_next_request_at = 0.0


def _is_quota_error(error: Exception) -> bool:
    message = str(error).upper()
    return "RESOURCE_EXHAUSTED" in message or "429" in message


def _retry_delay(error: Exception) -> float:
    match = re.search(r"retry\s+in\s+([\d.]+)\s*s", str(error), re.IGNORECASE)
    return float(match.group(1)) if match else 60.0


def _paced_embedding_request(request):
    global _next_request_at
    with _request_lock:
        for attempt in range(2):
            wait = max(0.0, _next_request_at - time.monotonic())
            if wait:
                time.sleep(wait)
            try:
                result = request()
            except Exception as error:
                if _is_quota_error(error) and attempt == 0:
                    _next_request_at = time.monotonic() + _retry_delay(error)
                    continue
                _next_request_at = time.monotonic() + settings.embedding_min_interval_seconds
                raise
            _next_request_at = time.monotonic() + settings.embedding_min_interval_seconds
            return result
    raise RuntimeError("Embedding retry loop ended unexpectedly")


@lru_cache(maxsize=1)
def embedding_model():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = []
    try:
        batch_size = min(100, max(1, settings.embedding_batch_size))
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vectors.extend(_paced_embedding_request(lambda: embedding_model().embed_documents(
                batch, task_type="RETRIEVAL_DOCUMENT"
            )))
    except Exception as error:
        quota_exhausted = _is_quota_error(error)
        retry_after = _retry_delay(error) if quota_exhausted else None
        message = (
            f"Gemini embedding quota is still exhausted after retrying. Wait at least {retry_after:g} seconds, "
            "then click Index PDFs again. Completed PDFs are skipped on retry."
            if quota_exhausted else "Gemini embedding request failed; check credentials and quota."
        )
        raise EmbeddingUnavailable(message, quota_exhausted=quota_exhausted, retry_after=retry_after) from error
    return vectors


def embed_query(text: str) -> list[float]:
    try:
        return _paced_embedding_request(lambda: embedding_model().embed_query(text, task_type="RETRIEVAL_QUERY"))
    except Exception as error:
        quota_exhausted = _is_quota_error(error)
        retry_after = _retry_delay(error) if quota_exhausted else None
        message = (
            f"Gemini embedding quota is exhausted. Wait at least {retry_after:g} seconds and try again."
            if quota_exhausted else "Gemini query embedding failed; check credentials and quota."
        )
        raise EmbeddingUnavailable(message, quota_exhausted=quota_exhausted, retry_after=retry_after) from error
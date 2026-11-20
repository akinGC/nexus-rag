from datetime import datetime, timezone
from hashlib import sha256
import logging
from pathlib import Path

from backend.ingestion.chunker import chunk_document
from backend.ingestion.embedder import EmbeddingUnavailable, embed_texts
from backend.ingestion.loader import Document, load_document
from backend.config import settings

logger = logging.getLogger(__name__)


def index_document(document: Document, vectors, graph, embed=embed_texts) -> dict:
	chunks = chunk_document(document)
	if not chunks:
		raise ValueError("Document has no indexable text")
	metadata = {
		"document_id": document.document_id, "title": document.title,
		"source": document.source, "source_type": document.source_type,
		"department": document.department, "allowed_roles": document.allowed_roles,
		"created_at": datetime.now(timezone.utc).isoformat(), "chunk_count": len(chunks),
	}
	for chunk in chunks:
		chunk["created_at"] = metadata["created_at"]
	try:
		batch_size = min(100, max(1, settings.embedding_batch_size))
		batch_count = (len(chunks) + batch_size - 1) // batch_size
		for batch_index, start in enumerate(range(0, len(chunks), batch_size), 1):
			batch = chunks[start:start + batch_size]
			logger.info("Indexing %s: embedding batch %d/%d (%d chunks)",
				document.source, batch_index, batch_count, len(batch))
			embeddings = embed([chunk["text"] for chunk in batch])
			vectors.index(batch, embeddings)
			logger.info("Indexing %s: stored vector batch %d/%d", document.source, batch_index, batch_count)
		graph.index(metadata, chunks)
		logger.info("Indexed %s: %d chunks", document.source, len(chunks))
	except Exception:
		try:
			if vectors.client.collection_exists(vectors.collection):
				vectors.delete_document(document.document_id, document.allowed_roles)
		except Exception:
			logger.exception("Could not clean up partial Qdrant index for %s", document.source)
		try:
			graph.delete_document(document.document_id, document.allowed_roles)
		except Exception:
			logger.exception("Could not clean up partial Neo4j index for %s", document.source)
		raise
	return metadata


def index_directory(directory: Path, vectors, graph, embed=embed_texts) -> dict:
	if not directory.is_dir():
		raise ValueError("The server's data/documents directory does not exist")
	report = {"indexed": [], "skipped": [], "failed": []}
	roles = ["employee", "manager", "hr_admin", "executive"]
	existing = vectors.chunks(roles)
	pdf_paths = [path for path in sorted(directory.iterdir())
				 if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".pdf"]
	logger.info("Found %d PDF files to inspect in %s", len(pdf_paths), directory)
	for index, path in enumerate(pdf_paths):
		try:
			logger.info("Checking PDF %s (%d/%d)", path.name, index + 1, len(pdf_paths))
			digest = sha256(path.read_bytes()).hexdigest()
			document_id = sha256(f"{path.name}:{digest}:{settings.embedding_model}".encode()).hexdigest()
			matches = [chunk for chunk in existing if chunk["document_id"] == document_id]
			document = load_document(path, roles)
			document.document_id = document_id
			expected = chunk_document(document)
			if matches and {chunk["chunk_id"] for chunk in matches} == {chunk["chunk_id"] for chunk in expected}:
				report["skipped"].append({"source": path.name})
				logger.info("Skipping unchanged PDF %s", path.name)
				continue
			metadata = index_document(document, vectors, graph, embed)
			old_ids = {chunk["document_id"] for chunk in existing
					   if chunk["source"] == path.name and chunk["document_id"] != document_id}
			for old_id in old_ids:
				graph.delete_document(old_id, roles)
				vectors.delete_document(old_id, roles)
			report["indexed"].append(metadata)
		except EmbeddingUnavailable as error:
			logger.exception("PDF embedding failed for %s", path.name)
			report["failed"].append({"source": path.name, "error": str(error)})
			if error.quota_exhausted:
				for remaining_path in pdf_paths[index + 1:]:
					report["failed"].append({
						"source": remaining_path.name,
						"error": "Not attempted because the Gemini embedding quota is exhausted.",
					})
				break
		except Exception:
			logger.exception("PDF indexing failed for %s", path.name)
			report["failed"].append({"source": path.name,
				"error": "Indexing failed. Check server logs, Gemini quota, PDF text, and database connections."})
	return report
from datetime import datetime, timezone
import logging
from threading import Lock
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from backend.config import settings
from backend.dependencies import identity, services
from backend.ingestion.indexer import index_directory

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


index_lock = Lock()
jobs_lock = Lock()
index_jobs = {}


def run_index_job(job_id: str, user_id: str, directory, vectors, graph):
	logger.info("PDF indexing job %s started", job_id)
	try:
		report = index_directory(directory, vectors, graph)
		with jobs_lock:
			index_jobs[job_id].update(status="complete", report=report,
				completed_at=datetime.now(timezone.utc).isoformat())
		logger.info("PDF indexing job %s complete: indexed=%d skipped=%d failed=%d",
			job_id, len(report["indexed"]), len(report["skipped"]), len(report["failed"]))
	except Exception:
		logger.exception("PDF indexing job %s failed", job_id)
		with jobs_lock:
			index_jobs[job_id].update(status="failed",
				error="Indexing failed unexpectedly. Check the backend logs.",
				completed_at=datetime.now(timezone.utc).isoformat())
	finally:
		index_lock.release()


@router.post("/index-documents", status_code=202)
def index_documents(background_tasks: BackgroundTasks, user=Depends(identity), stores=Depends(services)):
	if not index_lock.acquire(blocking=False):
		raise HTTPException(409, "PDF indexing is already running")
	job_id = uuid4().hex
	job = {"job_id": job_id, "user_id": user["user_id"], "status": "running",
		   "started_at": datetime.now(timezone.utc).isoformat()}
	with jobs_lock:
		index_jobs[job_id] = job
		completed = [key for key, value in index_jobs.items() if value["status"] != "running"]
		for key in completed[:-49]:
			index_jobs.pop(key, None)
	logger.info("Accepted PDF indexing job %s", job_id)
	background_tasks.add_task(
		run_index_job, job_id, user["user_id"], settings.data_dir / "documents",
		stores["vectors"], stores["graph"],
	)
	return {"job_id": job_id, "status": "running", "message": "PDF indexing started in the background."}


@router.get("/index-documents/{job_id}")
def index_status(job_id: str, user=Depends(identity)):
	with jobs_lock:
		job = index_jobs.get(job_id)
		if job is None or job["user_id"] != user["user_id"]:
			raise HTTPException(404, "Indexing job not found")
		return {key: value for key, value in job.items() if key != "user_id"}


@router.get("/documents")
def documents(user=Depends(identity), stores=Depends(services)):
	grouped = {}
	for chunk in stores["vectors"].chunks(user["roles"]):
		document_id = chunk["document_id"]
		if document_id not in grouped:
			grouped[document_id] = {key: chunk.get(key) for key in (
				"document_id", "title", "source", "source_type", "department", "allowed_roles", "created_at"
			)}
			grouped[document_id]["chunk_count"] = 0
		grouped[document_id]["chunk_count"] += 1
	return {"documents": list(grouped.values())}


@router.get("/documents/{document_id}")
def document(document_id: str, user=Depends(identity), stores=Depends(services)):
	chunks = stores["vectors"].chunks(user["roles"], document_id)
	if not chunks:
		raise HTTPException(404, "Document not found")
	return {"document": {key: chunks[0].get(key) for key in (
		"document_id", "title", "source", "source_type", "department", "allowed_roles"
	)}, "chunks": chunks}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, user=Depends(identity), stores=Depends(services)):
	if "hr_admin" not in user["roles"]:
		raise HTTPException(403, "Only HR admins may remove documents")
	if not stores["vectors"].chunks(user["roles"], document_id):
		raise HTTPException(404, "Document not found")
	stores["graph"].delete_document(document_id, user["roles"])
	stores["vectors"].delete_document(document_id, user["roles"])
	return {"deleted": document_id}
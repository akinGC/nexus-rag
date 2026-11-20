import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pymongo.errors import PyMongoError
from qdrant_client.http.exceptions import UnexpectedResponse
import httpx

from backend.agent.graph import build_graph
from backend.agent.gemini import GeminiUnavailable
from backend.ingestion.embedder import EmbeddingUnavailable
from backend.config import ROOT, settings
from backend.graph.neo4j import GraphStore
from backend.memory.history import ChatHistory
from backend.retrieval.vector import VectorStore
from backend.routes.chat import router as chat_router
from backend.routes.upload import router as upload_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
	vectors = VectorStore()
	graph = GraphStore()
	history = ChatHistory()
	app.state.services = {"vectors": vectors, "graph": graph, "history": history,
						  "workflow": build_graph(vectors, graph, history)}
	try:
		yield
	finally:
		for store in (vectors, graph, history):
			store.close()


app = FastAPI(title="Nexus", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(upload_router)


async def storage_error(request: Request, error: Exception):
	logger.exception("Storage request failed", exc_info=error)
	return JSONResponse(status_code=503, content={
		"detail": "A storage service is unavailable. Check Qdrant, Neo4j and MongoDB connection settings."
	})


for exception_type in (PyMongoError, Neo4jError, ServiceUnavailable, UnexpectedResponse, httpx.HTTPError):
	app.add_exception_handler(exception_type, storage_error)


@app.exception_handler(GeminiUnavailable)
@app.exception_handler(EmbeddingUnavailable)
async def gemini_error(request: Request, error: GeminiUnavailable):
	return JSONResponse(status_code=502, content={"detail": str(error)})


@app.get("/api/health")
def health(request: Request):
	stores = request.app.state.services
	status = {}
	chunks = []
	try:
		chunks = stores["vectors"].chunks(["employee"])
		status["qdrant"] = "connected"
	except Exception:
		status["qdrant"] = "unavailable"
	try:
		stores["graph"].driver.verify_connectivity()
		status["neo4j"] = "connected"
	except Exception:
		status["neo4j"] = "unavailable"
	try:
		stores["history"].client.admin.command("ping")
		stores["history"].initialize()
		status["mongodb"] = "connected"
	except Exception:
		status["mongodb"] = "unavailable"
	return {
		"status": "ready" if all(value == "connected" for value in status.values()) else "degraded",
		"services": status, "mode": settings.mode, "provider": settings.model,
		"embedding_provider": settings.embedding_model, "reranker": settings.reranker_model,
		"documents": len({chunk["document_id"] for chunk in chunks}),
		"chunks": len(chunks),
	}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")


import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


class Settings:
	data_dir = Path(os.getenv("NEXUS_DATA_DIR", str(ROOT / "data")))
	mode = os.getenv("NEXUS_MODE", "local")
	provider = "gemini"
	embedding_provider = "google"
	embedding_model = "models/gemini-embedding-001"
	reranker_model = os.getenv("NEXUS_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
	model = os.getenv("NEXUS_LLM_MODEL", "gemini-3.8-flash")
	embedding_batch_size = int(os.getenv("NEXUS_EMBEDDING_BATCH_SIZE", "32"))
	embedding_min_interval_seconds = float(os.getenv("NEXUS_EMBEDDING_MIN_INTERVAL_SECONDS", "1.1"))
	max_upload_bytes = 20 * 1024 * 1024
	top_k = 8
	context_chars = 18000
	qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
	qdrant_api_key = os.getenv("QDRANT_API_KEY")
	qdrant_collection = os.getenv("QDRANT_COLLECTION", "nexus_gemini_chunks")
	neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
	neo4j_user = os.getenv("NEO4J_USER", "neo4j")
	neo4j_password = os.getenv("NEO4J_PASSWORD", "")
	neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
	mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
	mongodb_database = os.getenv("MONGODB_DATABASE", "nexus")


settings = Settings()
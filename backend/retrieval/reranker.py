import json
import os
import subprocess
import sys

from backend.config import ROOT, settings


def rerank(query: str, chunks: list[dict], limit: int = 8) -> list[dict]:
	if not chunks:
		return []
	if not settings.reranker_model:
		raise ValueError("Set NEXUS_RERANKER_MODEL to use hybrid_rerank")
	environment = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
				   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
	try:
		result = subprocess.run(
			[sys.executable, "-m", "backend.retrieval.rerank_worker"],
			input=json.dumps({"model": settings.reranker_model, "query": query,
							 "texts": [chunk["text"] for chunk in chunks]}),
			text=True, capture_output=True, timeout=240, env=environment, cwd=ROOT,
		)
		if result.returncode != 0:
			raise ValueError("Reranking worker failed; check model availability or use Hybrid retrieval")
		scores = json.loads(result.stdout)
	except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
		raise ValueError("Reranking timed out or returned invalid output; try Hybrid retrieval") from error
	ranked = [{**chunk, "rerank_score": float(score)}
			  for chunk, score in zip(chunks, scores, strict=True)]
	return sorted(ranked, key=lambda chunk: chunk["rerank_score"], reverse=True)[:limit]
import re
from difflib import SequenceMatcher

from backend.config import settings
from backend.agent.gemini import assess_claims, generate_claims, verify_claims
from backend.ingestion.embedder import embed_query, embed_texts
from backend.retrieval.hybrid import fuse, keyword_search
from backend.retrieval.reranker import rerank


class WorkflowNodes:
	def __init__(self, vectors, graph, history, embed):
		self.vectors = vectors
		self.graph = graph
		self.history = history
		self.embed = embed

	def load_history(self, state):
		return {"history": self.history.recent(state["user_id"], state["session_id"])}

	def rewrite(self, state):
		query = state["question"]
		normalized = " ".join(query.lower().strip().rstrip(".!?").split())
		explanation = normalized in {
			"explain", "explain again", "explain that", "explain it", "explain more",
			"explain in simple terms", "simplify", "simplify that", "simplify it",
			"give an example", "give me an example", "tell me more", "what does that mean",
		}
		typo = re.fullmatch(r"(?:please )?explain ([a-z]+)", normalized)
		if typo and SequenceMatcher(None, typo.group(1), "again").ratio() >= 0.8:
			explanation = True
		followup = normalized.startswith(("what about", "and ", "how about"))
		if explanation or followup:
			previous = next((turn for turn in reversed(state["history"])
				if turn.get("response", {}).get("citations")), None)
			if previous:
				topic = previous["response"].get("trace", {}).get("rewritten_query") or previous["question"]
				topic = topic.split("\nFollow-up:", 1)[0]
				request = "Explain this topic again in simpler terms using the evidence." if explanation else query
				query = topic + "\nFollow-up: " + request
		return {"query": query}

	def retrieve(self, state):
		vector = embed_query(state["query"]) if self.embed is embed_texts else self.embed([state["query"]])[0]
		semantic = self.vectors.search(vector, state["roles"])
		if state["mode"] == "vector":
			return {"retrieved": semantic}
		keyword = keyword_search(state["query"], self.vectors.chunks(state["roles"]))
		return {"retrieved": fuse(semantic, keyword)}

	def expand(self, state):
		if state["mode"] == "vector":
			return {"evidence": state["retrieved"][:settings.top_k]}
		neighbors = self.graph.related(
			[chunk["chunk_id"] for chunk in state["retrieved"][:3]], state["roles"]
		)
		evidence = list({chunk["chunk_id"]: chunk
						 for chunk in [*state["retrieved"], *neighbors]}.values())
		if state["mode"] == "hybrid_rerank":
			evidence = rerank(state["query"], evidence, settings.top_k)
		return {"evidence": evidence[:settings.top_k]}

	def answer(self, state):
		citations = []
		remaining = settings.context_chars
		for chunk in state["evidence"]:
			if remaining <= 0:
				break
			excerpt = chunk["text"][:remaining]
			citations.append({**chunk, "text": excerpt, "number": len(citations) + 1})
			remaining -= len(excerpt)
		draft = generate_claims(state["query"], citations) if citations else None
		draft_retried = False
		if citations and (draft is None or (not draft.answer.strip() and not draft.claims)):
			draft = generate_claims(state["query"], citations, retry=True)
			draft_retried = True
		response = {
			"session_id": state["session_id"],
			"answer": draft.answer if draft else "I couldn't generate an answer.",
			"citations": citations, "provider": "gemini", "confidence": 0.0,
			"trace": {
				"rewritten_query": state["query"], "queries": [state["query"]],
				"retrieved": state["retrieved"], "reranked": state["evidence"]
					if state["mode"] == "hybrid_rerank" else [],
				"graph_expansion": "same-section" if state["mode"] != "vector" else "disabled",
				"draft_retried": draft_retried,
			},
		}
		return {"response": response, "draft": draft}

	def verify(self, state):
		response = state["response"]
		draft = state.get("draft")
		response.setdefault("answer", draft.answer if draft else "I couldn't generate an answer.")
		claims = []
		verification = None
		if draft and response["citations"]:
			verification = verify_claims(state["query"], draft, response["citations"])
			claims = assess_claims(draft, verification, response["citations"])
		supported = [claim for claim in claims if claim["supported"]]
		complete = bool(verification and verification.question_fully_answered
					and not verification.missing_information)
		supported_for_answer = supported
		if supported and len(supported) < len(claims):
			response["answer"] += "\n\nEvidence review: some claims were not supported by the retrieved sources."
		if not complete and verification and verification.missing_information:
			response["answer"] += "\n\nEvidence review: requested details not found in the supplied evidence: " + "; ".join(verification.missing_information)
		model_percent = verification.evidence_support_percent if verification else 0
		local_percent = round(100 * len(supported) / len(claims)) if claims else 0
		response["confidence"] = model_percent / 100
		response["evidence_support_percent"] = model_percent
		response["confidence_method"] = "Gemini's evidence-support estimate; local claim/citation checks are reported separately; not calibrated"
		used = {number for claim in claims for number in claim["citation_numbers"] if number > 0}
		if used:
			response["citations"] = [item for item in response["citations"] if item["number"] in used]
		response["trace"]["claims"] = claims
		response["trace"]["verification"] = {
			"supported": bool(claims) and len(supported) == len(claims),
			"evidence_support_percent": model_percent,
			"locally_supported_claims_percent": local_percent,
			"question_fully_answered": complete,
			"missing_information": verification.missing_information if verification else ["No complete answer draft was generated"],
			"method": "Gemini answer-completeness and entailment review plus exact-quote and citation validation",
			"supported_claims": len(supported), "total_claims": len(claims),
			"unsupported_withheld": len(claims) - len(supported),
		}
		return {"response": response}

	def save_history(self, state):
		self.history.append(
			state["user_id"], state["session_id"], state["question"], state["response"]
		)
		return {}
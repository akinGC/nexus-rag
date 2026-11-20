from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer


def keyword_search(query: str, chunks: list[dict], limit: int = 20) -> list[dict]:
	if not chunks:
		return []
	vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w.-]+\b", ngram_range=(1, 2))
	try:
		matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
	except ValueError:
		return []
	scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
	ranking = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)
	return [{**chunks[index], "keyword_score": float(scores[index])}
			for index in ranking[:limit] if scores[index] > 0]


def fuse(*rankings: list[dict], limit: int = 20) -> list[dict]:
	scores = defaultdict(float)
	items = {}
	for ranking in rankings:
		for rank, item in enumerate(ranking, 1):
			chunk_id = item["chunk_id"]
			scores[chunk_id] += 1 / (60 + rank)
			items[chunk_id] = {**items.get(chunk_id, {}), **item}
	return [{**items[chunk_id], "score": scores[chunk_id]}
			for chunk_id in sorted(scores, key=scores.get, reverse=True)[:limit]]
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from backend.config import settings


class VectorStore:
	def __init__(self, client=None, collection: str | None = None):
		self.client = client if client is not None else QdrantClient(
			url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=15
		)
		self.collection = collection or settings.qdrant_collection

	def index(self, chunks: list[dict], vectors: list[list[float]]) -> None:
		if not chunks or len(chunks) != len(vectors):
			raise ValueError("Each chunk requires an embedding")
		dimension = len(vectors[0])
		if not dimension or any(len(vector) != dimension for vector in vectors):
			raise ValueError("Embedding dimensions must be consistent")
		if not self.client.collection_exists(self.collection):
			self.client.create_collection(
				self.collection,
				vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
			)
			for field in ("allowed_roles", "document_id"):
				self.client.create_payload_index(
					self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
				)
		else:
			configuration = self.client.get_collection(self.collection).config.params.vectors
			if not isinstance(configuration, models.VectorParams) or configuration.size != dimension:
				raise ValueError("Qdrant embedding dimension differs; select a new QDRANT_COLLECTION and reindex")
		for start in range(0, len(chunks), 64):
			self.client.upsert(
				self.collection,
				points=[models.PointStruct(
					id=str(uuid5(NAMESPACE_URL, chunk["chunk_id"])), vector=vector, payload=chunk
				) for chunk, vector in zip(chunks[start:start + 64], vectors[start:start + 64], strict=True)],
				wait=True,
			)

	@staticmethod
	def permission_filter(roles: list[str], document_id: str | None = None):
		conditions = [models.FieldCondition(
			key="allowed_roles", match=models.MatchAny(any=["public", *roles])
		)]
		if document_id:
			conditions.append(models.FieldCondition(
				key="document_id", match=models.MatchValue(value=document_id)
			))
		return models.Filter(must=conditions)

	def search(self, vector: list[float], roles: list[str], limit: int = 20) -> list[dict]:
		if not self.client.collection_exists(self.collection):
			return []
		result = self.client.query_points(
			self.collection, query=vector, query_filter=self.permission_filter(roles),
			limit=limit, with_payload=True,
		)
		return [{**point.payload, "score": point.score} for point in result.points]

	def chunks(self, roles: list[str], document_id: str | None = None) -> list[dict]:
		if not self.client.collection_exists(self.collection):
			return []
		chunks = []
		offset = None
		while True:
			points, offset = self.client.scroll(
				self.collection, scroll_filter=self.permission_filter(roles, document_id),
				limit=256, offset=offset, with_payload=True, with_vectors=False,
			)
			chunks.extend(point.payload for point in points if point.payload)
			if offset is None:
				return chunks

	def delete_document(self, document_id: str, roles: list[str]) -> None:
		self.client.delete(
			self.collection,
			points_selector=models.FilterSelector(filter=self.permission_filter(roles, document_id)),
			wait=True,
		)

	def close(self) -> None:
		self.client.close()
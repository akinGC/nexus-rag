from neo4j import GraphDatabase

from backend.config import settings


class GraphStore:
	def __init__(self, driver=None):
		self.driver = driver if driver is not None else GraphDatabase.driver(
			settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
			connection_timeout=10,
		)

	def index(self, document: dict, chunks: list[dict]) -> None:
		query = """
		MERGE (document:Document {document_id: $document.document_id})
		SET document += $document
		WITH document
		UNWIND $chunks AS item
		MERGE (chunk:Chunk {chunk_id: item.chunk_id})
		SET chunk += item
		MERGE (document)-[:HAS_CHUNK]->(chunk)
		MERGE (section:Section {document_id: item.document_id, name: item.section})
		MERGE (document)-[:HAS_SECTION]->(section)
		MERGE (section)-[:HAS_CHUNK]->(chunk)
		"""
		self.driver.execute_query(query, document=document, chunks=chunks, database_=settings.neo4j_database)

	def related(self, chunk_ids: list[str], roles: list[str], limit: int = 8) -> list[dict]:
		query = """
		MATCH (document:Document)-[:HAS_CHUNK]->(seed:Chunk)
		WHERE seed.chunk_id IN $chunk_ids
		  AND any(role IN document.allowed_roles WHERE role IN $roles)
		MATCH (section:Section)-[:HAS_CHUNK]->(seed)
		MATCH (document)-[:HAS_CHUNK]->(neighbor:Chunk)<-[:HAS_CHUNK]-(section)
		WHERE NOT neighbor.chunk_id IN $chunk_ids
		  AND any(role IN neighbor.allowed_roles WHERE role IN $roles)
		RETURN DISTINCT properties(neighbor) AS chunk
		ORDER BY chunk.chunk_id LIMIT $limit
		"""
		records, _, _ = self.driver.execute_query(
			query, chunk_ids=chunk_ids, roles=["public", *roles], limit=limit,
			database_=settings.neo4j_database,
		)
		return [dict(record["chunk"]) for record in records]

	def delete_document(self, document_id: str, roles: list[str]) -> None:
		self.driver.execute_query(
			"""
			MATCH (document:Document {document_id: $document_id})
			WHERE any(role IN document.allowed_roles WHERE role IN $roles)
			OPTIONAL MATCH (document)-[:HAS_CHUNK|HAS_SECTION]->(child)
			DETACH DELETE child, document
			""",
			document_id=document_id, roles=["public", *roles], database_=settings.neo4j_database,
		)

	def close(self) -> None:
		self.driver.close()
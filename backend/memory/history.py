from datetime import datetime, timezone
from uuid import uuid4

from pymongo import ASCENDING, MongoClient

from backend.config import settings


class ChatHistory:
    def __init__(self, client=None):
        self.client = client if client is not None else MongoClient(
            settings.mongodb_uri, serverSelectionTimeoutMS=5000
        )
        self.database = self.client[settings.mongodb_database]

    def initialize(self) -> None:
        self.database.conversations.create_index(
            [("user_id", ASCENDING), ("session_id", ASCENDING)], unique=True
        )
        self.database.messages.create_index(
            [("user_id", ASCENDING), ("session_id", ASCENDING), ("created_at", ASCENDING)]
        )

    def create_session(self, user_id: str) -> str:
        session_id = uuid4().hex
        self.database.conversations.insert_one({
            "user_id": user_id, "session_id": session_id,
            "created_at": datetime.now(timezone.utc),
        })
        return session_id

    def require_session(self, user_id: str, session_id: str) -> None:
        if self.database.conversations.find_one({"user_id": user_id, "session_id": session_id}) is None:
            raise KeyError("Conversation not found")

    def append(self, user_id: str, session_id: str, question: str, response: dict) -> None:
        self.require_session(user_id, session_id)
        self.database.messages.insert_one({
            "user_id": user_id, "session_id": session_id,
            "question": question, "response": response,
            "created_at": datetime.now(timezone.utc),
        })

    def recent(self, user_id: str, session_id: str, limit: int = 10) -> list[dict]:
        self.require_session(user_id, session_id)
        messages = list(self.database.messages.find(
            {"user_id": user_id, "session_id": session_id}, {"_id": 0}
        ).sort([("created_at", -1), ("_id", -1)]).limit(limit))
        return list(reversed(messages))

    def sessions(self, user_id: str) -> list[dict]:
        return list(self.database.conversations.find(
            {"user_id": user_id}, {"_id": 0}
        ).sort("created_at", -1).limit(100))

    def close(self) -> None:
        self.client.close()
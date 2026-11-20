from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.dependencies import identity, services

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
	question: str = Field(min_length=1, max_length=4000)
	session_id: str | None = None
	mode: Literal["vector", "hybrid", "hybrid_rerank"] = "hybrid"


@router.post("/chat")
def chat(body: ChatRequest, user=Depends(identity), stores=Depends(services)):
	if not body.question.strip():
		raise HTTPException(422, "Question cannot be blank")
	try:
		session_id = body.session_id or stores["history"].create_session(user["user_id"])
		result = stores["workflow"].invoke({**body.model_dump(), **user, "session_id": session_id})
		return result["response"]
	except KeyError as error:
		raise HTTPException(404, "Conversation not found") from error
	except ValueError as error:
		raise HTTPException(422, str(error)) from error


@router.get("/conversations")
def conversations(user=Depends(identity), stores=Depends(services)):
	return {"conversations": stores["history"].sessions(user["user_id"])}


@router.get("/conversations/{session_id}")
def conversation(session_id: str, user=Depends(identity), stores=Depends(services)):
	try:
		return {"session_id": session_id,
				"turns": stores["history"].recent(user["user_id"], session_id, limit=100)}
	except KeyError as error:
		raise HTTPException(404, "Conversation not found") from error


@router.get("/memory")
def memory(user=Depends(identity)):
	return {"memories": [], "storage": "mongodb", "kind": "chat_history"}
from typing import TypedDict


class AgentState(TypedDict, total=False):
	user_id: str
	roles: list[str]
	session_id: str
	question: str
	mode: str
	history: list[dict]
	query: str
	retrieved: list[dict]
	evidence: list[dict]
	response: dict
	draft: object
from fastapi import Header, HTTPException, Request

from backend.config import settings


def identity(x_demo_user: str = Header(default="employee")) -> dict:
    if settings.mode != "local":
        raise HTTPException(503, "Production authentication is not configured")
    if x_demo_user not in {"employee", "manager", "hr_admin", "executive"}:
        raise HTTPException(403, "Unknown demo identity")
    return {"user_id": f"demo:{x_demo_user}", "roles": [x_demo_user]}


def services(request: Request):
    return request.app.state.services
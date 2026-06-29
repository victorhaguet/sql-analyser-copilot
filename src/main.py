"""Convenience entrypoints and FastAPI application wiring."""

from __future__ import annotations

from typing import Any
from fastapi import FastAPI, Header, HTTPException # pylint: disable=unused-import

from tools.database import DatabaseError
from config import create_app_from_env
from models import (
    LoginRequest,
    QueryRequest,
    QueryResumeRequest,
    UserCreate,
    UserUpdate,
    HTTP_400_BAD_REQUEST,
)
from app_auth import (
    get_user_by_username,
    verify_password,
    get_user_by_sub,
    create_user,
    hash_password,
    list_users,
    update_user,
    delete_user,
)
from core import _select_requested_databases, _serialize_state, resume_question, start_question

app = create_app_from_env()

if app is None:
    raise RuntimeError("Failed to create FastAPI application.")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Health check endpoint

    Returns:
        dict[str, str]: Status of the FastAPI application
    """
    return {"status": "ok"}


@app.post("/auth/login")
def login(payload: LoginRequest) -> dict[str, Any]:
    """Login endpoint

    Args:
        payload (LoginRequest): Information of the login request

    Raises:
        HTTPException: Raise an error if credentials aren't valid
        HTTPException: Raise an error if the account is disabled

    Returns:
        dict[str, Any]: Information of the logged user (if loging worked)
    """
    user = get_user_by_username(payload.username)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return {
        "sub": user["sub"],
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "is_active": user["is_active"],
        "created_at": user["created_at"],
    }


@app.get("/auth/me")
def get_me(x_user_sub: str = Header(...)) -> dict[str, Any]:
    """Return info of the current authenticated user

    Args:
        x_user_sub (str, optional): Current authenticated user ID. Defaults to Header(...).

    Raises:
        HTTPException: Raise an error if the user can't be find in the user database using his/her ID
        HTTPException: Raise an error if the user account isn't active

    Returns:
        dict[str, Any]: Information of the authenticated user
    """
    user = get_user_by_sub(x_user_sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return {
        "sub": user["sub"],
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "is_active": user["is_active"],
        "created_at": user["created_at"],
    }


@app.post("/auth/users")
def register_user(payload: UserCreate, x_user_role: str = Header(...)) -> dict[str, Any]:
    """Create a new user

    Args:
        payload (UserCreate): Information of the user to create
        x_user_role (str, optional): ID of the authentificated user (when the request was send). Defaults to Header(...).

    Raises:
        HTTPException: Raise an error if the authenticated user isn't an admin
        HTTPException: Raise an error when the role of the new user is incorrect
        HTTPException: Raise an error when the creation of the user doesn't work

    Returns:
        dict[str, Any]: Information of the new user after the creation
    """
    if x_user_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if payload.role not in {"admin", "editor", "readonly"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        user = create_user(
            username=payload.username,
            password_hash=hash_password(payload.password),
            name=payload.name,
            role=payload.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "sub": user["sub"],
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "is_active": user["is_active"],
        "created_at": user["created_at"],
    }


@app.get("/auth/users")
def get_users(x_user_role: str = Header(...)) -> list[dict[str, Any]]:
    """List all the users

    Args:
        x_user_role (str, optional): ID of the authenticated user. Defaults to Header(...).

    Raises:
        HTTPException: Raise an error if the authenticated user isn't an admin

    Returns:
        list[dict[str, Any]]: List of the users information
    """
    if x_user_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    users = list_users()
    return [
        {
            "sub": u["sub"],
            "username": u["username"],
            "name": u["name"],
            "role": u["role"],
            "is_active": u["is_active"],
            "created_at": u["created_at"],
        }
        for u in users
    ]


@app.put("/auth/users/{sub}")
def update_user_endpoint(
    sub: str,
    payload: UserUpdate,
    x_user_role: str = Header(...),
) -> dict[str, Any]:
    """Update the user information (role or status)

    Args:
        sub (str): ID of the user to modify
        payload (UserUpdate): New information to set to the user to modify
        x_user_role (str, optional): Role of the authentificated user. Defaults to Header(...).

    Raises:
        HTTPException: Raise an error if the authentificated user isn't an admin
        HTTPException: Raise an error if the new role of the user to modify doesn't exist
        HTTPException: Raise an error if the user to modify doesn't exist

    Returns:
        dict[str, Any]: Information of the user after modification
    """
    if x_user_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if payload.role is not None and payload.role not in {"admin", "editor", "readonly"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    user = update_user(sub, name=payload.name, role=payload.role, is_active=payload.is_active)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "sub": user["sub"],
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "is_active": user["is_active"],
        "created_at": user["created_at"],
    }


@app.delete("/auth/users/{sub}")
def delete_user_endpoint(sub: str, x_user_role: str = Header(...)) -> dict[str, str]:
    """Delete a user account

    Args:
        sub (str): Id of the user account to delete
        x_user_role (str, optional): Role of the authentificated user. Defaults to Header(...).

    Raises:
        HTTPException: Raise an error if the authentificated user isn't an admin
        HTTPException: Raise an error if the user account to delete can't be find

    Returns:
        dict[str, str]: _description_
    """
    if x_user_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    deleted = delete_user(sub)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "ok"}


@app.post("/query")
def query(payload: QueryRequest, x_user_sub: str = Header(...), x_user_role: str = Header(...)) -> dict[str, Any]:
    """Query the graph

    Args:
        payload (QueryRequest): Input of the graph (question)
        x_user_sub (str, optional): ID of the authentificated user. Defaults to Header(...).
        x_user_role (str, optional): Role of the authentificated user. Defaults to Header(...).

    Raises:
        HTTPException: Raise an error if the authentificated user can't be find in the user database
        HTTPException: Raise an error if the authentificated user account is disabled
        HTTPException: Raise an error if no LLM was configured as sql generator before running the graph
        HTTPException: Raise an error if the graph didn't give an answer

    Returns:
        dict[str, Any]: State of the graph after running it
    """
    get_user_by_sub(x_user_sub)

    try:
        databases = _select_requested_databases(
            payload.selected_databases,
            app.state.databases,
        )
        result = start_question(
            question=payload.question,
            sql_generator_model=app.state.sql_generator_model,
            analyst_model=app.state.analyst_model,
            selector_model=app.state.selector_model,
            intent_model=app.state.intent_model,
            databases=databases,
            validator=app.state.validator,
            execution_limit=payload.execution_limit,
            include_trace=app.state.enable_trace,
            trace_log_dir=app.state.trace_log_dir,
            user_role=x_user_role,
            user_sub=x_user_sub,
            pending_approval_sessions=app.state.pending_approval_sessions,
        )
    except DatabaseError as exc:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return _serialize_state(result, question=payload.question)


@app.post("/query/confirm")
def confirm_query(
    payload: QueryResumeRequest,
    x_user_sub: str = Header(...),
) -> dict[str, Any]:
    """Resume a modification query after the user approves or rejects it."""

    get_user_by_sub(x_user_sub)

    try:
        result = resume_question(
            payload.thread_id,
            payload.decision,
            pending_approval_sessions=app.state.pending_approval_sessions,
            include_trace=app.state.enable_trace,
            trace_log_dir=app.state.trace_log_dir,
            user_sub=x_user_sub,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return _serialize_state(result, question=result.get("question", ""))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

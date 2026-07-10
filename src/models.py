"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

HTTP_400_BAD_REQUEST = 400
HTTP_503_SERVICE_UNAVAILABLE = 503


class QueryRequest(BaseModel):
    """Incoming payload for SQL copilot queries."""

    question: str = Field(min_length=1)
    execution_limit: int = Field(default=200, gt=0)
    selected_database: str | None = None


class QueryResumeRequest(BaseModel):
    """Incoming payload for resuming approval-gated SQL modifications."""

    thread_id: str = Field(min_length=1)
    decision: Literal["approve", "reject"]


class LoginRequest(BaseModel):
    """Login credentials."""

    username: str
    password: str


class UserCreate(BaseModel):
    """Payload for creating a new user."""

    username: str
    password: str
    name: str | None = None
    role: str = "readonly"


class UserUpdate(BaseModel):
    """Payload for updating a user."""

    name: str | None = None
    role: str | None = None
    is_active: bool | None = None


class UserResponse(BaseModel):
    """Public user representation (no password)."""

    sub: str
    username: str
    name: str | None = None
    role: str
    is_active: bool
    created_at: str


class QueryResultPayload(BaseModel):
    """Serializable query result payload."""

    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool


class QueryResponse(BaseModel):
    """HTTP response model for SQL copilot queries."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    intent: str | None = None
    intent_error: str | None = None
    schema_overview: str | None = None
    selected_database: str | None = None
    generated_sql: str | None = None
    validated_sql: str | None = None
    sql_validation_error: str | None = None
    authorization_error: str | None = None
    is_authorized: bool | None = None
    query_result: QueryResultPayload | None = None
    execution_error: str | None = None
    execution_requested: bool = False
    execution_confirmed: bool = False
    thread_id: str | None = None
    interrupt: dict[str, Any] | None = None
    analysis: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

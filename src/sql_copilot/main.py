"""Convenience entrypoints and FastAPI application wiring."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, cast
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import authentification functions
from sql_copilot.auth import (
    create_user,
    ensure_admin_exists,
    get_user_by_sub,
    get_user_by_username,
    hash_password,
    list_users,
    update_user,
    verify_password,
    delete_user
)

# Import graph functions
from sql_copilot.graph import SQLAgentTraceStep, build_sql_agent_graph, stream_sql_agent_execution
from sql_copilot.nodes.sql_generator import TextModel
from sql_copilot.state import SQLAgentState
from sql_copilot.tools.database import (
    DatabaseError,
    QueryResult,
    RegisteredDatabase,
    load_database_catalog_from_env,
)
from sql_copilot.tools.sql_safety import SQLSafetyValidator

from sql_copilot.llms.openai_compatible import OpenAICompatibleResponsesModel

load_dotenv()

# HTTP status codes for error handling in the API
HTTP_400_BAD_REQUEST = 400
HTTP_503_SERVICE_UNAVAILABLE = 503

# Trace Variables
TRACE_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
TRACE_HEADER_WIDTH = 80

# Create classes
class QueryRequest(BaseModel):
    """Incoming payload for SQL copilot queries."""

    question: str = Field(min_length=1)
    execution_limit: int = Field(default=200, gt=0)
    selected_databases: list[str] | None = None


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


def _get_user_from_header(x_user_sub: str) -> dict[str, Any]:
    """Get and validate user from header.

    Args:
        x_user_sub: User ID from header.

    Returns:
        User dictionary.

    Raises:
        HTTPException: If user not found or inactive.
    """
    user = get_user_by_sub(x_user_sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return user


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
    schema_overview: str | None = None
    selected_database: str | None = None
    generated_sql: str | None = None
    validated_sql: str | None = None
    sql_validation_error: str | None = None
    query_result: QueryResultPayload | None = None
    execution_error: str | None = None
    analysis: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _format_trace_header(title: str) -> str:
    """
    Return a centered trace section header.
    
    Args:
        title: The title to display in the header.

    Returns:
        A formatted header string with the title centered and padded by '=' characters.
    """
    return f" {title} ".center(TRACE_HEADER_WIDTH, "=")


def _format_trace_payload(payload: Any) -> str:
    """
    Serialize a trace payload into readable text.

    Args:
        payload: The payload to serialize.

    Returns:
        A string representation of the payload.
    """
    if payload is None:
        return "(none)"
    if isinstance(payload, str):
        return payload
    if isinstance(payload, QueryResult):
        summary = [
            f"Columns: {', '.join(payload.columns) if payload.columns else '(none)'}",
            f"Row Count: {payload.row_count}",
            f"Truncated: {payload.truncated}",
            "Rows:",
            json.dumps(payload.rows, indent=2, ensure_ascii=True),
        ]
        return "\n".join(summary)
    return json.dumps(payload, indent=2, ensure_ascii=True, default=str)


def _format_trace_step(step: SQLAgentTraceStep) -> str:
    """
    Render one normalized step into the persisted trace format.
    
    Args:
        step: The SQLAgentTraceStep dictionary containing node name, state update, and outcome label.

    Returns:
        A formatted string representing the step for trace logs.
    """

    node = step["node"]
    update = step["update"]
    metadata = update.get("metadata") or {}

    if node == "database_selector":
        selected_database = metadata.get("selected_database") 
        lines = [f"Node: {node}", f"Outcome: {step['outcome']}"]
        if selected_database: 
            lines.append(f"Selected Database: {selected_database}") 
        if metadata.get("candidate_databases"): 
            lines.append(
                "Candidate Databases: " 
                + ", ".join(cast(list[str], metadata["candidate_databases"]))
            )
        if metadata.get("database_selection_reason"): 
            lines.append(f"Reason: {metadata['database_selection_reason']}") 
        if update.get("execution_error"): 
            lines.append(f"Error: {update['execution_error']}") 
        body = "\n".join(lines)
        return f"{_format_trace_header('Ai Message')}\n{body}"


    if node == "sql_generator":
        body = "\n".join( 
            [
                f"Node: {node}",
                f"Outcome: {step['outcome']}",
                update.get("generated_sql", ""),
            ]
        ).strip()
        return f"{_format_trace_header('Ai Message')}\n{body}" 

    if node == "sql_validator":
        body = update.get("validated_sql") or update.get("sql_validation_error") or "(no output)"
        return ( 
            f"{_format_trace_header('Tool Message')}\n"
            f"Name: {node}\n\n"
            f"{body}"
        )


    if node == "sql_executor":
        parts = [f"Outcome: {step['outcome']}"]
        validated_sql = step["state"].get("validated_sql")
        if validated_sql: 
            parts.extend(["SQL:", str(validated_sql)]) 
        if update.get("query_result") is not None: 
            parts.append(_format_trace_payload(update["query_result"])) 
        elif update.get("execution_error"): 
            parts.append(f"Error: {update['execution_error']}")
        return (
            f"{_format_trace_header('Tool Message')}\n"
            f"Name: {node}\n\n"
            f"{chr(10).join(parts)}"
        )

    body = update.get("analysis") or _format_trace_payload(update)
    return f"{_format_trace_header('Ai Message')}\n{body}"


def _build_trace_log_content(question: str, trace: list[SQLAgentTraceStep]) -> str:
    """
    Render the full persisted trace file content for one run.
    
    Args:
        question: The original natural language question that was asked.
        trace: The list of SQLAgentTraceStep dictionaries representing the execution trace.

    Returns:
        A formatted string representing the entire trace log content for this run.
    """
    sections = [_format_trace_header("Human Message"), question]
    sections.extend(_format_trace_step(step) for step in trace)
    return "\n".join(sections) + "\n"


def _write_trace_log(
    question: str,
    trace: list[SQLAgentTraceStep],
    trace_log_dir: str | Path | None,
) -> Path:
    """
    Persist one run trace under the configured log directory.
    
    Args:
        question: The original natural language question that was asked.
        trace: The list of SQLAgentTraceStep dictionaries representing the execution trace.
        trace_log_dir: The directory where the trace log should be saved.

    Returns:
        The path to the saved trace log file.
    """
    log_dir = Path(trace_log_dir) if trace_log_dir is not None else TRACE_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"trace_{timestamp}_{uuid4().hex[:8]}.log"
    log_path.write_text(_build_trace_log_content(question, trace), encoding="utf-8")
    return log_path


def _run_question_with_trace(
    question: str,
    graph: Any,
    trace_log_dir: str | Path | None,
    write_log: bool = True,
) -> tuple[SQLAgentState, list[SQLAgentTraceStep], Path | None]:
    """Run the graph once, collecting the trace and optionally persisting it to disk."""
    trace = list(stream_sql_agent_execution(graph, {"question": question}))
    result: SQLAgentState = trace[-1]["state"] if trace else {"question": question}
    if write_log: # If write_log is True, persist the trace log and attach the path to the result metadata
        log_path = _write_trace_log(question, trace, trace_log_dir)
        metadata = dict(result.get("metadata") or {})
        metadata["trace_log_path"] = str(log_path)
        result["metadata"] = metadata
    else:
        log_path = None
    return result, trace, log_path


def _serialize_user(user: dict[str, Any]) -> UserResponse:
    """Convert user dictionary to UserResponse.

    Args:
        user: User dictionary from database.

    Returns:
        UserResponse instance.
    """
    return UserResponse(
        sub=user["sub"],
        username=user["username"],
        name=user["name"],
        role=user["role"],
        is_active=user["is_active"],
        created_at=user["created_at"],
    )


def _serialize_query_result(result: QueryResult | None) -> QueryResultPayload | None:
    """
    Convert the internal query result dataclass into an API payload.

    Args:
        result: The internal QueryResult instance to serialize.

    Returns:
        A QueryResultPayload instance suitable for API responses, or None if the input is None.
    """
    if result is None:
        return None
    return QueryResultPayload(
        columns=result.columns,
        rows=result.rows,
        row_count=result.row_count,
        truncated=result.truncated,
    )


def _serialize_state(state: SQLAgentState, question: str) -> QueryResponse:
    """
    Convert the internal graph state into the public API response model.

    Args:
        state: The internal state dictionary from the SQL copilot graph.
        question: The original natural language question.

    Returns:
        A QueryResponse instance suitable for API responses.
    """
    return QueryResponse(
        question=question,
        schema_overview=state.get("schema_overview"),
        selected_database=(state.get("metadata") or {}).get("selected_database"),
        generated_sql=state.get("generated_sql"),
        validated_sql=state.get("validated_sql"),
        sql_validation_error=state.get("sql_validation_error"),
        query_result=_serialize_query_result(state.get("query_result")),
        execution_error=state.get("execution_error"),
        analysis=state.get("analysis"),
        metadata=state.get("metadata") or {},
    )


def answer_question(
    question: str,
    sql_generator_model: TextModel,
    analyst_model: TextModel | None = None,
    selector_model: TextModel | None = None,
    databases: list[RegisteredDatabase] | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
    include_trace: bool = False,
    trace_log_dir: str | Path | None = None,
) -> SQLAgentState:
    """
    Run the SQL copilot graph.

    Args:
        question: The natural language question to answer.
        sql_generator_model: The language model to use for SQL generation.
        analyst_model: The language model to use for result analysis (optional).
        selector_model: The language model to use for database selection (optional).
        databases: The registered databases available for routing (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).
        include_trace: When true, attach a normalized per-step execution trace to `metadata["execution_trace"]`.
        trace_log_dir: Optional directory where per-run trace logs are persisted. Defaults to `<repo>/logs`.

    Returns:
        An instance of SQLAgentState representing the state of the SQL copilot graph after processing the question.
    """
    graph = build_sql_agent_graph(
        sql_generator_model=sql_generator_model,
        analyst_model=analyst_model,
        selector_model=selector_model,
        databases=databases,
        validator=validator,
        execution_limit=execution_limit,
    )
    if include_trace: 
        result, trace, _log_path = _run_question_with_trace(question, graph, trace_log_dir, write_log=True)
        metadata = dict(result.get("metadata") or {})
        metadata["execution_trace"] = trace
        result["metadata"] = metadata
        return result

    result = graph.invoke({"question": question})
    return result


def stream_question(
    question: str,
    sql_generator_model: TextModel,
    analyst_model: TextModel | None = None,
    selector_model: TextModel | None = None,
    databases: list[RegisteredDatabase] | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
) -> Iterator[SQLAgentTraceStep]:
    """
    Stream normalized execution steps for a single question.

    This is a graph-specific debugging helper over LangGraph's raw `stream()`
    output. It exposes node names, state deltas, and a compact outcome label.

    Args:
        question: The natural language question to answer.
        sql_generator_model: The language model to use for SQL generation.
        analyst_model: The language model to use for result analysis (optional).
        selector_model: The language model to use for database selection (optional).
        databases: The registered databases available for routing (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).

    Returns:
        An iterator of SQLAgentTraceStep dictionaries representing the execution trace of the graph.
    """
    graph = build_sql_agent_graph(
        sql_generator_model=sql_generator_model,
        analyst_model=analyst_model,
        selector_model=selector_model,
        databases=databases,
        validator=validator,
        execution_limit=execution_limit,
    )
    return stream_sql_agent_execution(graph, {"question": question})


def _select_requested_databases(
    requested_names: list[str] | None,
    configured_databases: list[RegisteredDatabase] | None,
) -> list[RegisteredDatabase] | None:
    """
    Resolve the database subset requested by the client.

    Args:
        requested_names: Optional list of database names selected by the client.
        configured_databases: The configured database catalog for the app.

    Returns:
        The filtered catalog, or the original configured catalog when no subset was requested.

    Raises:
        DatabaseError: If the requested selection is empty or contains unknown names.
    """
    if configured_databases is None or requested_names is None:
        return configured_databases
    if not requested_names:
        raise DatabaseError("At least one database must be selected.")

    configured_by_name = {database.name: database for database in configured_databases}
    selected_databases: list[RegisteredDatabase] = []
    unknown_names: list[str] = []
    for name in requested_names:
        database = configured_by_name.get(name)
        if database is None:
            unknown_names.append(name)
            continue
        selected_databases.append(database)

    if unknown_names:
        raise DatabaseError(
            "Unknown databases requested: " + ", ".join(sorted(set(unknown_names)))
        )
    if not selected_databases:
        raise DatabaseError("At least one database must be selected.")
    return selected_databases


def create_app(
    sql_generator_model: TextModel | None = None,
    analyst_model: TextModel | None = None,
    selector_model: TextModel | None = None,
    databases: list[RegisteredDatabase] | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
) -> FastAPI:
    """
    Build the FastAPI application around the SQL copilot core.

    The models are injected at startup time so the HTTP layer stays independent
    from the concrete LLM provider implementation.

    Args:
        sql_generator_model: The language model to use for SQL generation (optional).
        analyst_model: The language model to use for result analysis (optional).
        selector_model: The language model to use for database selection (optional).
        databases: The registered databases available for routing (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).

    Returns:
        A FastAPI application instance ready to serve SQL copilot requests.
    """
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed. Install project dependencies before creating the API app."
        )

    ensure_admin_exists()

    fastapi_app = FastAPI(title="SQL Analyser Copilot", version="0.1.0")
    fastapi_app.state.sql_generator_model = sql_generator_model
    fastapi_app.state.analyst_model = analyst_model
    fastapi_app.state.selector_model = selector_model
    fastapi_app.state.databases = databases
    fastapi_app.state.validator = validator
    fastapi_app.state.execution_limit = execution_limit

    @fastapi_app.get("/health")
    def healthcheck() -> dict[str, str]:
        """Simple liveness endpoint."""
        return {"status": "ok"}

    @fastapi_app.post("/auth/login")
    def login(payload: LoginRequest) -> UserResponse:
        """Identify a user and return their profile.

        Args:
            payload: User login credentials.

        Raises:
            HTTPException: If credentials are invalid or account is disabled.

        Returns:
            UserResponse: User information.
        """
        user = get_user_by_username(payload.username)
        if not user or not verify_password(payload.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        if not user["is_active"]:
            raise HTTPException(status_code=403, detail="Account is disabled")
        return _serialize_user(user)

    @fastapi_app.get("/auth/me")
    def get_me(x_user_sub: str = Header(...)) -> UserResponse:
        """Return the current authenticated profile.

        Args:
            x_user_sub: User ID from header.

        Raises:
            HTTPException: If user not found or inactive.

        Returns:
            UserResponse: User information.
        """
        user = _get_user_from_header(x_user_sub)
        return _serialize_user(user)

    @fastapi_app.post("/auth/users")
    def register_user(
        payload: UserCreate,
        x_user_role: str = Header(...),
    ) -> UserResponse:
        """Create a new user. Only admins can use this feature.

        Args:
            payload: Payload of the new user to create.
            x_user_role: Admin role from header.

        Raises:
            HTTPException: If not admin, invalid role, or user creation fails.

        Returns:
            UserResponse: Created user information.
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
        return _serialize_user(user)

    @fastapi_app.get("/auth/users")
    def get_users(
        x_user_role: str = Header(...),
    ) -> list[UserResponse]:
        """List all users.

        Args:
            x_user_role: Current user role from header.

        Raises:
            HTTPException: If not admin.

        Returns:
            List of all users.
        """
        if x_user_role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        users = list_users()
        return [_serialize_user(u) for u in users]

    @fastapi_app.put("/auth/users/{sub}")
    def update_user_endpoint(
        sub: str,
        payload: UserUpdate,
        x_user_role: str = Header(...),
    ) -> UserResponse:
        """Update a user role or status.

        Args:
            sub: ID of the user to update.
            payload: Information to update.
            x_user_role: Role of the current user from header.

        Raises:
            HTTPException: If not admin, invalid role, or user not found.

        Returns:
            UserResponse: Updated user information.
        """
        if x_user_role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        if payload.role is not None and payload.role not in {"admin", "editor", "readonly"}:
            raise HTTPException(status_code=400, detail="Invalid role")
        user = update_user(sub, name=payload.name, role=payload.role, is_active=payload.is_active)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return _serialize_user(user)

    @fastapi_app.delete("/auth/users/{sub}")
    def delete_user_endpoint(
        sub: str,
        x_user_role: str = Header(...),
    ) -> dict[str, str]:
        """Delete a user account.

        Args:
            sub: ID of the account to delete.
            x_user_role: Role of the current user from header.

        Raises:
            HTTPException: If not admin or user not found.

        Returns:
            Validation dictionary.
        """
        if x_user_role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        deleted = delete_user(sub)
        if not deleted:
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "ok"}

    @fastapi_app.post("/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        x_user_sub: str = Header(...),
    ) -> QueryResponse:
        """Send a request to the graph

        Args:
            payload (QueryRequest): Query request payload
            x_user_sub (str, optional): User that send the request. Defaults to Header(...).

        Raises:
            HTTPException: If the user doesn't exist, raise an error
            HTTPException: If the user isn't active, raise an error
            HTTPException: Raise an error if the model isn't configured
            HTTPException: Raise an error if the query didn't run correctly

        Returns:
            QueryResponse: _description_
        """
        # CHeck if the user cna send a request
        user = get_user_by_sub(x_user_sub)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        if not user["is_active"]:
            raise HTTPException(status_code=403, detail="Account is disabled")

        if fastapi_app.state.sql_generator_model is None:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "No SQL generator model is configured. "
                    "Instantiate the app with a model before serving requests."
                ),
            )

        try:
            databases = _select_requested_databases(
                payload.selected_databases,
                fastapi_app.state.databases,
            )
            result = answer_question(
                question=payload.question,
                sql_generator_model=fastapi_app.state.sql_generator_model,
                analyst_model=fastapi_app.state.analyst_model,
                selector_model=fastapi_app.state.selector_model,
                databases=databases,
                validator=fastapi_app.state.validator,
                execution_limit=payload.execution_limit,
            )
        except DatabaseError as exc:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        return _serialize_state(result, question=payload.question)

    return fastapi_app


def load_models_from_env() -> tuple[OpenAICompatibleResponsesModel | None, OpenAICompatibleResponsesModel | None]:
    """
    Create default SQL generator and analyst models from environment variables.
    
    Returns:
        A tuple containing the generator model and analyst model instances, or (None, None) if the API key is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None

    base_url: str | None = os.getenv("OPENAI_BASE_URL")
    generator_model_name: str | None = os.getenv("SQL_COPILOT_MODEL")
    analyst_model_name: str | None = os.getenv("SQL_COPILOT_ANALYST_MODEL") or generator_model_name

    return (
        OpenAICompatibleResponsesModel(
            model=generator_model_name, # type: ignore
            api_key=api_key, # type: ignore
            base_url=base_url,
        ),
        OpenAICompatibleResponsesModel(
            model=analyst_model_name, # type: ignore
            api_key=api_key,
            base_url=base_url,
        ),
    )


def create_app_from_env() -> FastAPI:
    """Create the FastAPI app using OpenAI-compatible environment variables when present."""
    sql_generator_model, analyst_model = load_models_from_env()
    return create_app(
        sql_generator_model=sql_generator_model,
        analyst_model=analyst_model,
        selector_model=sql_generator_model,
        databases=load_database_catalog_from_env(),
    )


app = create_app_from_env() if FastAPI is not None else None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("sql_copilot.main:app", host="127.0.0.1", port=8000, reload=True)

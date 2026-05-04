"""Convenience entrypoints and FastAPI application wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sql_copilot.graph import build_sql_agent_graph
from sql_copilot.llms import load_models_from_env
from sql_copilot.nodes.sql_generator import TextModel
from sql_copilot.tools.database import DatabaseError, QueryResult, SQLiteDatabase
from sql_copilot.tools.sql_safety import SQLSafetyValidator

load_dotenv()

# HTTP status codes for error handling in the API
HTTP_400_BAD_REQUEST = 400
HTTP_503_SERVICE_UNAVAILABLE = 503

class QueryRequest(BaseModel):
    """Incoming payload for SQL copilot queries."""

    question: str = Field(min_length=1)
    execution_limit: int = Field(default=200, gt=0)


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
    generated_sql: str | None = None
    validated_sql: str | None = None
    sql_validation_error: str | None = None
    query_result: QueryResultPayload | None = None
    execution_error: str | None = None
    analysis: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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


def _serialize_state(state: dict[str, Any], question: str) -> QueryResponse:
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
    database: SQLiteDatabase | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
) -> dict[str, Any]:
    """
    Run the SQL copilot graph.

    Args:
        question: The natural language question to answer.
        sql_generator_model: The language model to use for SQL generation.
        analyst_model: The language model to use for result analysis (optional).
        database: The SQLite database instance to use for query execution (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).

    Returns:
        A dictionary representing the state of the SQL copilot graph after processing the question.
    """
    graph = build_sql_agent_graph(
        sql_generator_model=sql_generator_model,
        analyst_model=analyst_model,
        database=database,
        validator=validator,
        execution_limit=execution_limit,
    )
    return graph.invoke({"question": question})


def create_app(
    sql_generator_model: TextModel | None = None,
    analyst_model: TextModel | None = None,
    database: SQLiteDatabase | None = None,
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
        database: The SQLite database instance to use for query execution (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).

    Returns:
        A FastAPI application instance ready to serve SQL copilot requests.
    """
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed. Install project dependencies before creating the API app."
        )

    fastapi_app = FastAPI(title="SQL Analyser Copilot", version="0.1.0")
    fastapi_app.state.sql_generator_model = sql_generator_model
    fastapi_app.state.analyst_model = analyst_model
    fastapi_app.state.database = database
    fastapi_app.state.validator = validator
    fastapi_app.state.execution_limit = execution_limit

    @fastapi_app.get("/health")
    def healthcheck() -> dict[str, str]:
        """Simple liveness endpoint."""
        return {"status": "ok"}

    @fastapi_app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        """Run the SQL copilot graph for a single natural-language question."""
        if fastapi_app.state.sql_generator_model is None:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "No SQL generator model is configured. "
                    "Instantiate the app with a model before serving requests."
                ),
            )

        try:
            result = answer_question(
                question=payload.question,
                sql_generator_model=fastapi_app.state.sql_generator_model,
                analyst_model=fastapi_app.state.analyst_model,
                database=fastapi_app.state.database,
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


def create_app_from_env() -> FastAPI:
    """Create the FastAPI app using OpenAI-compatible environment variables when present."""
    sql_generator_model, analyst_model = load_models_from_env()
    return create_app(
        sql_generator_model=sql_generator_model,
        analyst_model=analyst_model,
    )


app = create_app_from_env() if FastAPI is not None else None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("sql_copilot.main:app", host="127.0.0.1", port=8000, reload=True)

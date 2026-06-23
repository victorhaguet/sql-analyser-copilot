"""Core business logic for the SQL copilot."""

from __future__ import annotations

from typing import Any, Iterator, cast
from pathlib import Path

from graph import SQLAgentTraceStep, build_sql_agent_graph, stream_sql_agent_execution
from nodes.sql_generator import LLM
from state import SQLAgentState
from tools.database import DatabaseError, RegisteredDatabase, QueryResult
from tools.sql_safety import SQLSafetyValidator
from tracing import write_trace_log


def _select_requested_databases(
    requested_names: list[str] | None,
    configured_databases: list[RegisteredDatabase] | None,
) -> list[RegisteredDatabase] | None:
    """Resolve the database subset requested by the client.

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


def _serialize_user(user: dict[str, Any]) -> dict[str, Any]:
    """Convert user dictionary to UserResponse-like dict.

    Args:
        user: User dictionary from database.

    Returns:
        Dictionary with user information suitable for API responses.
    """
    return {
        "sub": user["sub"],
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "is_active": user["is_active"],
        "created_at": user["created_at"],
    }


def _serialize_query_result(result: QueryResult | None) -> dict[str, Any] | None:
    """Convert the internal query result dataclass into an API payload.

    Args:
        result: The internal QueryResult instance to serialize.

    Returns:
        A dictionary suitable for API responses, or None if the input is None.
    """
    if result is None:
        return None
    return {
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
    }


def _serialize_state(state: SQLAgentState, question: str) -> dict[str, Any]:
    """Convert the internal graph state into the public API response model.

    Args:
        state: The internal state dictionary from the SQL copilot graph.
        question: The original natural language question.

    Returns:
        A dictionary suitable for API responses.
    """
    return {
        "question": question,
        "schema_overview": state.get("schema_overview"),
        "selected_database": (state.get("metadata") or {}).get("selected_database"),
        "generated_sql": state.get("generated_sql"),
        "validated_sql": state.get("validated_sql"),
        "sql_validation_error": state.get("sql_validation_error"),
        "query_result": _serialize_query_result(state.get("query_result")),
        "execution_error": state.get("execution_error"),
        "analysis": state.get("analysis"),
        "metadata": state.get("metadata") or {},
    }


def answer_question(
    question: str,
    sql_generator_model: LLM,
    analyst_model: LLM | None = None,
    selector_model: LLM | None = None,
    databases: list[RegisteredDatabase] | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
    include_trace: bool = False,
    trace_log_dir: str | Path | None = None,
) -> SQLAgentState:
    """Run the SQL copilot graph.

    Args:
        question: The natural language question to answer.
        sql_generator_model: The language model to use for SQL generation.
        analyst_model: The language model to use for result analysis (optional).
        selector_model: The language model to use for database selection (optional).
        databases: The registered databases available for routing (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).
        include_trace: When true, attach a normalized per-step execution trace to `metadata["execution_trace"]`.
        trace_log_dir: Optional directory where per-run trace logs are persisted. Defaults to `logs/`.

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
        trace: list[SQLAgentTraceStep] = list(
            stream_sql_agent_execution(graph, cast(SQLAgentState, {"question": question}))
        )
        result: SQLAgentState = (
            cast(SQLAgentState, trace[-1]["state"])
            if trace
            else cast(SQLAgentState, {"question": question})
        )
        log_path = write_trace_log(question, trace, trace_log_dir)
        metadata = dict(result.get("metadata") or {})
        metadata["trace_log_path"] = str(log_path)
        metadata["execution_trace"] = trace
        result["metadata"] = metadata
        return result

    result = cast(SQLAgentState, graph.invoke({"question": question}))
    return result


def stream_question(
    question: str,
    sql_generator_model: LLM,
    analyst_model: LLM | None = None,
    selector_model: LLM | None = None,
    databases: list[RegisteredDatabase] | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
) -> Iterator[SQLAgentTraceStep]:
    """Stream normalized execution steps for a single question.

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
    return stream_sql_agent_execution(graph, cast(SQLAgentState, {"question": question}))

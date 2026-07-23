"""SQL execution node."""

from __future__ import annotations

from typing import Any

from state import SQLAgentState
from tools.database import DatabaseError, SQLiteDatabase


def _format_execution_error(error: DatabaseError, state: SQLAgentState) -> dict[str, Any]:
    """Format execution error response.

    Args:
        error (DatabaseError): Content of the error
        state (SQLAgentState): Current state to preserve retry count

    Returns:
        dict[str, Any]: Formatted error message
    """    
    return {
        "execution_error": str(error),
        "analysis": f"SQL execution failed: {error}",
        "last_execution_error": str(error),
        "retry_count": state.get("retry_count", 0),
    }

def _execute_sql(
    state: SQLAgentState,
    database: SQLiteDatabase,
    limit: int,
) -> dict[str, Any]:
    """Execute an SQL query

    Args:
        state (SQLAgentState): Current state of the Graph
        database (SQLiteDatabase): Database that will be queried
        limit (int): Limit of the output of the query

    Returns:
        dict[str, Any]: Directory with the result or an error
    """
    try:
        result = database.execute_command(_resolve_sql_query(state), limit=limit)
    except DatabaseError as exc:
        return _format_execution_error(exc, state)
    return {
        "query_result": result,
        "execution_error": None,
        "last_execution_error": None,
        "retry_count": state.get("retry_count", 0),
    }


def _resolve_sql_query(state: SQLAgentState) -> str:
    """Resolve the SQL text to execute from the current graph state.

    Args:
        state (SQLAgentState): State of the graph

    Returns:
        str: Get the SQL query to execute
    """
    intent = state.get("intent")
    if intent == "modification":
        return str(state.get("generated_sql", ""))
    return str(state.get("validated_sql", ""))


class SQLExecutorNode:
    """Execute validated SQL against SQLite."""

    def __init__(
        self,
        database: SQLiteDatabase,
        limit: int = 200,
    ) -> None:
        """Initialize the SQLExecutorNode."""
        if database is None:
            raise ValueError("database is required for SQLExecutorNode")
        self.database = database
        self.limit = limit

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """
        Execute an SQL query  and update the state with the query result.

        Args:
            state: The current state of the SQL agent.

        Returns:
            A dictionary containing the query result and any errors.

        Raises:
            DatabaseError: If the SQL execution fails.
        """
        return _execute_sql(
            state,
            self.database,
            self.limit,
        )

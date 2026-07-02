"""SQL execution node."""

from __future__ import annotations

from typing import Any

from state import SQLAgentState
from tools.database import (
    DatabaseError,
    RegisteredDatabase,
    SQLiteDatabase,
    get_default_database,
)
from nodes import get_database


def _format_execution_error(error: DatabaseError) -> dict[str, Any]:
    """Format execution error response.

    Args:
        error (DatabaseError): Content of the error

    Returns:
        dict[str, Any]: Formatted error message
    """    
    return {
        "execution_error": str(error),
        "analysis": f"SQL execution failed: {error}",
    }

def _execute_sql(
    state: SQLAgentState,
    sql_query: str,
    database: SQLiteDatabase,
    database_catalog: list[RegisteredDatabase] | None,
    limit: int,
) -> dict[str, Any]:
    """Execute an SQL query

    Args:
        state (SQLAgentState): Current state of the Graph
        sql_query (str): Query to execute
        database (SQLiteDatabase): Database that will be queried
        database_catalog (list[RegisteredDatabase] | None): Database catalog
        limit (int): Limit of the output of the query

    Returns:
        dict[str, Any]: Directory with the result or an error
    """
    database = get_database(state, database, database_catalog)
    try:
        result = database.execute_command(sql_query, limit=limit)
    except DatabaseError as exc:
        return _format_execution_error(exc)
    return {
        "query_result": result,
        "execution_error": None,
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
        database: SQLiteDatabase | None = None,
        database_catalog: list[RegisteredDatabase] | None = None,
        limit: int = 200,
    ) -> None:
        """Initialize the SQLExecutorNode."""
        self.database = database or get_default_database()
        self.database_catalog = database_catalog
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
        sql_query = _resolve_sql_query(state)
        return _execute_sql(
            state,
            sql_query,
            self.database,
            self.database_catalog,
            self.limit,
        )

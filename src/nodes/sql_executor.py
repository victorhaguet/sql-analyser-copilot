"""SQL execution node."""

from __future__ import annotations

from typing import Any

from state import SQLAgentState
from tools.database import DatabaseError, SQLiteDatabase, get_default_database


def _get_database(state: SQLAgentState, default: SQLiteDatabase) -> SQLiteDatabase:
    """Get database from state or return default.

    Args:
        state (SQLAgentState): State of the graph
        default (SQLiteDatabase): Default database

    Returns:
        SQLiteDatabase: Current database in the state
    """    
    return state.get("selected_database") or default


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


class SQLExecutorNode:
    """Execute validated SQL against SQLite."""

    def __init__(
        self,
        database: SQLiteDatabase | None = None,
        limit: int = 200,
    ) -> None:
        """Initialize the SQLExecutorNode."""
        self.database = database or get_default_database()
        self.limit = limit

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """
        Execute the validated SQL and update the state with the query result.

        Args:
            state: The current state of the SQL agent.

        Returns:
            A dictionary containing the query result and any errors.

        Raises:
            DatabaseError: If the SQL execution fails.
        """
        validated_sql = state.get("validated_sql", "")
        database = _get_database(state, self.database)
        try:
            result = database.execute_command(validated_sql, limit=self.limit) 
        except DatabaseError as exc:
            return _format_execution_error(exc)
        return {
            "query_result": result,
            "execution_error": None,
        }

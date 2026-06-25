"""SQL execution node."""

from __future__ import annotations

from typing import Any

from state import SQLAgentState
from tools.database import DatabaseError, SQLiteDatabase, get_default_database, RegisteredDatabase
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
        Execute the validated SQL and update the state with the query result.

        Args:
            state: The current state of the SQL agent.

        Returns:
            A dictionary containing the query result and any errors.

        Raises:
            DatabaseError: If the SQL execution fails.
        """
        validated_sql = state.get("validated_sql", "")
        database = get_database(state, self.database, self.database_catalog)
        try:
            result = database.execute_command(validated_sql, limit=self.limit) 
        except DatabaseError as exc:
            return _format_execution_error(exc)
        return {
            "query_result": result,
            "execution_error": None,
        }

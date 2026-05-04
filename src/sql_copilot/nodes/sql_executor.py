"""SQL execution node."""

from __future__ import annotations

from typing import Any

from sql_copilot.state import SQLAgentState
from sql_copilot.tools.database import DatabaseError, SQLiteDatabase, get_default_database


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
        try:
            result = self.database.execute_command(validated_sql, limit=self.limit)
        except DatabaseError as exc:
            return {
                "execution_error": str(exc),
                "analysis": f"SQL execution failed: {exc}",
            }
        return {
            "query_result": result,
            "execution_error": None,
        }

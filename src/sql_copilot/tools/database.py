"""
Database manager

This module provides a simple SQLite database manager focused on read-only access.
It allows you to list tables, get table schemas, 
preview table data, and execute SELECT queries with parameterization and result limits. 
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# Default path to the SQLite database file (Chinook sample database)
DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "Chinook_Sqlite.sqlite"
)

# Exceptions and errors related to database operations
class DatabaseError(Exception):
    """Base exception for database operations."""

class DatabaseNotFoundError(DatabaseError):
    """Raised when the configured SQLite database file does not exist."""

class TableNotFoundError(DatabaseError):
    """Raised when a requested table does not exist."""

@dataclass(slots=True)
class QueryResult:
    """Class QueryResult to hold the results of a SELECT query execution."""
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool


class SQLiteDatabase:
    """Small SQLite helper focused on read-only access."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        """
        Initialize the SQLiteDatabase with the given database path or use the default.

        Args:
            database_path: Optional path to the SQLite database file. If not provided, 
            the default path will be used.

        Raises:
            DatabaseNotFoundError: If the specified database file does not exist.
        """
        self.database_path = Path(database_path or DEFAULT_DB_PATH).expanduser().resolve()
        if not self.database_path.exists():
            raise DatabaseNotFoundError(
                f"SQLite database file was not found: {self.database_path}"
            )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """
        Context manager for SQLite database connection.

        Returns:
            sqlite3.Connection: SQLite database connection.
        """
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def list_tables(self) -> list[str]:
        """
        List all tables in the SQLite database.

        Returns:
            list[str]: A list of table names.
        """
        query = """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """
        with self.connection() as conn:
            rows = conn.execute(query).fetchall()
        return [str(row["name"]) for row in rows]

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the SQLite database.

        Args:
            table_name: The name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        query = """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            LIMIT 1
        """
        with self.connection() as conn:
            row = conn.execute(query, (table_name,)).fetchone()
        return row is not None

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """
        Get the schema of a table in the SQLite database.

        Args:
            table_name: The name of the table.

        Returns:
            list[dict[str, Any]]: A list of dictionaries representing the table schema.
        """
        with self.connection() as conn:
            rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()

        return [
            {
                "cid": row["cid"],
                "name": row["name"],
                "type": row["type"],
                "notnull": bool(row["notnull"]),
                "default_value": row["dflt_value"],
                "primary_key": bool(row["pk"]),
            }
            for row in rows
        ]

    def get_database_schema(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get the schema of the entire SQLite database.

        Returns:
            dict[str, list[dict[str, Any]]]: 
            A dictionary where keys are table names and values are lists of dictionaries 
            representing the table schema.
        """
        return {
            table_name: self.get_table_schema(table_name)
            for table_name in self.list_tables()
        }

    def preview_table(self, table_name: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Preview the first table rows of a specified table in the SQLite database.
        Args:
            table_name: The name of the table to preview.
            limit: The maximum number of rows to return (default is 5).

        Returns:
            list[dict[str, Any]]: A list of dictionaries representing
            the previewed rows of the table.
        """
        safe_limit = self._check_limit(limit)
        query = f'SELECT * FROM "{table_name}" LIMIT ?'
        with self.connection() as conn:
            rows = conn.execute(query, (safe_limit,)).fetchall()
        return [dict(row) for row in rows]

    def execute_command(
        self,
        query: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
        limit: int | None = 200,
    ) -> QueryResult:
        """
        Execute a SQL query against the SQLite database with optional
        parameterization and result limiting.

        Args:
            query: The SQL SELECT query to execute.
            parameters: Optional parameters for the SQL query, 
            either as a sequence for positional parameters or a mapping for named parameters.
            limit: Optional maximum number of rows to return. 
            If None, a default limit of 200 will be applied. Must be a positive integer.

        Returns:
            QueryResult: An object containing the columns, rows, row count, 
            and whether the results were truncated due to the limit.
        """

        # Validate the limit before execution
        row_limit = self._check_limit(limit)

        # Execute the query and fetch results with the specified limit
        try:
            with self.connection() as conn:
                cursor = conn.execute(query, parameters or ())
                columns = [column[0] for column in cursor.description or []]
                rows = cursor.fetchmany(row_limit)
                extra_row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(str(exc)) from exc

        # Return the results in a structured format
        result_rows = [dict(row) for row in rows]
        return QueryResult(
            columns=columns,
            rows=result_rows,
            row_count=len(result_rows),
            truncated=extra_row is not None,
        )

    def describe(self) -> dict[str, Any]:
        """
        Describe the SQLite database, including its path and the list of tables it contains.
        Returns:
            dict[str, Any]: A dictionary containing the database path 
            and a list of tables in the database.
        """
        return {
            "database_path": str(self.database_path),
            "tables": self.list_tables(),
        }

    def _check_limit(self, limit: int | None) -> int:
        """
        Check if the provided limit is valid (positive integer) and return it, 
        or raise an error if it's invalid.

        Args:
            limit: The limit to check.
        Returns:
            int: The validated limit value.

        Raises:
            DatabaseError: If the limit is not a positive integer.
        """
        if limit is None:
            return 200
        if limit <= 0:
            raise DatabaseError("Limit must be a positive integer.")
        return limit


def get_default_database() -> SQLiteDatabase:
    """
    Get the default SQLite database instance.

    Returns:
        SQLiteDatabase: The default SQLite database instance.
    """
    return SQLiteDatabase()

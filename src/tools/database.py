"""
Database manager

This module provides a simple SQLite database manager focused on read-only access.
It allows you to list tables, get table schemas, 
preview table data, and execute SELECT queries with parameterization and result limits. 
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# Get the project root directory (where src/ is a subdirectory)
# Walk up from the current file's location to find the project root
_current_file = Path(__file__).resolve()
_project_root = _current_file.parents[1] if (_current_file.parent.parent / "src").exists() else _current_file.parents[2]

# Default path to the SQLite database file (Chinook sample database)
from utils.paths import DEFAULT_DB_PATH

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


@dataclass(slots=True)
class RegisteredDatabase:
    """Database entry available to the SQL copilot router."""

    name: str
    description: str
    database: SQLiteDatabase


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

    def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        """
        Get the outgoing foreign keys declared on a table in the SQLite database.

        Args:
            table_name: The name of the table.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, one per foreign key column,
            with `from_column`, `to_table`, `to_column`, and `on_delete`.
        """
        with self.connection() as conn:
            rows = conn.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall()

        return [
            {
                "from_column": row["from"],
                "to_table": row["table"],
                "to_column": row["to"],
                "on_delete": row["on_delete"],
            }
            for row in rows
        ]

    def get_indexes(self, table_name: str) -> list[dict[str, Any]]:
        """
        Get the indexes declared on a table in the SQLite database.

        Args:
            table_name: The name of the table.

        Returns:
            list[dict[str, Any]]: A list of dictionaries with `name`, `unique`, and
            `columns` for each index.
        """
        with self.connection() as conn:
            index_rows = conn.execute(f'PRAGMA index_list("{table_name}")').fetchall()
            indexes = []
            for index_row in index_rows:
                index_name = index_row["name"]
                column_rows = conn.execute(f'PRAGMA index_info("{index_name}")').fetchall()
                indexes.append(
                    {
                        "name": index_name,
                        "unique": bool(index_row["unique"]),
                        "columns": [column_row["name"] for column_row in column_rows],
                    }
                )
        return indexes

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

                conn.commit()
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


def format_database_schema(database: SQLiteDatabase) -> str:
    """
    Render the SQLite schema into a compact prompt-friendly string.

    Args:
        database: The SQLite database to format.

    Returns:
        A compact string representation of the schema.
    """
    lines: list[str] = []
    for table_name, columns in database.get_database_schema().items():
        column_bits = []
        for column in columns:
            descriptor = f'{column["name"]} {column["type"]}'
            if column["primary_key"]:
                descriptor += " PRIMARY KEY"
            column_bits.append(descriptor)
        lines.append(f'{table_name}({", ".join(column_bits)})')
    return "\n".join(lines)


def list_referencing_tables(database: SQLiteDatabase, table_name: str) -> list[dict[str, Any]]:
    """
    List the foreign keys, across every table in the database, that reference the
    given table (i.e. the "incoming" foreign keys of `table_name`).

    Args:
        database: The database to inspect.
        table_name: The referenced table.

    Returns:
        list[dict[str, Any]]: One entry per referencing column, with `table`,
        `from_column`, and `to_column`.
    """
    referencing: list[dict[str, Any]] = []
    for other_table_name in database.list_tables():
        if other_table_name == table_name:
            continue
        for foreign_key in database.get_foreign_keys(other_table_name):
            if foreign_key["to_table"] == table_name:
                referencing.append(
                    {
                        "table": other_table_name,
                        "from_column": foreign_key["from_column"],
                        "to_column": foreign_key["to_column"],
                    }
                )
    return referencing


def format_table_detail(
    database: SQLiteDatabase,
    table_name: str,
    *,
    include_sample_rows: bool = False,
    sample_row_limit: int = 3,
) -> str:
    """
    Render a single table into a prompt-friendly, detailed block: columns with
    type/NOT NULL/default/PK, outgoing and incoming foreign keys, unique indexes,
    and optionally a few sample rows.

    Args:
        database: The database the table belongs to.
        table_name: The table to describe.
        include_sample_rows: Whether to append a handful of sample rows.
        sample_row_limit: How many sample rows to include when requested.

    Returns:
        A multi-line, human/model-readable description of the table.

    Raises:
        TableNotFoundError: If the table does not exist in the database.
    """
    if not database.table_exists(table_name):
        raise TableNotFoundError(f"Table not found: {table_name}")

    columns = database.get_table_schema(table_name)
    outgoing_foreign_keys = database.get_foreign_keys(table_name)
    incoming_foreign_keys = list_referencing_tables(database, table_name)
    unique_indexes = [index for index in database.get_indexes(table_name) if index["unique"]]

    lines = [f"Table: {table_name}", "Columns:"]
    for column in columns:
        descriptor = f'  - {column["name"]} {column["type"]}'
        if column["primary_key"]:
            descriptor += " PRIMARY KEY"
        if column["notnull"]:
            descriptor += " NOT NULL"
        if column["default_value"] is not None:
            descriptor += f' DEFAULT {column["default_value"]}'
        lines.append(descriptor)

    lines.append("Foreign keys (outgoing):")
    if outgoing_foreign_keys:
        for foreign_key in outgoing_foreign_keys:
            lines.append(
                f'  - {table_name}.{foreign_key["from_column"]} -> '
                f'{foreign_key["to_table"]}.{foreign_key["to_column"]}'
            )
    else:
        lines.append("  - (none)")

    lines.append("Foreign keys (incoming):")
    if incoming_foreign_keys:
        for reference in incoming_foreign_keys:
            lines.append(
                f'  - {reference["table"]}.{reference["from_column"]} -> '
                f'{table_name}.{reference["to_column"]}'
            )
    else:
        lines.append("  - (none)")

    lines.append("Unique indexes:")
    if unique_indexes:
        for index in unique_indexes:
            lines.append(f'  - {index["name"]} ({", ".join(index["columns"])})')
    else:
        lines.append("  - (none)")

    if include_sample_rows:
        sample_rows = database.preview_table(table_name, limit=sample_row_limit)
        lines.append("Sample rows:")
        if sample_rows:
            for row in sample_rows:
                lines.append(f"  - {row}")
        else:
            lines.append("  - (empty table)")

    return "\n".join(lines)


def format_full_schema(
    database: SQLiteDatabase, *, max_chars: int | None = None
) -> tuple[str, bool]:
    """
    Render the complete database schema, table by table, via `format_table_detail`
    (no sample rows) — every column, and every foreign key in both directions.

    Args:
        database: The database to render.
        max_chars: If the rendered text would exceed this many characters, fall back
            to the compact `format_database_schema` output instead. `None` disables
            the guard.

    Returns:
        A `(text, truncated)` pair. `truncated` is `True` when the detailed rendering
        was too large and the compact fallback was returned instead.
    """
    blocks = [
        format_table_detail(database, table_name, include_sample_rows=False)
        for table_name in database.list_tables()
    ]
    full_text = "\n\n".join(blocks)

    if max_chars is not None and len(full_text) > max_chars:
        return format_database_schema(database), True

    return full_text, False


def register_database(
    database: SQLiteDatabase,
    *,
    name: str | None = None,
    description: str | None = None,
) -> RegisteredDatabase:
    """
    Wrap a SQLiteDatabase with routing metadata.

    Args:
        database: The database instance to register.
        name: Optional logical name for the database.
        description: Optional business description used by the selector.

    Returns:
        A registered database entry.
    """
    database_name = (name or database.database_path.stem).strip()
    database_description = (description or f"SQLite database at {database.database_path.name}").strip()
    return RegisteredDatabase(
        name=database_name,
        description=database_description,
        database=database,
    )


def get_default_database() -> SQLiteDatabase:
    """
    Get the default SQLite database instance.
    
    Uses auto-discovered database if available, otherwise falls back to Chinook.
    """
    data_dir = _project_root / "data"
    if data_dir.exists():
        sqlite_files = list(data_dir.glob("*.sqlite")) + list(data_dir.glob("*.db"))
        if sqlite_files:
            return SQLiteDatabase(str(sqlite_files[0]))
    
    return SQLiteDatabase()


def load_database_catalog_from_env() -> list[RegisteredDatabase]:
    """
    Load the database catalog from `SQL_COPILOT_DATABASES`.

    The variable must contain a JSON array. Each item supports:
    `name`, `path`, and optional `description`.

    Returns:
        A list of registered databases. Falls back to the default catalog when unset.
    """
    raw_catalog = os.getenv("SQL_COPILOT_DATABASES", "").strip()
    if not raw_catalog:
         return [register_database(get_default_database())] 

    try:
        entries = json.loads(raw_catalog)
    except json.JSONDecodeError as exc:
        raise DatabaseError("SQL_COPILOT_DATABASES must be valid JSON.") from exc

    if not isinstance(entries, list) or not entries:
        raise DatabaseError("SQL_COPILOT_DATABASES must be a non-empty JSON array.")

    catalog: list[RegisteredDatabase] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise DatabaseError("Each SQL_COPILOT_DATABASES entry must be a JSON object.")
        path = entry.get("path")
        if not isinstance(path, str) or not path.strip():
            raise DatabaseError("Each SQL_COPILOT_DATABASES entry must define a string `path`.")
        catalog.append(
            register_database(
                SQLiteDatabase(path),
                name=entry.get("name"),
                description=entry.get("description"),
            )
        )
    return catalog

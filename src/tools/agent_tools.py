"""
Agent tools for the SQL generation loop.

Three self-contained, model-facing capabilities the agent composes freely while
reasoning towards a final SQL statement: inspecting the schema in detail,
asking the user a clarifying question, and running a read-only probe of a
candidate SELECT. Built by a factory (`build_agent_tools`) because each tool
needs the target database and the safety validator.

Docstrings on the `@tool`-decorated functions are sent to the model as the tool
description — they are written for the model, not for a maintainer.
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

from tools.database import SQLiteDatabase, format_table_detail
from tools.exceptions import SQLSafetyError
from tools.sql_safety import SQLSafetyValidator

# Env-tunable (Step 10, D5); the other AgentToolLimits fields aren't named in
# the plan's guardrail list and stay static defaults.
DEFAULT_PROBE_ROW_LIMIT = int(os.getenv("SQL_AGENT_PROBE_ROW_LIMIT", "20"))


@dataclass(slots=True)
class AgentToolLimits:
    """Tunable limits for the agent tools."""

    probe_row_limit: int = DEFAULT_PROBE_ROW_LIMIT
    probe_row_limit_max: int = 50
    probe_timeout_seconds: float = 5.0
    sample_row_limit: int = 3


def build_agent_tools(
    database: SQLiteDatabase,
    validator: SQLSafetyValidator,
    limits: AgentToolLimits | None = None,
) -> list[BaseTool]:
    """
    Build the three agent tools, bound to the given database and validator.

    Args:
        database: The database the agent reasons about.
        validator: Validator used to gate `run_readonly_probe` to read-only SELECTs.
        limits: Tunable row/timeout limits. Defaults to `AgentToolLimits()`.

    Returns:
        `[inspect_schema, ask_user, run_readonly_probe]`, ready to pass to
        `model.bind_tools(...)`.
    """
    resolved_limits = limits or AgentToolLimits()

    @tool
    def inspect_schema(
        tables: list[str] | None = None,
        include_sample_rows: bool = False,
    ) -> str:
        """Inspect the database schema.

        The complete schema (every table, every column with NOT NULL/default/PK,
        and every foreign key in both directions) is already provided in your
        system prompt. Do NOT call this tool to re-read anything already visible
        there. Use it only for two things: getting a few sample rows from a table
        (to see real data conventions, e.g. date formats or exact string casing),
        or re-checking a table's detail if you are unsure or the schema you were
        given was reported as truncated.

        Call with no arguments to get the list of tables with row counts, as an
        overview. Call again with `tables` set to get full detail for specific
        tables.

        Args:
            tables: Table names to describe in detail. Omit (or pass null) to get
                the table overview instead.
            include_sample_rows: When `tables` is set, also include a few sample
                rows per table.

        Returns:
            A plain-text description. Never raises: an unknown table name comes
            back as an error string listing the valid table names.
        """
        return _inspect_schema(
            database,
            tables=tables,
            include_sample_rows=include_sample_rows,
            sample_row_limit=resolved_limits.sample_row_limit,
        )

    @tool
    def ask_user(questions: list[dict]) -> str:
        """Ask the user one or more clarifying questions before finalizing the SQL.

        Use this when a business decision or a value cannot be derived from the
        question or the database itself — never for anything `inspect_schema` or
        `run_readonly_probe` could answer for you.

        Each item in `questions` must be a dict with keys:
          - key: short machine-friendly identifier for the answer, e.g. "genre"
          - question: the question to show the user
          - why: REQUIRED. A short explanation of why you need this, shown in the UI.
          - suggested_default: optional pre-filled suggestion

        Batch every question you currently need into a single call — one dialog
        with several fields beats several round-trips.

        IMPORTANT: call this tool ALONE. Never call it together with
        `inspect_schema` or `run_readonly_probe` in the same turn — a mixed batch
        is rejected and you will be asked to retry with `ask_user` on its own.

        Returns:
            The user's answers, as plain text you can use in your next step.
        """
        answers = interrupt({"kind": "clarification", "questions": questions})
        return _format_answers(answers)

    @tool
    def run_readonly_probe(sql: str, limit: int = resolved_limits.probe_row_limit) -> str:
        """Run a read-only probe of a single SELECT statement against the database.

        Use this to verify your reasoning before committing to an answer: confirm
        a join actually returns rows, check whether a value already exists, or
        count how many rows an UPDATE/DELETE would target by probing the
        equivalent SELECT. Always probe the exact SELECT you intend to return as
        your final answer before finalizing it.

        Args:
            sql: A single read-only SELECT statement. Writes are rejected — this
                tool can never modify the database, regardless of what you send.
            limit: Maximum rows to return (default 20, capped at 50).

        Returns:
            A compact text table of columns and rows plus a row count and
            truncation flag, or an error message if the statement is unsafe,
            invalid, or fails to execute. Never raises.
        """
        return _run_readonly_probe(
            database,
            validator,
            sql=sql,
            limit=limit,
            row_limit_max=resolved_limits.probe_row_limit_max,
            timeout_seconds=resolved_limits.probe_timeout_seconds,
        )

    return [inspect_schema, ask_user, run_readonly_probe]


def _inspect_schema(
    database: SQLiteDatabase,
    *,
    tables: list[str] | None,
    include_sample_rows: bool,
    sample_row_limit: int,
) -> str:
    """Implementation behind the `inspect_schema` tool (kept plain for testing).

    Args:
        database (SQLiteDatabase): Database
        tables (list[str] | None): List of tables in the database
        include_sample_rows (bool): whether to include sample rows in the description
        sample_row_limit (int): Sample row limit

    Returns:
        str: Plain text description of the database
    """
    known_tables = database.list_tables()

    if not tables:
        if not known_tables:
            return "The database has no tables."
        return "\n".join(_table_overview_line(database, table_name) for table_name in known_tables)

    unknown_tables = [table_name for table_name in tables if table_name not in known_tables]
    if unknown_tables:
        return (
            f"Unknown table(s): {', '.join(unknown_tables)}. "
            f"Valid tables are: {', '.join(known_tables)}."
        )

    blocks = [
        format_table_detail(
            database,
            table_name,
            include_sample_rows=include_sample_rows,
            sample_row_limit=sample_row_limit,
        )
        for table_name in tables
    ]
    return "\n\n".join(blocks)


def _table_overview_line(database: SQLiteDatabase, table_name: str) -> str:
    """Render one `table (N rows, M columns)` summary line.

    Args:
        database (SQLiteDatabase): Database to study
        table_name (str): Name of the table to focus on

    Returns:
        str: Description of the table
    """
    column_count = len(database.get_table_schema(table_name))
    with database.connection() as conn:
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    return f"{table_name} ({row_count} rows, {column_count} columns)"


def _format_answers(answers: Any) -> str:
    """Render the value resumed into `ask_user`'s `interrupt()` as tool output text.

    Args:
        answers (Any): Answer of the user

    Returns:
        str: formatted answer
    """
    if isinstance(answers, list):
        if not answers:
            return "The user did not provide any answers."
        lines = []
        for item in answers:
            if isinstance(item, dict):
                key = item.get("key", "answer")
                value = item.get("answer", item.get("value", ""))
                lines.append(f"{key}: {value}")
            else:
                lines.append(str(item))
        return "\n".join(lines)
    if isinstance(answers, dict):
        return "\n".join(f"{key}: {value}" for key, value in answers.items())
    return str(answers)


def _run_readonly_probe(
    database: SQLiteDatabase,
    validator: SQLSafetyValidator,
    *,
    sql: str,
    limit: int,
    row_limit_max: int,
    timeout_seconds: float,
) -> str:
    """Implementation behind the `run_readonly_probe` tool (kept plain for testing).

    Args:
        database (SQLiteDatabase): database to query
        validator (SQLSafetyValidator): validator used to reject anything unsafe
        sql (str): SQL statement requested by the model
        limit (int): rows requested by the model; clamped to `row_limit_max`
            and floored at 1, so an invalid value (e.g. 0 or negative) still
            returns a usable probe instead of falling back to a default
        row_limit_max (int): hard ceiling on rows returned
        timeout_seconds (float): wall-clock budget before the query is aborted

    Returns:
        str: formatted probe result, or an error string. Never raises.
    """
    try:
        safe_sql = validator.assert_safe_select(sql)
    except SQLSafetyError as exc:
        return f"Rejected: {exc}"

    safe_limit = min(max(limit, 1), row_limit_max)

    try:
        with _read_only_connection(database.database_path) as conn:
            _install_timeout(conn, timeout_seconds)
            cursor = conn.execute(safe_sql)
            columns = [column[0] for column in cursor.description or []]
            rows = cursor.fetchmany(safe_limit)
            extra_row = cursor.fetchone()
    except sqlite3.Error as exc:
        return f"Query failed: {exc}"

    return _format_probe_result(columns, rows, truncated=extra_row is not None)


@contextmanager
def _read_only_connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection that SQLite itself refuses to write through.

    This is the guard that makes probes read-only *by construction*: even if the
    safety validator were bypassed or buggy, a write raises at the SQLite level.

    Args:
        database_path (Path): path of the database

    Yields:
        Iterator[sqlite3.Connection]: Connect to the database in read_only mode
    """
    uri = f"file:{quote(str(database_path))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        yield conn
    finally:
        conn.close()


def _install_timeout(conn: sqlite3.Connection, timeout_seconds: float) -> None:
    """Abort the currently running query once `timeout_seconds` have elapsed.

    Args:
        conn (sqlite3.Connection): Connection to a SQLite3 database
        timeout_seconds (float): time out duration

    """
    deadline = time.monotonic() + timeout_seconds

    def handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(handler, 1000)


def _format_probe_result(
    columns: list[str],
    rows: list[tuple[Any, ...]],
    *,
    truncated: bool,
) -> str:
    """Render probe results as a compact text table.

    Args:
        columns (list[str]): Result columns of the sql query
        rows (list[tuple[Any, ...]]): Result rows of the sql query
        truncated (bool): boolean saying if the result was truncated or not

    Returns:
        str: _description_
    """
    if not rows:
        return f"0 rows. Columns: {', '.join(columns)}."

    lines = [", ".join(columns)]
    for row in rows:
        lines.append(", ".join("" if value is None else str(value) for value in row))

    summary = f"{len(rows)} row(s)" + (" (truncated)" if truncated else "")
    return f"{summary}\n" + "\n".join(lines)

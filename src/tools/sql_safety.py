"""
SQL safety guardrails.

This module validates user or model generated SQL before the database layer executes it.
Its responsibility is to accept only well-formed read-only SELECT statements and reject
multi-statement or mutating SQL.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.database import DEFAULT_DB_PATH

# Keywords forbidden in SQL queries
FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "analyze",
    "begin",
    "commit",
    "rollback",
    "savepoint",
    "release",
)

# Regular expression patterns for validation
LEADING_SELECT_PATTERN = re.compile(r"^(select|with)\b", re.IGNORECASE)

# Regular expression to detect forbidden keywords
FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Regular expressions to detect parameters
NAMED_PARAMETER_PATTERN = re.compile(r"(?<!:)([:@$])([A-Za-z_][A-Za-z0-9_]*)")
POSITIONAL_PARAMETER_PATTERN = re.compile(r"\?")

# Errors to raise when validation fails
class SQLSafetyError(Exception):
    """Base error raised by SQL safety validation."""


class EmptyQueryError(SQLSafetyError):
    """Raised when the SQL query is empty."""


class MultipleStatementsError(SQLSafetyError):
    """Raised when more than one SQL statement is supplied."""


class NonSelectQueryError(SQLSafetyError):
    """Raised when the SQL statement is not a read-only SELECT/CTE query."""


class ForbiddenKeywordError(SQLSafetyError):
    """Raised when a forbidden mutating SQL keyword is present."""


class InvalidQueryError(SQLSafetyError):
    """Raised when SQLite cannot parse the query."""


@dataclass(slots=True)
class ValidatedSQL:
    """Validation output used by downstream nodes before executing SQL."""

    original_query: str
    normalized_query: str


class SQLSafetyValidator:
    """Validate SQL so only well-formed read-only statements are allowed through."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        """Initialize the SQLSafetyValidator with the given database path or use the default."""
        self.database_path = Path(database_path or DEFAULT_DB_PATH).expanduser().resolve()

    def validate_select(
        self,
        query: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> ValidatedSQL:
        """
        Validate a SQL SELECT query to ensure it is safe for execution.

        Args:
            query: The SQL SELECT query to validate.
            parameters: Optional parameters for the SQL query, either as a sequence for positional parameters or a mapping for named parameters.

        Returns:
            ValidatedSQL: An object containing the original and normalized query.

        Raises:
            SQLSafetyError: If the query is not safe for execution.
        """
        # Step 1: Normalize the query 
        normalized_query = self._normalize_query(query)

        # Step 2: Mask literals and comments
        masked_query = _mask_literals_and_comments(normalized_query)

        # Step 3: Ensure only a single statement is present
        self._ensure_single_statement(masked_query)

        # Step 4: Ensure the statement is a SELECT or CTE query
        self._ensure_select_statement(masked_query)

        # Step 5: Ensure no forbidden keywords are present
        self._ensure_no_forbidden_keywords(masked_query)

        # Step 6: Ensure SQLite can parse the query with EXPLAIN
        self._ensure_sqlite_can_parse(normalized_query, masked_query, parameters)

        return ValidatedSQL(
            original_query=query,
            normalized_query=normalized_query,
        )

    def assert_safe_select(
        self,
        query: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> str:
        """
        Return a validated normalized query by calling validate_select function 
        and normalizing the result.

        Args:
            query: The SQL SELECT query to validate and normalize.
            parameters: Optional parameters for the SQL query.

        Returns:
            str: The normalized SQL query if validation passes.
        """
        return self.validate_select(query, parameters).normalized_query

    def is_safe_select(
        self,
        query: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> bool:
        """
        Return True when the query passes validation, False otherwise.
        
        Args:
            query: The SQL SELECT query to validate.
            parameters: Optional parameters for the SQL query.

        Returns:
            bool: True if the query is safe, False otherwise.
        """
        try:
            self.validate_select(query, parameters)
        except SQLSafetyError:
            return False
        return True

    def _normalize_query(self, query: str) -> str:
        """
        Normalize the SQL query by trimming whitespace and removing trailing semicolons.

        Args:
            query: The SQL query to normalize.

        Returns:
            str: The normalized SQL query.

        Raises:
            EmptyQueryError: If the query is empty after normalization.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise EmptyQueryError("SQL query cannot be empty.")

        if normalized_query.endswith(";"):
            normalized_query = normalized_query[:-1].rstrip()

        if not normalized_query:
            raise EmptyQueryError("SQL query cannot be empty.")

        return normalized_query

    def _ensure_single_statement(self, masked_query: str) -> None:
        """
        Ensure that only a single SQL statement is present.

        Args:
            masked_query: The SQL query with literals and comments masked.
        
        Raises:
            MultipleStatementsError: If more than one SQL statement is detected.
        """
        if ";" in masked_query:
            raise MultipleStatementsError("Only a single SQL statement is allowed.")

    def _ensure_select_statement(self, masked_query: str) -> None:
        """
        Ensure that the SQL statement is a read-only SELECT or CTE query.

        Args:
            masked_query: The SQL query with literals and comments masked.

        Raises:
            NonSelectQueryError: If the SQL statement is not a SELECT or CTE query.
        """
        leading_sql = masked_query.lstrip()
        if not LEADING_SELECT_PATTERN.match(leading_sql):
            raise NonSelectQueryError("Only SELECT queries are allowed.")

    def _ensure_no_forbidden_keywords(self, masked_query: str) -> None:
        """
        Ensure that the SQL statement does not contain forbidden keywords.

        Args:
            masked_query: The SQL query with literals and comments masked.

        Raises:
            ForbiddenKeywordError: If a forbidden SQL keyword is detected.
        """
        match = FORBIDDEN_PATTERN.search(masked_query)
        if match:
            raise ForbiddenKeywordError(
                f"Forbidden SQL keyword detected: {match.group(1).upper()}"
            )

    def _ensure_sqlite_can_parse(
        self,
        normalized_query: str,
        masked_query: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None,
    ) -> None:
        """
        Ensure that SQLite can parse the query by running EXPLAIN QUERY PLAN.

        Args:
            normalized_query: The normalized SQL query.
            masked_query: The SQL query with literals and comments masked.
            parameters: Optional parameters for the SQL query.

        Raises:
            InvalidQueryError: If SQLite cannot parse the query.
        """
        # Build the EXPLAIN QUERY PLAN statement with the normalized query
        explain_query = f"EXPLAIN QUERY PLAN {normalized_query}"
        explain_parameters = parameters
        if explain_parameters is None:
            explain_parameters = _build_dummy_parameters(masked_query)

        # Try to execute the EXPLAIN QUERY PLAN statement to ensure SQLite can parse it
        try:
            conn = sqlite3.connect(self.database_path)
            try:
                conn.execute(explain_query, explain_parameters)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise InvalidQueryError(f"Invalid SELECT query: {exc}") from exc


def validate_select_query(
    query: str,
    parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    database_path: str | Path | None = None,
) -> ValidatedSQL:
    """
    Convenience function for validating a SELECT query.

    Args:
        query: The SQL SELECT query to validate.
        parameters: Optional parameters for the SQL query, either as a sequence for positional parameters or a mapping for named parameters.
        database_path: Optional path to the SQLite database file. If not provided, the default path will be used.
    
    Returns:
        ValidatedSQL: An object containing the original and normalized query.
    """
    return SQLSafetyValidator(database_path).validate_select(query, parameters)


def ensure_safe_select_query(
    query: str,
    parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    database_path: str | Path | None = None,
) -> str:
    """
    Convenience function returning normalized SQL or raising on failure.
    
    Args:
        query: The SQL SELECT query to validate.
        parameters: Optional parameters for the SQL query, either as a sequence for positional parameters or a mapping for named parameters.
        database_path: Optional path to the SQLite database file. If not provided, the default path will be used.

    Returns:
        str: The normalized SQL query.
    """
    return SQLSafetyValidator(database_path).assert_safe_select(query, parameters)


def _mask_literals_and_comments(sql: str) -> str:
    """
    Replace strings, quoted identifiers, and comments with spaces while preserving
    statement structure and keyword positions.

    Args:
        sql: The SQL query to mask.

    Returns:
        str: The masked SQL query.
    """
    chars = list(sql)
    masked: list[str] = []
    i = 0
    length = len(chars)

    # Mask literals, quoted identifiers, and comments
    while i < length:
        current = chars[i]
        next_char = chars[i + 1] if i + 1 < length else ""

        # Mask single-line comments starting with --
        if current == "-" and next_char == "-":
            masked.extend("  ")
            i += 2
            while i < length and chars[i] != "\n":
                masked.append(" ")
                i += 1
            continue

        # Mask multi-line comments starting with /* and ending with */
        if current == "/" and next_char == "*":
            masked.extend("  ")
            i += 2
            while i < length:
                if chars[i] == "*" and i + 1 < length and chars[i + 1] == "/":
                    masked.extend("  ")
                    i += 2
                    break
                masked.append("\n" if chars[i] == "\n" else " ")
                i += 1
            continue

        # Mask string literals enclosed in single quotes, handling escaped single quotes
        if current == "'":
            masked.append(" ")
            i += 1
            while i < length:
                char = chars[i]
                masked.append("\n" if char == "\n" else " ")
                if char == "'" and not (i + 1 < length and chars[i + 1] == "'"):
                    i += 1
                    break
                if char == "'" and i + 1 < length and chars[i + 1] == "'":
                    masked.append(" ")
                    i += 2
                    continue
                i += 1
            continue

        # Mask quoted identifiers enclosed in double quotes, backticks, or square brackets
        if current in {'"', "`"}:
            quote = current
            masked.append(" ")
            i += 1
            while i < length:
                char = chars[i]
                masked.append("\n" if char == "\n" else " ")
                if char == quote:
                    i += 1
                    break
                i += 1
            continue

        # Mask quoted identifiers enclosed in square brackets, which can contain nested brackets
        if current == "[":
            masked.append(" ")
            i += 1
            while i < length:
                char = chars[i]
                masked.append("\n" if char == "\n" else " ")
                if char == "]":
                    i += 1
                    break
                i += 1
            continue

        masked.append(current)
        i += 1

    return "".join(masked)


def _build_dummy_parameters(masked_query: str) -> Sequence[Any] | Mapping[str, Any]:
    """
    Build placeholder values so SQLite can parse parameterized statements with EXPLAIN.

    Args:
        masked_query: The SQL query with literals and comments masked.

    Returns:
        Sequence[Any] | Mapping[str, Any]: A sequence of None for positional parameters or a
        mapping of parameter names to None for named parameters.
    """
    named_matches = NAMED_PARAMETER_PATTERN.findall(masked_query)
    positional_count = len(POSITIONAL_PARAMETER_PATTERN.findall(masked_query))

    if named_matches and positional_count:
        return {"mixed_parameters_not_supported": None}

    if named_matches:
        return {name: None for _, name in named_matches}

    if positional_count:
        return tuple(None for _ in range(positional_count))

    return ()

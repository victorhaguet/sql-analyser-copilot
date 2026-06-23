"""SQL safety exception classes."""

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

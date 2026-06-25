"""SQL validation node."""

from __future__ import annotations

from typing import Any

from state import SQLAgentState
from tools.sql_safety import SQLSafetyError, SQLSafetyValidator
from tools.database import RegisteredDatabase


def _get_validator(state: SQLAgentState, default: SQLSafetyValidator, database_catalog: list[RegisteredDatabase] | None = None) -> SQLSafetyValidator:
    """Get validator based on selected database in state.

    Args:
        state (SQLAgentState): Current state of the graph
        default (SQLSafetyValidator): Default validator
        database_catalog (list[RegisteredDatabase] | None): Database catalog for lookup

    Returns:
        SQLSafetyValidator: Validator of the current graph
    """    
    selected_database_name = state.get("selected_database")
    if selected_database_name is None:
        return default
    
    if database_catalog:
        for entry in database_catalog:
            if entry.name == selected_database_name:
                return SQLSafetyValidator(entry.database.database_path)
    
    return default


def _format_validation_error(error: SQLSafetyError) -> dict[str, Any]:
    """Format validation error response.

    Args:
        error (SQLSafetyError): Error's content

    Returns:
        dict[str, Any]: Formatted error message
    """    
    return {
        "validated_sql": "",
        "sql_validation_error": str(error),
        "analysis": f"SQL validation failed: {error}",
    }


class SQLValidatorNode:
    """Validate model-generated SQL before execution."""

    def __init__(self, validator: SQLSafetyValidator | None = None, database_catalog: list[RegisteredDatabase] | None = None) -> None:
        """Initialize the SQLValidatorNode."""
        self.validator = validator or SQLSafetyValidator()
        self.database_catalog = database_catalog

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """
        Validate the generated SQL and update the state with the validation result.

        Args:
            state: The current state of the SQL agent.

        Returns:
            A dictionary containing the validation result and any errors.

        Raises:
            SQLSafetyError: If the generated SQL is deemed unsafe.
        """
        generated_sql = state.get("generated_sql", "")
        validator = _get_validator(state, self.validator, self.database_catalog)
        try:
            validated_sql = validator.assert_safe_select(generated_sql)
        except SQLSafetyError as exc:
            return _format_validation_error(exc)
        return {
            "validated_sql": validated_sql,
            "sql_validation_error": None,
        }

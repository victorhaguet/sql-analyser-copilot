"""SQL validation node."""

from __future__ import annotations

from typing import Any

from sql_copilot.state import SQLAgentState
from sql_copilot.tools.sql_safety import SQLSafetyError, SQLSafetyValidator


def _get_validator(state: SQLAgentState, default: SQLSafetyValidator) -> SQLSafetyValidator:
    """Get validator based on selected database in state.

    Args:
        state (SQLAgentState): Current state of the graph
        default (SQLSafetyValidator): Default validator

    Returns:
        SQLSafetyValidator: Validator of the current graph
    """    
    selected_database = state.get("selected_database")
    return (
        SQLSafetyValidator(selected_database.database_path)
        if selected_database is not None
        else default
    )


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

    def __init__(self, validator: SQLSafetyValidator | None = None) -> None:
        """Initialize the SQLValidatorNode."""
        self.validator = validator or SQLSafetyValidator()

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
        validator = _get_validator(state, self.validator)
        try:
            validated_sql = validator.assert_safe_select(generated_sql)
        except SQLSafetyError as exc:
            return _format_validation_error(exc)
        return {
            "validated_sql": validated_sql,
            "sql_validation_error": None,
        }

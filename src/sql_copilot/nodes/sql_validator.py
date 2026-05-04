"""SQL validation node."""

from __future__ import annotations

from typing import Any

from sql_copilot.state import SQLAgentState
from sql_copilot.tools.sql_safety import SQLSafetyError, SQLSafetyValidator


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
        try:
            validated_sql = self.validator.assert_safe_select(generated_sql)
        except SQLSafetyError as exc:
            return {
                "validated_sql": "",
                "sql_validation_error": str(exc),
                "analysis": f"SQL validation failed: {exc}",
            }
        return {
            "validated_sql": validated_sql,
            "sql_validation_error": None,
        }

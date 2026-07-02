"""SQL modification validation node with user interrupt."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from state import SQLAgentState


class SQLModificationValidatorNode:
    """Validate SQL modifications and ask user confirmation via interrupt."""

    def __init__(self) -> None:
        """Initialize the SQLModificationValidatorNode."""

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """
        Ask user validation for SQL modifications using interrupt.
        
        Args:
            state: The current state of the SQL agent, expected to contain
                'generated_sql' and 'question'.

        Returns:
            A dictionary containing the approval result.
        """
        user_decision = interrupt({
            "question": "Please verify that the following query correspond to the modification you want to apply. Every modification is definitive and can't be cancelled after your confirmation.",
            "request": state.get("question", ""),
            "draft": state.get("generated_sql", ""),
            "options": ["approve", "reject"],
        })

        return {
            "execution_confirmed": user_decision == "approve",
            "execution_requested": False,
        }

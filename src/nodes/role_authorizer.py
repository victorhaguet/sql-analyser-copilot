"""Role authorization node."""

from __future__ import annotations

from typing import Any

from state import SQLAgentState


AUTHORIZATION_ERROR = "Authorization denied: User does not have permission to execute SQL queries."


class RoleAuthorizerNode:
    """Check user role authorization before proceeding with SQL operations."""

    def __init__(self) -> None:
        """Initialize the RoleAuthorizerNode."""

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """
        Check if the user has sufficient role (editor or admin) to proceed.

        Args:
            state: The current state of the SQL agent, expected to contain 'user_role'.

        Returns:
            A dictionary containing authorization result and any errors.
        """
        user_role = state.get("user_role", "readonly")

        if user_role not in {"editor", "admin"}:
            return {
                "authorization_error": AUTHORIZATION_ERROR,
                "is_authorized": False,
            }

        return {
            "authorization_error": None,
            "is_authorized": True,
        }

"""Helpers for the standalone Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_empty_result_state() -> dict[str, object]:
    """
    Return the default result state used by the UI.
    
    Returns:
        A dictionary with default values for all fields expected by the UI.
    """
    return {
        "question": "",
        "sql_query": "",
        "ai_answer": "",
        "selected_database": "",
        "query_result": None,
        "schema_overview": "",
        "metadata": {},
        "has_error": False,
        "error_message": "",
    }


def build_display_context(response_data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize the API response into a UI-friendly display state.

    Args:
        response_data: The JSON response returned by the FastAPI `/query` endpoint.

    Returns:
        A dictionary with values formatted for display in the UI.
    """
    # Get the SQL query to display
    sql_query = (
        response_data.get("validated_sql")
        or response_data.get("generated_sql")
        or response_data.get("sql_validation_error")
        or response_data.get("execution_error")
        or ""
    )

    # Get the AI summary
    ai_answer = (
        response_data.get("analysis")
        or response_data.get("intent_error")
        or response_data.get("execution_error")
        or response_data.get("sql_validation_error")
        or "No analysis returned."
    )
    return {
        **build_empty_result_state(),
        "question": response_data.get("question") or "",
        "sql_query": sql_query,
        "ai_answer": ai_answer,
        "selected_database": response_data.get("selected_database") or "",
        "query_result": response_data.get("query_result"),
        "schema_overview": response_data.get("schema_overview") or "",
        "metadata": response_data.get("metadata") or {},
        "has_error": bool(
            response_data.get("execution_error") or response_data.get("sql_validation_error") or response_data.get("intent_error")
        ),
    }


def build_display_payload(response_data: dict[str, Any]) -> tuple[str, str, str]:
    """
    Extract the three user-facing values shown in the UI.

    Args:
        response_data: The JSON response returned by the FastAPI `/query` endpoint.

    Returns:
        A tuple containing the original query, the SQL query, and the AI answer.
    """
    context = build_display_context(response_data)
    return context["question"], context["sql_query"], context["ai_answer"]


def load_stylesheet() -> str:
    """
    Load the external stylesheet used by the Streamlit UI.

    Returns:
        The contents of the stylesheet as a string.
    """
    stylesheet_path = Path(__file__).resolve().parent / "assets" / "streamlit_styles.html"
    return stylesheet_path.read_text(encoding="utf-8")

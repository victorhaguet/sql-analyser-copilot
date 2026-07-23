"""Helpers for the standalone Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_empty_result_state() -> dict[str, Any]:
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
        "execution_requested": False,
        "execution_confirmed": False,
        "thread_id": "",
        "interrupt": None,
        "intent": None,
        "retry_count": 0,
        "max_retries": 3,
        "last_execution_error": None,
        "regeneration_error": None,
        "previous_sql": "",
        "regeneration_explanation": "",
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
    # For modification queries, use generated_sql (contains the final executed SQL after user edits/regenerations)
    # For query-type intents, use validated_sql (contains the safe SELECT statement)
    intent = response_data.get("intent")
    if intent == "modification":
        sql_query = (
            response_data.get("generated_sql")
            or response_data.get("validated_sql")
            or response_data.get("sql_validation_error")
            or response_data.get("execution_error")
            or ""
        )
    else:
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
        "execution_requested": bool(response_data.get("execution_requested")),
        "execution_confirmed": bool(response_data.get("execution_confirmed")),
        "thread_id": response_data.get("thread_id") or "",
        "interrupt": response_data.get("interrupt"),
        "intent": response_data.get("intent"),
        "retry_count": response_data.get("retry_count", 0),
        "max_retries": response_data.get("max_retries", 3),
        "last_execution_error": response_data.get("last_execution_error"),
        "regeneration_error": response_data.get("regeneration_error"),
        "previous_sql": response_data.get("previous_sql") or "",
        "regeneration_explanation": response_data.get("regeneration_explanation") or "",
        "execution_error": response_data.get("execution_error"),
        "sql_validation_error": response_data.get("sql_validation_error"),
        "has_error": bool(
            response_data.get("execution_error")
            or response_data.get("sql_validation_error")
            or response_data.get("intent_error")
            or response_data.get("regeneration_error")
        ),
        "error_message": response_data.get("regeneration_error")
        or response_data.get("execution_error")
        or response_data.get("sql_validation_error")
        or response_data.get("intent_error")
        or "",
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

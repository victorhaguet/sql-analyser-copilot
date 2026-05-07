"""Streamlit entrypoint for the SQL copilot UI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sql_copilot.streamlit_ui import (
    build_display_context,
    build_empty_result_state,
    load_stylesheet,
)

# Constants and helper functions for the Streamlit UI
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_QUESTION = ""
SAMPLE_QUESTIONS = [
    "Which 5 artists have the most albums?",
    "Show me the top 10 customers by total invoice amount.",
    "Which genres generate the highest revenue?",
]


def call_api(question: str, execution_limit: int, api_base_url: str) -> dict[str, object]:
    """
    Call the FastAPI backend and map the response into UI fields.

    Args:
        question: The user's natural language question.
        execution_limit: The maximum number of rows to return from the SQL query.
        api_base_url: The base URL of the FastAPI backend.

    Returns:
        A dictionary with values formatted for display in the UI.
    """
    clean_question = question.strip()

    # Handle the case where the user tries to submit an empty question
    if not clean_question:
        empty_state = build_empty_result_state()
        empty_state["ai_answer"] = "Enter a question before running the analysis."
        empty_state["has_error"] = True
        empty_state["error_message"] = "Missing question"
        return empty_state

    try:
        # Make the API request to the FastAPI backend
        response = httpx.post(
            f"{api_base_url.rstrip('/')}/query",
            json={
                "question": clean_question,
                "execution_limit": execution_limit,
            },
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        error_state = build_empty_result_state()
        error_state["question"] = clean_question
        error_state["ai_answer"] = f"FastAPI request failed: {exc}"
        error_state["has_error"] = True
        error_state["error_message"] = "API request failed"
        return error_state

    try:
        # Parse the JSON response from the API
        response_data = response.json()
    except ValueError:
        response_data = {}

    # Handle the case where the API request itself failed (e.g. non-200 status code)
    if response.is_error:
        detail = response_data.get("detail") or "The API request failed."
        error_state = build_empty_result_state()
        error_state["question"] = clean_question
        error_state["ai_answer"] = str(detail)
        error_state["has_error"] = True
        error_state["error_message"] = f"HTTP {response.status_code}"
        return error_state

    return build_display_context(response_data)


def _initialize_state() -> None:
    """Populate session state with predictable defaults."""
    st.session_state.setdefault("question_input", DEFAULT_QUESTION)
    st.session_state.setdefault("result_state", build_empty_result_state())


def _apply_sample_question(question: str) -> None:
    """
    Write a sample question into the main text input.

    Args:
        question: The sample question to apply.
    """
    st.session_state["question_input"] = question


def _render_header() -> None:
    """Render the page header."""
    with st.container(border=True, key="header"):
        st.title("Natural language to SQL copilot")
        st.write(
            "Ask a question to your database. Get an answer. Check the SQL query."
        )


def _render_sample_prompts() -> None:
    """Render a row of quick-start prompts."""
    st.caption("Starter prompts")
    columns = st.columns([1.05, 1.2, 1.0], gap="small")
    for column, prompt in zip(columns, SAMPLE_QUESTIONS):
        with column:
            if st.button(prompt, key=f"sample::{prompt}", use_container_width=False):
                _apply_sample_question(prompt)


def _render_question() -> tuple[str, bool]:
    """
    Render the question form.

    Returns:
        The current question value and whether the form was submitted.
    """
    with st.form("sql-copilot-form"):
        question = st.text_area(
            "Question",
            key="question_input",
            placeholder="Example: Which 5 artists have the most albums ?",
            height=180,
        )
        submit_columns = st.columns([1, 4])
        with submit_columns[0]:
            submitted = st.form_submit_button("Run analysis", use_container_width=True)

    return question, submitted


def _render_sidebar_controls() -> int:
    """
    Render the execution limit control in the sidebar.

    Returns:
        The selected execution limit.
    """
    st.sidebar.title("Settings")
    st.sidebar.markdown("### Runtime")
    execution_limit = st.sidebar.slider(
        "Execution limit",
        min_value=1,
        max_value=500,
        value=st.session_state.get("execution_limit", 200),
        step=1,
    )
    st.sidebar.markdown("<p style='font-size:0.875rem;'>This limit caps the number of rows returned by the SQL execution preview.</p>", unsafe_allow_html=True)
    st.session_state["execution_limit"] = execution_limit
    return execution_limit


def _render_result_summary(result_state: dict[str, object]) -> None:
    """
    Show the top-line status for the latest run.

    Args:
        result_state: The current state of the result to display.
    """
    # Extract row count and truncation status from the query result metadata
    query_result = result_state.get("query_result") or {}
    row_count = int(query_result.get("row_count") or 0) if isinstance(query_result, dict) else 0
    truncated = bool(query_result.get("truncated")) if isinstance(query_result, dict) else False

    # Display key metrics about the query result in a horizontal layout
    metric_columns = st.columns(3)
    metric_columns[0].metric("Rows returned", row_count)
    metric_columns[1].metric("Execution limit", st.session_state.get("execution_limit", 200))
    metric_columns[2].metric("Result status", "Needs review" if result_state.get("has_error") else "Ready")

    if truncated:
        st.info(
            "The query returned more rows than the selected execution limit, so the preview is truncated."
        )


def _render_results(result_state: dict[str, object]) -> None:
    """
    Render the main result tabs.

    Args:
        result_state: The current state of the result to display.
    """
    # If no question has been submitted yet, show an empty state message instead of the tabs
    if not any(
        [
            result_state.get("question"),
            result_state.get("sql_query"),
            result_state.get("ai_answer"),
        ]
    ):
        st.info("Submit a question to see the generated SQL, answer, and preview rows here.")
        return

    # Render the result summary and tabs
    _render_result_summary(result_state)
    answer_tab, sql_tab, data_tab = st.tabs(["Answer", "SQL", "Result Preview"])

    # Render the AI answer
    with answer_tab:
        if result_state.get("has_error"):
            st.error(str(result_state.get("ai_answer") or "The request failed."))
        else:
            st.write(str(result_state.get("ai_answer") or ""))

        # with st.expander("Question and execution context", expanded=False):
        #     st.text_area(
        #         "Original question",
        #         value=str(result_state.get("question") or ""),
        #         height=120,
        #         disabled=True,
        #     )
        #     schema_overview = str(result_state.get("schema_overview") or "")
        #     if schema_overview:
        #         st.code(schema_overview, language="text")
        #     metadata = result_state.get("metadata") or {}
        #     if metadata:
        #         st.json(metadata)

    # Render the generated SQL query
    with sql_tab:
        st.code(str(result_state.get("sql_query") or "-- SQL will appear here"), language="sql")

    # Render the query result preview
    with data_tab:
        query_result = result_state.get("query_result")
        if isinstance(query_result, dict) and query_result.get("rows"):
            st.dataframe(query_result["rows"], use_container_width=True, hide_index=True)
        else:
            st.info("No result rows were returned for this run.")


def render_app() -> None:
    """Render the Streamlit application."""
    # Set page configuration and load custom stylesheet
    st.set_page_config(
        page_title="SQL Analyser Copilot",
        page_icon=":material/database:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load the custom stylesheet for the app
    st.markdown(load_stylesheet(), unsafe_allow_html=True)

    # Initialize the application state
    _initialize_state()
    api_base_url = os.getenv("SQL_COPILOT_API_BASE_URL", DEFAULT_API_BASE_URL) # FastAPI backend URL
    execution_limit = _render_sidebar_controls()

    # Create the initial UI components for the app
    _render_header()
    _render_sample_prompts()
    question, submitted = _render_question()

    if submitted:
        with st.spinner("Running query pipeline..."):
            st.session_state["result_state"] = call_api(
                question=question,
                execution_limit=execution_limit,
                api_base_url=api_base_url,
            )

    with st.container(border=True, key="result"):
        _render_results(st.session_state["result_state"])


if __name__ == "__main__":
    render_app()

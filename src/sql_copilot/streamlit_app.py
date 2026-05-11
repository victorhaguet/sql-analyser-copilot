"""Streamlit entrypoint for the SQL copilot UI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sql_copilot.streamlit_ui import (
    build_display_context,
    build_empty_result_state,
    load_stylesheet,
)
from sql_copilot.tools.database import (
    DatabaseError, 
    RegisteredDatabase, 
    load_database_catalog_from_env
)

load_dotenv()

# Constants and helper functions for the Streamlit UI
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_QUESTION = ""


def call_api(
    question: str,
    execution_limit: int,
    api_base_url: str,
    selected_databases: list[str],
) -> dict[str, object]:
    """
    Call the FastAPI backend and map the response into UI fields.

    Args:
        question: The user's natural language question.
        execution_limit: The maximum number of rows to return from the SQL query.
        api_base_url: The base URL of the FastAPI backend.
        selected_databases: The database names enabled by the user.

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
                "selected_databases": selected_databases, # Get the selected databases
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
    st.session_state.setdefault("selected_databases", [])


def _sync_selected_databases(databases: list[RegisteredDatabase]) -> list[str]:
    """
    Reconcile the selected database state with the configured catalog.

    Args:
        databases: The currently configured database catalog.

    Returns:
        The normalized list of selected database names.
    """
    available_names = [database.name for database in databases]
    current_selection = [
        name for name in st.session_state.get("selected_databases", [])
        if name in available_names
    ]
    if not current_selection:
        current_selection = available_names.copy()
    st.session_state["selected_databases"] = current_selection
    return current_selection


def _render_header() -> None:
    """Render the page header."""
    with st.container(border=True, key="header"):
        st.title("Natural language to SQL copilot")
        st.write(
            "Ask a question to your database. Get an answer. Check the SQL query."
        )


@st.cache_data(show_spinner=False)
def _load_database_catalog() -> tuple[list[RegisteredDatabase], str | None]:
    """
    Load the configured databases for UI display.

    Returns:
        A tuple containing the configured databases and an optional error message.
    """
    try:
        return load_database_catalog_from_env(), None
    except DatabaseError as exc:
        return [], str(exc)


def _render_database_catalog() -> list[str]:
    """Render the configured database catalog and return the active selection."""
    databases, error_message = _load_database_catalog()

    with st.container(border=True, key="database-catalog"):
        st.text("Available databases")

        # Errors
        if error_message:
            st.error(f"Database catalog configuration is invalid: {error_message}")
            return []

        if not databases:
            st.info("No databases are configured.")
            return []

        selected_databases = _sync_selected_databases(databases)
        selected_count = len(selected_databases)
        st.caption(f"{selected_count} of {len(databases)} databases active")

        # Database entries
        columns = st.columns(min(len(databases), 3), gap="small")
        for index, database_entry in enumerate(databases):
            metadata = database_entry.database.describe()
            table_count = len(metadata.get("tables") or [])
            is_selected = database_entry.name in selected_databases
            disabled = is_selected and selected_count == 1
            with columns[index % len(columns)]:
                with st.container(border=True):
                    st.markdown(f"**{database_entry.name}**")
                    st.caption("Selected" if is_selected else "Inactive")
                    st.write(database_entry.description)
                    st.caption(
                        f"{table_count} tables • {Path(str(metadata['database_path'])).name}"
                    )
                    updated_value = st.toggle(
                        "Use this database",
                        value=is_selected,
                        key=f"database-toggle::{database_entry.name}",
                        disabled=disabled,
                    )
                    if updated_value != is_selected:
                        if updated_value:
                            selected_databases.append(database_entry.name)
                        else:
                            selected_databases = [
                                name for name in selected_databases if name != database_entry.name
                            ]
                        st.session_state["selected_databases"] = selected_databases
                        st.rerun()

        return st.session_state["selected_databases"]


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
    selected_database = str(result_state.get("selected_database") or "")

    # Display key metrics about the query result in a horizontal layout
    metric_columns = st.columns(4)
    metric_columns[0].metric("Database used", selected_database or "Unknown")
    metric_columns[1].metric("Rows returned", row_count)
    metric_columns[2].metric("Execution limit", st.session_state.get("execution_limit", 200))
    metric_columns[3].metric("Result status", "Needs review" if result_state.get("has_error") else "Ready")

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
    selected_databases = _render_database_catalog()
    question, submitted = _render_question()

    if submitted:
        with st.spinner("Running query pipeline..."):
            st.session_state["result_state"] = call_api(
                question=question,
                execution_limit=execution_limit,
                api_base_url=api_base_url,
                selected_databases=selected_databases,
            )

    result_state = st.session_state["result_state"]
    has_result_content = any(
        [
            result_state.get("question"),
            result_state.get("sql_query"),
            result_state.get("ai_answer"),
        ]
    )
    if has_result_content:
        with st.container(border=True, key="result"):
            _render_results(result_state)


if __name__ == "__main__":
    render_app()

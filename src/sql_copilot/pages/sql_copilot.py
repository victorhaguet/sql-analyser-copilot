"""SQL Copilot page rendering."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sql_copilot.pages.login import is_authenticated, logout, show_login_page # pylint: disable=no-name-in-module
from sql_copilot.streamlit_ui import ( # pylint: disable=no-name-in-module
    build_display_context,
    build_empty_result_state,
)
from sql_copilot.tools.database import ( # pylint: disable=no-name-in-module
    DatabaseError,
    RegisteredDatabase,
    load_database_catalog_from_env,
)

load_dotenv()

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_QUESTION = ""


def call_api(
    question: str,
    execution_limit: int,
    api_base_url: str,
    selected_databases: list[str],
) -> dict[str, object]:
    """Call the SQL copilot

    Args:
        question (str): User question
        execution_limit (int): Execution limit (max number of rows in the answer)
        api_base_url (str): Base URL of the fast API backend
        selected_databases (list[str]): Database studied

    Returns:
        dict[str, object]: Result of the question
    """
    # Clear the question
    clean_question = question.strip()

    # If the question is empty, send error message
    if not clean_question:
        empty_state = build_empty_result_state()
        empty_state["ai_answer"] = "Enter a question before running the analysis."
        empty_state["has_error"] = True
        empty_state["error_message"] = "Missing question"
        return empty_state

    # Get user information (state + ID and role)
    user = st.session_state.get("user", {})
    headers = {
        "X-User-Sub": user.get("sub", ""),
        "X-User-Role": user.get("role", ""),
    }

    try:
        # Call the FastAPI backend
        response = httpx.post(
            f"{api_base_url.rstrip('/')}/query",
            json={
                "question": clean_question,
                "execution_limit": execution_limit,
                "selected_databases": selected_databases,
            },
            headers=headers,
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        # If it doesn't work, display an error message
        error_state = build_empty_result_state()
        error_state["question"] = clean_question
        error_state["ai_answer"] = f"FastAPI request failed: {exc}"
        error_state["has_error"] = True
        error_state["error_message"] = "API request failed"
        return error_state

    # Get the response into json format
    try:
        response_data = response.json()
    except ValueError:
        response_data = {}

    # Return an error message in case the fastAPi call didn't work
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
    """Initialize the state of the page
    """
    st.session_state.setdefault("question_input", DEFAULT_QUESTION)
    st.session_state.setdefault("result_state", build_empty_result_state())
    st.session_state.setdefault("selected_databases", [])


def _sync_selected_databases(databases: list[RegisteredDatabase]) -> list[str]:
    """_summary_

    Args:
        databases (list[RegisteredDatabase]): All the databases selected by the user

    Returns:
        list[str]: List of the databases selected
    """
    # Check available databases (selected by the user)
    available_names = [database.name for database in databases]
    current_selection = [
        name
        for name in st.session_state.get("selected_databases", [])
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
        st.write("Ask a question to your database. Get an answer. Check the SQL query.")


@st.cache_data(show_spinner=False)
def _load_database_catalog() -> tuple[list[RegisteredDatabase], str | None]:
    """Load all the database catalog

    Returns:
        tuple[list[RegisteredDatabase], str | None]: List of all the databases
    """
    try:
        return load_database_catalog_from_env(), None
    except DatabaseError as exc:
        return [], str(exc)


def _render_database_catalog() -> list[str]:
    """Render databases catalog.
    Each database has one container where the user can access to its metadata and select/unselect it.

    Returns:
        list[str]: List of selected databases
    """
    # Load the databases catalog
    databases, error_message = _load_database_catalog()

    # Show all the accessible databases
    with st.container(border=True, key="database-catalog"):
        st.text("Available databases")

        if error_message:
            st.error(f"Database catalog configuration is invalid: {error_message}")
            return []

        if not databases:
            st.info("No databases are configured.")
            return []

        # Check the selected databases
        selected_databases = _sync_selected_databases(databases)
        selected_count = len(selected_databases)
        st.caption(f"{selected_count} of {len(databases)} databases active")

        # Display one block per database
        columns = st.columns(min(len(databases), 3), gap="small")
        for index, database_entry in enumerate(databases):
            # For each database
            metadata = database_entry.database.describe()
            table_count = len(metadata.get("tables") or [])
            is_selected = database_entry.name in selected_databases
            disabled = is_selected and selected_count == 1
            # Information showed in the database container
            with columns[index % len(columns)]:
                with st.container(border=True):
                    st.markdown(f"**{database_entry.name}**")
                    st.caption("Selected" if is_selected else "Inactive")
                    st.write(database_entry.description)
                    st.caption(
                        f"{table_count} tables • {Path(str(metadata['database_path'])).name}"
                    )
                    # Enable/desable a database
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
                                name
                                for name in selected_databases
                                if name != database_entry.name
                            ]
                        st.session_state["selected_databases"] = selected_databases
                        st.rerun()

        return st.session_state["selected_databases"]


def _render_question_panel() -> tuple[str, int, bool]:
    """Render question panel

    Returns:
        tuple[str, int, bool]: List of input for the agent
    """
    with st.form("sql-copilot-form"):
        # Question area
        question = st.text_area(
            "Question",
            key="question_input",
            placeholder="Example: Which 5 artists have the most albums ?",
            height=180,
        )
        # Execution limit bar
        execution_limit = st.slider(
            "Execution limit",
            min_value=1,
            max_value=500,
            value=st.session_state.get("execution_limit", 200),
            step=1,
        )
        st.caption(
            "This limit caps the number of rows returned by the SQL execution preview."
        )
        submit_columns = st.columns([1, 4])

        # Get all info when the run analysis button is pressed
        with submit_columns[0]:
            submitted = st.form_submit_button("Run analysis", use_container_width=True)

    st.session_state["execution_limit"] = execution_limit
    return question, execution_limit, submitted


def _render_result_summary(result_state: dict[str, object]) -> None:
    """Render result metrics:
    - Database used
    - rows returned
    - Execution limit
    - Result status

    Args:
        result_state (dict[str, object]): Output of the agent
    """
    # Get result information
    query_result = result_state.get("query_result") or {}
    row_count = (
        int(query_result.get("row_count") or 0)
        if isinstance(query_result, dict)
        else 0
    )
    truncated = (
        bool(query_result.get("truncated"))
        if isinstance(query_result, dict)
        else False
    )
    selected_database = str(result_state.get("selected_database") or "")

    # Show them
    metric_columns = st.columns(4)
    metric_columns[0].metric("Database used", selected_database or "Unknown")
    metric_columns[1].metric("Rows returned", row_count)
    metric_columns[2].metric(
        "Execution limit",
        st.session_state.get("execution_limit", 200),
    )
    metric_columns[3].metric(
        "Result status",
        "Needs review" if result_state.get("has_error") else "Ready",
    )

    # Tell the user if the preview is truncated 
    if truncated:
        st.info(
            "The query returned more rows than the selected execution limit, so the preview is truncated."
        )


def _render_results(result_state: dict[str, object]) -> None:
    """Render results:
    - Summary of the result
    - SQL request used
    - Table obtained from the SQL request

    Args:
        result_state (dict[str, object]): _description_
    """
    # Render result metrics
    _render_result_summary(result_state)

    # Get result values
    answer_tab, sql_tab, data_tab = st.tabs(["Answer", "SQL", "Result Preview"])

    # Write the AI anwser
    with answer_tab:
        if result_state.get("has_error"):
            st.error(str(result_state.get("ai_answer") or "The request failed."))
        else:
            st.write(str(result_state.get("ai_answer") or ""))

    # Write the SQL request used
    with sql_tab:
        st.code(
            str(result_state.get("sql_query") or "-- SQL will appear here"),
            language="sql",
        )

    # Show the Table obtained from the SQL request
    with data_tab:
        query_result = result_state.get("query_result")
        if isinstance(query_result, dict) and query_result.get("rows"):
            st.dataframe(query_result["rows"], use_container_width=True, hide_index=True)
        else:
            st.info("No result rows were returned for this run.")


def _render_logout_button() -> None:
    """Render log out button"""
    # Log out button
    user = st.session_state.get("user", {})
    header_columns = st.columns([5, 1])
    with header_columns[0]:
        st.caption(f"Signed in as {user.get('username', '')}")
    with header_columns[1]:
        if st.button("Sign out", use_container_width=True):
            logout()


def render_sql_copilot_page() -> None:
    """Render the SQL Copilot page or the login form when logged out."""
    # Check if somebody is logged
    if not is_authenticated():
        show_login_page()
        return

    # Initialize the page + the log out button
    _render_logout_button()
    _initialize_state()
    api_base_url = os.getenv("SQL_COPILOT_API_BASE_URL", DEFAULT_API_BASE_URL)

    # Render elements of the page
    _render_header()
    selected_databases = _render_database_catalog()
    question, execution_limit, submitted = _render_question_panel()

    # When a query is submitted, call the api
    if submitted:
        with st.spinner("Running query pipeline..."):
            st.session_state["result_state"] = call_api(
                question=question,
                execution_limit=execution_limit,
                api_base_url=api_base_url,
                selected_databases=selected_databases,
            )

    # Get a result and render it
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

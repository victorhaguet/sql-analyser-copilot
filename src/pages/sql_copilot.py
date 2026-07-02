"""SQL Copilot page rendering."""

from __future__ import annotations

from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

from pages.login import is_authenticated_page, show_login_page
from pages.auth import render_logout_button
from pages.config import DEFAULT_API_BASE_URL, DEFAULT_QUESTION
from streamlit_ui import build_display_context, build_empty_result_state
from tools.database import DatabaseError, RegisteredDatabase, load_database_catalog_from_env

load_dotenv()


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
    clean_question = question.strip()

    if not clean_question:
        empty_state = build_empty_result_state()
        empty_state["ai_answer"] = "Enter a question before running the analysis."
        empty_state["has_error"] = True
        empty_state["error_message"] = "Missing question"
        return empty_state

    user = st.session_state.get("user", {})
    headers = {
        "X-User-Sub": user.get("sub", ""),
        "X-User-Role": user.get("role", ""),
    }

    try:
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
        error_state = build_empty_result_state()
        error_state["question"] = clean_question
        error_state["ai_answer"] = f"FastAPI request failed: {exc}"
        error_state["has_error"] = True
        error_state["error_message"] = "API request failed"
        return error_state

    try:
        response_data = response.json()
    except ValueError:
        response_data = {}

    if response.is_error:
        detail = response_data.get("detail") or "The API request failed."
        error_state = build_empty_result_state()
        error_state["question"] = clean_question
        error_state["ai_answer"] = str(detail)
        error_state["has_error"] = True
        error_state["error_message"] = f"HTTP {response.status_code}"
        return error_state

    result_state = build_display_context(response_data)
    
    return result_state


def call_confirmation_api(
    thread_id: str,
    decision: str,
    api_base_url: str,
) -> dict[str, object]:
    """Resume a pending SQL modification after approval or rejection.

    Args:
        thread_id (str): memory identifier for the graph
        decision (str): Decision of the user
        api_base_url (str): Base URL of the fastAPI application

    Returns:
        dict[str, object]: A directory with values formatted to display the UI
    """

    user = st.session_state.get("user", {})
    headers = {
        "X-User-Sub": user.get("sub", ""),
        "X-User-Role": user.get("role", ""),
    }

    try:
        response = httpx.post(
            f"{api_base_url.rstrip('/')}/query/confirm",
            json={
                "thread_id": thread_id,
                "decision": decision,
            },
            headers=headers,
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        error_state = build_empty_result_state()
        error_state["ai_answer"] = f"FastAPI request failed: {exc}"
        error_state["has_error"] = True
        error_state["error_message"] = "API request failed"
        return error_state

    try:
        response_data = response.json()
    except ValueError:
        response_data = {}

    if response.is_error:
        detail = response_data.get("detail") or "The API request failed."
        error_state = build_empty_result_state()
        error_state["ai_answer"] = str(detail)
        error_state["has_error"] = True
        error_state["error_message"] = f"HTTP {response.status_code}"
        return error_state

    return build_display_context(response_data)


def _initialize_state() -> None:
    """Initialize the state of the page"""
    st.session_state.setdefault("question_input", DEFAULT_QUESTION)
    st.session_state.setdefault("result_state", build_empty_result_state())
    st.session_state.setdefault("selected_databases", [])
    st.session_state.setdefault("approval_dialog_open", False)


def _sync_selected_databases(databases: list[RegisteredDatabase]) -> list[str]:
    """Sync selected databases with session state

    Args:
        databases (list[RegisteredDatabase]): All the databases selected by the user

    Returns:
        list[str]: List of the databases selected
    """
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
    databases, error_message = _load_database_catalog()

    with st.container(border=True, key="database-catalog"):
        st.text("Available databases")

        if error_message:
            st.error(f"Database catalog configuration is invalid: {error_message}")
            return []

        if not databases:
            st.info("No databases are configured.")
            return []

        selected_databases = _sync_selected_databases(databases)
        selected_count = len(selected_databases)
        st.caption(f"{selected_count} of {len(databases)} databases active")

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
                                name
                                for name in selected_databases
                                if name != database_entry.name
                            ]
                        st.session_state["selected_databases"] = selected_databases
                        st.rerun()

        return st.session_state["selected_databases"]


def _render_question_panel() ->bool:
    """Render question panel

    Returns:
        bool: return if the question was submitted or not
    """
    with st.form("sql-copilot-form"):
        question = st.text_area(
            "Question",
            key="question_input",
            placeholder="Example: Which 5 artists have the most albums ?",
            height=180,
        )
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

        with submit_columns[0]:
            submitted = st.form_submit_button("Run analysis", use_container_width=True)

    st.session_state["execution_limit"] = execution_limit
    return submitted


def _render_result_summary(result_state: dict[str, object]) -> None:
    """Render result metrics:
    - Database used
    - rows returned
    - Execution limit
    - Result status

    Args:
        result_state (dict[str, object]): Output of the agent
    """
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
    _render_result_summary(result_state)
    rejected_modification: bool = (
        not result_state.get("execution_confirmed") and not result_state.get("execution_requested") and result_state.get("intent") == "modification"
    )
    answer_tab, sql_tab, data_tab = st.tabs(["Answer", "SQL", "Result Preview"])

    with answer_tab:
        if result_state.get("has_error"):
            st.error(str(result_state.get("ai_answer") or "The request failed."))
        elif rejected_modification:
            st.warning("The SQL modification was rejected. Check the SQL tab to see the draft query.")
        else:
            st.write(str(result_state.get("ai_answer") or ""))

    with sql_tab:
        st.code(
            str(result_state.get("sql_query") or "-- SQL will appear here"),
            language="sql",
        )

    with data_tab:
        query_result = result_state.get("query_result")
        if isinstance(query_result, dict) and query_result.get("rows"):
            st.dataframe(query_result["rows"], use_container_width=True, hide_index=True)
        else:
            st.info("No result rows were returned for this run.")


@st.dialog("Confirm SQL modification")
def _render_approval_dialog(
    result_state: dict[str, object],
    api_base_url: str,
) -> None:
    """Render the approval popup for modification queries."""

    interrupt_payload = result_state.get("interrupt")
    prompt_payload = interrupt_payload if isinstance(interrupt_payload, dict) else {}
    message = str(
        prompt_payload.get("question")
        or "Please review the SQL statement before continuing."
    )
    sql_draft = str(prompt_payload.get("draft") or result_state.get("sql_query") or "")

    st.write(message)
    if prompt_payload.get("request"):
        st.caption(f"Request: {prompt_payload['request']}")
    st.code(sql_draft or "-- SQL will appear here", language="sql")

    approve_col, reject_col = st.columns([4.5 ,1.5])
    with approve_col:
        if st.button("Approve", use_container_width=True, key="approve_sql_modification"):
            with st.spinner("Executing approved SQL..."):
                updated_state = call_confirmation_api(
                    str(result_state.get("thread_id") or ""),
                    "approve",
                    api_base_url,
                )
            st.session_state["result_state"] = updated_state
            st.session_state["approval_dialog_open"] = False
            st.rerun()
    with reject_col:
        if st.button(
            "Reject", 
            key="reject_sql_modification",
            width="stretch"
        ):
            with st.spinner("Rejecting SQL..."):
                updated_state = call_confirmation_api(
                    str(result_state.get("thread_id") or ""),
                    "reject",
                    api_base_url,
                )
            st.session_state["result_state"] = updated_state
            st.session_state["approval_dialog_open"] = False
            st.rerun()


def render_sql_copilot_page() -> None:
    """Render the SQL Copilot page or the login form when logged out."""
    if not is_authenticated_page():
        show_login_page()
        return

    render_logout_button()
    _initialize_state()
    api_base_url = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)

    _render_header()
    selected_databases = _render_database_catalog()
    submitted = _render_question_panel()

    question = st.session_state.get("question_input", "")
    execution_limit = st.session_state.get("execution_limit", 200)
    result_state = st.session_state["result_state"]

    if submitted:
        with st.spinner("Running query pipeline..."):
            result_state = call_api(
                question=question,
                execution_limit=execution_limit,
                api_base_url=api_base_url,
                selected_databases=selected_databases,
            )
            st.session_state["result_state"] = result_state
            st.session_state["approval_dialog_open"] = bool(
                result_state.get("execution_requested")
                and not result_state.get("execution_confirmed")
            )
            result_state = st.session_state["result_state"]

    if result_state.get("authorization_error"):
        st.error(str(result_state.get("authorization_error")))
        return

    pending_approval = bool(
        result_state.get("execution_requested")
        and not result_state.get("execution_confirmed")
    )

    if pending_approval:
        st.warning("This SQL modification is waiting for your approval.")
        if st.button("Review SQL command", key="review_sql_command_btn"):
            st.session_state["approval_dialog_open"] = True

    if pending_approval and st.session_state.get("approval_dialog_open"):
        _render_approval_dialog(result_state, api_base_url)

    has_result_content = any(
        [
            result_state.get("question"),
            result_state.get("sql_query"),
            result_state.get("ai_answer"),
        ]
    )
    if has_result_content and not pending_approval:
        with st.container(border=True, key="result"):
            _render_results(result_state)

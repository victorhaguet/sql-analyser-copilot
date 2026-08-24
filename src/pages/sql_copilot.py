"""SQL Copilot page rendering."""

from __future__ import annotations

from difflib import unified_diff
from pathlib import Path
from typing import cast

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
    selected_database: str | None,
) -> dict[str, object]:
    """Call the SQL copilot

    Args:
        question (str): User question
        execution_limit (int): Execution limit (max number of rows in the answer)
        api_base_url (str): Base URL of the fast API backend
        selected_database (str | None): Database to query

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
                "selected_database": selected_database,
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
    edited_sql: str | None = None,
    answers: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Resume an interrupted SQL copilot run: modification,
    or an agent clarification.

    Args:
        thread_id (str): memory identifier for the graph
        decision (str): Decision of the user: "approve" | "reject" | "answer" | "cancel"
        api_base_url (str): Base URL of the fastAPI application
        edited_sql (str | None): Optional edited SQL from user, applied on "approve"
        answers (list[dict[str, str]] | None): `{key, answer}` dicts, applied on "answer"

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
                "edited_sql": edited_sql,
                "answers": answers,
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
    st.session_state.setdefault("selected_database", None)


def _pending_interrupt_kind(result_state: dict[str, object]) -> str | None:
    """Return which interrupt is currently pausing the run, if any.

    Reads `interrupt["kind"]` — `"modification_approval"` or `"clarification"`
    — which every interrupt payload carries (see `SQLModificationValidatorNode`
    and `SQLAgentClarifyNode`), so the page can dispatch to the right dialog
    without guessing from which fields are present.

    Args:
        result_state (dict[str, object]): State of the current graph

    Returns:
        str | None: The interrupt kind, or None if nothing is pending.
    """
    if result_state.get("execution_confirmed"):
        return None

    interrupt_payload = result_state.get("interrupt")
    if isinstance(interrupt_payload, dict):
        kind = interrupt_payload.get("kind")
        return str(kind) if kind else "modification_approval"

    if result_state.get("execution_requested"):
        return "modification_approval"

    return None


def _sql_editor_key(thread_id: str, retry_count: int) -> str:
    """Return a fresh widget key for each regenerated SQL draft.

    Args:
        thread_id (str): Id of the thread
        retry_count (int): Number of the retry (max is 3)

    Returns:
        str: Widget key
    """

    return f"edited_sql_{thread_id}_{retry_count}"


def _clarification_answer_key(thread_id: str, clarification_round: int, question_key: str) -> str:
    """Return a fresh widget key for each clarification round's question inputs.

    Keys must include the round, not just the thread: the same thread can
    pause for `ask_user` multiple times, and Streamlit reuses stale widget
    values for a key it has already seen, mirroring the `_sql_editor_key` trap.

    Args:
        thread_id (str): Id of the thread
        clarification_round (int): Number of the clarification round
        question_key (str): The `key` field of the question being answered

    Returns:
        str: Widget key
    """

    return f"clarify_{thread_id}_{clarification_round}_{question_key}"


def _format_sql_diff(previous_sql: str, regenerated_sql: str) -> str:
    """Format failed and regenerated SQL for syntax-highlighted display.

    Args:
        previous_sql (str): SQL query that failed
        regenerated_sql (str): New SQL query (generated base on the failure of the previous one)

    Returns:
        str: Coding differences between the two SQL queries
    """

    if not previous_sql or previous_sql.strip() == regenerated_sql.strip():
        return "No textual SQL changes were produced."
    return "\n".join(
        unified_diff(
            previous_sql.splitlines(),
            regenerated_sql.splitlines(),
            fromfile="failed.sql",
            tofile="corrected.sql",
            lineterm="",
        )
    )


def _sync_selected_database(databases: list[RegisteredDatabase]) -> str | None:
    """Sync selected database with session state

    Args:
        databases (list[RegisteredDatabase]): All the databases available

    Returns:
        str | None: The selected database name
    """
    available_names = [database.name for database in databases]
    current_selection = st.session_state.get("selected_database")

    if current_selection is None or current_selection not in available_names:
        if available_names:
            st.session_state["selected_database"] = available_names[0]
            return available_names[0]
        return None

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


def _render_database_catalog() -> str | None:
    """Render databases catalog as single-select dropdown.

    Returns:
        str | None: The selected database name
    """
    databases, error_message = _load_database_catalog()

    with st.container(border=True, key="database-catalog"):
        st.text("Database selection")

        if error_message:
            st.error(f"Database catalog configuration is invalid: {error_message}")
            return None

        if not databases:
            st.error("No databases are configured. Please add at least one database to the data/ folder.")
            return None

        selected_database = _sync_selected_database(databases)
        
        if selected_database is None:
            st.error("No database available for selection.")
            return None

        database_options = {db.name: db for db in databases}
        
        selected_name = st.selectbox(
            "Choose the database to query:",
            options=list(database_options.keys()),
            index=list(database_options.keys()).index(selected_database) if selected_database else 0,
            key="database_dropdown",
        )
        
        if selected_name and selected_name != selected_database:
            st.session_state["selected_database"] = selected_name
            st.rerun()

        if selected_name and selected_name in database_options:
            db = database_options[selected_name]
            metadata = db.database.describe()
            table_count = len(metadata.get("tables") or [])
            with st.expander("Database details", expanded=False):
                st.write(f"**Description:** {db.description}")
                st.caption(f"{table_count} tables • {Path(str(metadata['database_path'])).name}")
                st.code("Tables:\n" + "\n".join(f"- {t}" for t in metadata.get("tables", [])), language="text")

        return selected_name


def _render_question_panel() -> bool:
    """Render question panel

    Returns:
        bool: return if the question was submitted or not
    """
    with st.form("sql-copilot-form"):
        st.text_area(
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
            submitted = st.form_submit_button("Run analysis", width="stretch")

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
    answer_tab, sql_tab, data_tab, agent_tab = st.tabs(
        ["Answer", "SQL", "Result Preview", "Agent steps"]
    )

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
            st.dataframe(query_result["rows"], width="stretch", hide_index=True)
        else:
            st.info("No result rows were returned for this run.")

    with agent_tab:
        _render_agent_steps(result_state)


def _render_agent_steps(result_state: dict[str, object]) -> None:
    """Render the SQL generation agent's tool-call trail.

    Reads `agent_tool_log`/`agent_iterations`/`probe_count`, which are derived
    server-side from the agent loop's `messages` transcript (see
    `core._derive_agent_tool_log`) rather than a separately maintained log.

    Args:
        result_state (dict[str, object]): State of the current graph run.
    """
    metric_columns = st.columns(2)
    metric_columns[0].metric(
        "Agent iterations", int(cast(int, result_state.get("agent_iterations")) or 0)
    )
    metric_columns[1].metric("Probes run", int(cast(int, result_state.get("probe_count")) or 0))

    agent_error = result_state.get("agent_error")
    if agent_error:
        st.error(str(agent_error))

    tool_log = result_state.get("agent_tool_log")
    if not isinstance(tool_log, list) or not tool_log:
        st.info("No agent tool calls were recorded for this run.")
        return

    for entry in tool_log:
        if not isinstance(entry, dict):
            continue
        with st.expander(
            f"Iteration {entry.get('iteration')} — {entry.get('tool')}", expanded=False
        ):
            st.write("**Arguments**")
            st.json(entry.get("arguments") or {})
            st.write("**Result**")
            result_value = entry.get("result")
            if result_value is None:
                st.caption("Awaiting result...")
            else:
                st.code(str(result_value), language="text", wrap_lines=True)


@st.dialog(
    "Review SQL modification",
    width="medium",
    dismissible=False,
    icon=":material/database:",
)
def _render_approval_dialog(
    result_state: dict[str, object],
    api_base_url: str,
) -> None:
    """Render the approval popup for modification queries.

    Args:
        result_state (dict[str, object]): State of the generated SQL query
        api_base_url (str): API to call
    """

    interrupt_payload = result_state.get("interrupt")
    prompt_payload = interrupt_payload if isinstance(interrupt_payload, dict) else {}
    message = str(
        prompt_payload.get("question")
        or "Please review the SQL statement before continuing."
    )
    sql_draft = str(prompt_payload.get("draft") or result_state.get("sql_query") or "")
    thread_id = str(result_state.get("thread_id") or "")
    retry_count = cast(int, result_state.get("retry_count") or 0)
    max_retries = result_state.get("max_retries") or 3
    previous_sql = str(
        prompt_payload.get("previous_sql") or result_state.get("previous_sql") or ""
    )
    previous_error = str(
        prompt_payload.get("previous_error")
        or result_state.get("last_execution_error")
        or ""
    )

    st.write(message)
    if prompt_payload.get("request"):
        st.caption(f"Request: {prompt_payload['request']}")

    agent_rationale = result_state.get("agent_rationale")
    if agent_rationale:
        st.markdown("**Why this SQL**")
        st.write(str(agent_rationale))

    clarification_answers = result_state.get("clarification_answers")
    if isinstance(clarification_answers, list) and clarification_answers:
        with st.expander("Answers already given", expanded=False):
            for entry in clarification_answers:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("question") or entry.get("key") or ""
                st.write(f"**{label}:** {entry.get('answer', '')}")

    if retry_count:
        st.info(
            f"Fallback generated corrected SQL — attempt {retry_count} of {max_retries}.",
            icon=":material/refresh:",
        )
        st.markdown("**Why it changed**")
        st.write(
            "The previous execution failed : "
        )
        if previous_error:
            st.error(f"{previous_error}")
        st.write(
            "The fallback rewrote the statement using the supplied database schema. Review the highlighted changes before approving it."
        )
        if previous_sql:
            with st.expander("See what changed", expanded=True):
                st.code(
                    _format_sql_diff(previous_sql, sql_draft),
                    language="diff",
                    wrap_lines=True,
                )

    st.markdown("**SQL proposed for approval**")
    st.code(sql_draft, language="sql", wrap_lines=True)

    edited_sql = sql_draft
    with st.expander("Edit proposed SQL", expanded=False):
        edited_sql = st.text_area( # type: ignore[call-overload]
            "SQL command",
            value=sql_draft,
            height=200,
            key=_sql_editor_key(thread_id, retry_count),
            label_visibility="collapsed",
        )

    with st.container(horizontal=True, horizontal_alignment="distribute"):
        if st.button("Approve", width="content", key="approve_sql_modification"):
            with st.spinner("Executing approved SQL..."):
                updated_state = call_confirmation_api(
                    thread_id,
                    "approve",
                    api_base_url,
                    edited_sql=edited_sql if edited_sql != sql_draft else None,
                )
            st.session_state["result_state"] = updated_state
            st.rerun(scope="app")
        if st.button(
            "Reject",
            key="reject_sql_modification",
            width="content",
        ):
            with st.spinner("Rejecting SQL..."):
                updated_state = call_confirmation_api(
                    thread_id,
                    "reject",
                    api_base_url,
                )
            st.session_state["result_state"] = updated_state
            st.rerun(scope="app")


@st.dialog(
    "The copilot needs more details",
    width="medium",
    dismissible=False,
    icon=":material/help:",
)
def _render_clarification_dialog(
    result_state: dict[str, object],
    api_base_url: str,
) -> None:
    """Render the clarification popup for pending `ask_user` questions.

    Args:
        result_state (dict[str, object]): State of the current graph run
        api_base_url (str): API to call
    """

    interrupt_payload = result_state.get("interrupt")
    prompt_payload = interrupt_payload if isinstance(interrupt_payload, dict) else {}
    questions = prompt_payload.get("questions") or []
    thread_id = str(result_state.get("thread_id") or "")
    clarification_round = cast(int, result_state.get("clarification_rounds") or 0)

    st.write(
        "The copilot needs a bit more information before generating the SQL statement."
    )

    answers: dict[str, str] = {}
    for question in questions:
        if not isinstance(question, dict):
            continue
        question_key = str(question.get("key") or "")
        label = str(question.get("question") or question_key)
        why = question.get("why")
        default_value = str(question.get("suggested_default") or "")

        if why:
            st.caption(str(why))
        answers[question_key] = st.text_input(
            label,
            value=default_value,
            key=_clarification_answer_key(thread_id, clarification_round, question_key),
        )

    with st.container(horizontal=True, horizontal_alignment="distribute"):
        if st.button("Send answer", width="content", key="send_clarification_answers"):
            with st.spinner("Sending answers..."):
                updated_state = call_confirmation_api(
                    thread_id,
                    "answer",
                    api_base_url,
                    answers=[
                        {"key": question_key, "answer": answer}
                        for question_key, answer in answers.items()
                    ],
                )
            st.session_state["result_state"] = updated_state
            st.rerun(scope="app")
        if st.button("Cancel", key="cancel_clarification", width="content"):
            with st.spinner("Cancelling..."):
                updated_state = call_confirmation_api(
                    thread_id,
                    "cancel",
                    api_base_url,
                )
            st.session_state["result_state"] = updated_state
            st.rerun(scope="app")


def render_sql_copilot_page() -> None:
    """Render the SQL Copilot page or the login form when logged out."""
    if not is_authenticated_page():
        show_login_page()
        return

    render_logout_button()
    _initialize_state()
    api_base_url = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)

    _render_header()
    selected_database = _render_database_catalog()
    submitted = _render_question_panel()

    question = st.session_state.get("question_input", "")
    execution_limit = st.session_state.get("execution_limit", 200)
    result_state = st.session_state["result_state"]

    if submitted:
        if not selected_database:
            result_state = build_empty_result_state()
            result_state["ai_answer"] = "No database selected. Please choose a database from the dropdown."
            result_state["has_error"] = True
            result_state["error_message"] = "Missing database selection"
            st.session_state["result_state"] = result_state
        else:
            with st.spinner("Running query pipeline..."):
                result_state = call_api(
                    question=question,
                    execution_limit=execution_limit,
                    api_base_url=api_base_url,
                    selected_database=selected_database,
                )
                st.session_state["result_state"] = result_state
    result_state = st.session_state["result_state"]

    if result_state.get("authorization_error"):
        st.error(str(result_state.get("authorization_error")))
        return

    pending_interrupt_kind = _pending_interrupt_kind(result_state)

    if pending_interrupt_kind == "clarification":
        st.warning("The copilot needs more details before generating the SQL statement.")
        _render_clarification_dialog(result_state, api_base_url)
    elif pending_interrupt_kind == "modification_approval":
        st.warning("This SQL modification is waiting for your approval.")
        _render_approval_dialog(result_state, api_base_url)

    has_result_content = any(
        [
            result_state.get("question"),
            result_state.get("sql_query"),
            result_state.get("ai_answer"),
        ]
    )
    if has_result_content and not pending_interrupt_kind:
        with st.container(border=True, key="result"):
            _render_results(result_state)

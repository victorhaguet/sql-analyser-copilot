"""Core business logic for the SQL copilot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.types import Command, Interrupt
from graph import SQLAgentTraceStep, build_sql_agent_graph, stream_sql_agent_execution
from state import SQLAgentState
from tools.database import QueryResult, SQLiteDatabase
from tools.sql_safety import SQLSafetyValidator
from tracing import write_trace_log
from utils.nodes import LLM, ToolCallingChatModel

INTERRUPT_KEY = "__interrupt__"

# Backstop for a router bug producing an unbounded loop (D5), env-tunable via
# SQL_AGENT_RECURSION_LIMIT.
DEFAULT_AGENT_RECURSION_LIMIT = int(os.getenv("SQL_AGENT_RECURSION_LIMIT", "40"))


@dataclass
class PendingApprovalSession:
    """In-memory data required to resume an interrupted modification run."""

    graph: Any
    question: str
    user_sub: str
    trace_log_path: Path | None = None


def _serialize_user(user: dict[str, Any]) -> dict[str, Any]:
    """Convert user dictionary to UserResponse-like dict.
    Args:
        user: User dictionary from database.
    Returns:
        Dictionary with user information suitable for API responses.
    """
    return {
        "sub": user["sub"],
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "is_active": user["is_active"],
        "created_at": user["created_at"],
    }


def _serialize_query_result(result: QueryResult | None) -> dict[str, Any] | None:
    """Convert the internal query result dataclass into an API payload.

    Args:
        result: The internal QueryResult instance to serialize.

    Returns:
        A dictionary suitable for API responses, or None if the input is None.
    """
    if result is None:
        return None
    return {
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
    }


def _serialize_trace(trace: list[SQLAgentTraceStep]) -> list[dict[str, Any]]:
    """Serialize trace steps for JSON storage.

    Args:
        trace: List of trace steps

    Returns:
        List of serialized trace steps
    """
    serialized = []
    for step in trace:
        serialized_step: dict[str, Any] = {
            "node": step["node"],
            "update": step["update"],
            "outcome": step["outcome"],
        }
        serialized.append(serialized_step)
    return serialized


def _serialize_messages(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Convert LangChain message objects into JSON-friendly dicts.

    Message objects (`SystemMessage`, `HumanMessage`, `AIMessage`,
    `ToolMessage`) are not directly JSON-serializable by FastAPI, so `/query`
    would 500 the first time the agent runs without this.

    Args:
        messages: The agent loop transcript (`state["messages"]`).

    Returns:
        list[dict[str, Any]]: `{role, content, tool_calls, tool_call_id}` per message.
    """
    serialized: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        serialized.append(
            {
                "role": getattr(message, "type", message.__class__.__name__.lower()),
                "content": message.content,
                "tool_calls": (
                    [
                        {"name": call["name"], "args": call.get("args", {}), "id": call.get("id")}
                        for call in tool_calls
                    ]
                    if tool_calls
                    else None
                ),
                "tool_call_id": getattr(message, "tool_call_id", None),
            }
        )
    return serialized


def _derive_agent_tool_log(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Derive the UI-facing tool-call log from the transcript, not a parallel list.

    `messages` is the single source of truth for the agent loop; the
    "Agent steps" UI reads the log this produces rather than a separately
    maintained field.

    Args:
        messages: The agent loop transcript (`state["messages"]`).

    Returns:
        list[dict[str, Any]]: `{iteration, tool, arguments, result}` per tool
        call, in the order the calls were made. `result` is `None` until the
        matching `ToolMessage` is found (e.g. a call still awaiting `interrupt()` resume).
    """
    tool_log: list[dict[str, Any]] = []
    pending_by_call_id: dict[str, dict[str, Any]] = {}
    iteration = 0
    for message in messages:
        if isinstance(message, AIMessage):
            iteration += 1
            for tool_call in message.tool_calls:
                entry: dict[str, Any] = {
                    "iteration": iteration,
                    "tool": tool_call["name"],
                    "arguments": tool_call.get("args", {}),
                    "result": None,
                }
                tool_log.append(entry)
                call_id = tool_call.get("id")
                if call_id is not None:
                    pending_by_call_id[call_id] = entry
        elif isinstance(message, ToolMessage):
            pending_entry = pending_by_call_id.get(message.tool_call_id)
            if pending_entry is not None:
                pending_entry["result"] = message.content
                del pending_by_call_id[message.tool_call_id]
    return tool_log


def _serialize_state(state: SQLAgentState, question: str) -> dict[str, Any]:
    """Convert the internal graph state into the public API response model.

    Args:
        state: The internal state dictionary from the SQL copilot graph.
        question: The original natural language question.

    Returns:
        A dictionary suitable for API responses.
    """
    messages = state.get("messages") or []
    return {
        "question": question,
        "schema_overview": state.get("schema_overview"),
        "selected_database": state.get("selected_database"),
        "intent": state.get("intent"),
        "intent_error": state.get("intent_error"),
        "generated_sql": state.get("generated_sql"),
        "validated_sql": state.get("validated_sql"),
        "sql_validation_error": state.get("sql_validation_error"),
        "authorization_error": state.get("authorization_error"),
        "is_authorized": state.get("is_authorized"),
        "query_result": _serialize_query_result(state.get("query_result")),
        "execution_error": state.get("execution_error"),
        "execution_requested": state.get("execution_requested", False),
        "execution_confirmed": state.get("execution_confirmed", False),
        "thread_id": state.get("thread_id"),
        "interrupt": state.get("interrupt"),
        "analysis": state.get("analysis"),
        "metadata": state.get("metadata") or {},
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 3),
        "last_execution_error": state.get("last_execution_error"),
        "regeneration_error": state.get("regeneration_error"),
        "previous_sql": state.get("previous_sql"),
        "regeneration_explanation": state.get("regeneration_explanation"),
        "messages": _serialize_messages(messages),
        "agent_status": state.get("agent_status"),
        "agent_iterations": state.get("agent_iterations", 0),
        "max_agent_iterations": state.get("max_agent_iterations"),
        "probe_count": state.get("probe_count", 0),
        "max_probes": state.get("max_probes"),
        "clarification_rounds": state.get("clarification_rounds", 0),
        "max_clarifications": state.get("max_clarifications"),
        "clarification_answers": state.get("clarification_answers") or [],
        "agent_rationale": state.get("agent_rationale"),
        "agent_error": state.get("agent_error"),
        "agent_tool_log": _derive_agent_tool_log(messages),
    }


def _build_thread_config(
    thread_id: str, *, recursion_limit: int = DEFAULT_AGENT_RECURSION_LIMIT
) -> dict[str, Any]:
    """Create the LangGraph config payload for a resumable thread.

    `recursion_limit` is a backstop independent of the router's own budget
    checks: even a router bug cannot produce an unbounded loop.

    Args:
        thread_id (str): current thread used by the agent
        recursion_limit (int, optional): recursion limit. Defaults to DEFAULT_AGENT_RECURSION_LIMIT.

    Returns:
        dict[str, Any]: Info of the thread
    """

    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }


def _serialize_interrupt(interrupt: Interrupt) -> dict[str, Any]:
    """Convert a LangGraph interrupt object into a JSON-friendly payload.

    Args:
        interrupt (Interrupt): Interruption containing dictionnary of the interrupt state

    Returns:
        dict[str, Any]: Interrupt value extracted
    """

    if isinstance(interrupt.value, dict):
        return cast(dict[str, Any], interrupt.value)
    return {"value": interrupt.value}


def _extract_interrupt_payload(result: Any) -> dict[str, Any] | None:
    """Extract the first interrupt payload from a graph invoke result.

    Args:
        result (Any): Output of the graph

    Returns:
        dict[str, Any] | None: value of the interrupt state if there is an interrupt
    """

    if not isinstance(result, dict):
        return None
    raw_interrupts = result.get(INTERRUPT_KEY)
    if not isinstance(raw_interrupts, list) or not raw_interrupts:
        return None
    first_interrupt = raw_interrupts[0]
    if not isinstance(first_interrupt, Interrupt):
        return None
    return _serialize_interrupt(first_interrupt)


def _snapshot_to_state(snapshot: Any, fallback: SQLAgentState) -> SQLAgentState:
    """Read the current LangGraph snapshot values as a mutable state dictionary.

    Args:
        snapshot (Any): Output of the graph (snapshot) to transform
        fallback (SQLAgentState): Default fallback value if the tranformation doesn't work

    Returns:
        SQLAgentState: Clean SQLAgentState object
    """

    values = getattr(snapshot, "values", None)
    if not isinstance(values, dict):
        return fallback
    return cast(SQLAgentState, dict(values))


def _snapshot_interrupt_payload(snapshot: Any) -> dict[str, Any] | None:
    """Read the first interrupt payload from a state snapshot, if any.

    Args:
        snapshot (Any): Snapshot after an interrupt call

    Returns:
        dict[str, Any] | None: Content of the snapshot
    """

    interrupts = getattr(snapshot, "interrupts", ())
    if not isinstance(interrupts, (list, tuple)) or not interrupts:
        return None
    first_interrupt = interrupts[0]
    if not isinstance(first_interrupt, Interrupt):
        return None
    return _serialize_interrupt(first_interrupt)


def _build_pending_approval_state(
    state: SQLAgentState,
    *,
    thread_id: str,
    interrupt_payload: dict[str, Any],
) -> SQLAgentState:
    """Annotate a graph state with approval metadata for the API response."""

    pending_state = dict(state)
    pending_state["execution_requested"] = True
    pending_state["execution_confirmed"] = False
    pending_state["thread_id"] = thread_id
    pending_state["interrupt"] = interrupt_payload
    return cast(SQLAgentState, pending_state)


def _attach_trace_metadata(
    state: SQLAgentState,
    *,
    question: str,
    trace: list[SQLAgentTraceStep],
    trace_log_dir: str | Path | None,
    existing_log_path: Path | str | dict | None = None,
) -> tuple[SQLAgentState, Path]:
    """Persist and attach the normalized execution trace metadata.

    Args:
        state: The current SQL agent state.
        question: The original natural language question.
        trace: The execution trace steps.
        trace_log_dir: Directory to save trace logs.
        existing_log_path: Optional existing log path for merging.

    Returns:
        Tuple of (updated_state, log_path).
    """

    log_path = write_trace_log(question, trace, trace_log_dir, existing_log_path=existing_log_path)
    metadata = dict(state.get("metadata") or {})
    metadata["execution_trace"] = _serialize_trace(trace)
    metadata["trace_log_path"] = str(log_path)
    state["metadata"] = metadata
    return state, log_path


def start_question(
    question: str,
    sql_generator_model: LLM,
    agent_model: ToolCallingChatModel,
    analyst_model: LLM | None = None,
    selector_model: LLM | None = None,
    intent_model: LLM | None = None,
    selected_database: SQLiteDatabase | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
    include_trace: bool = False,
    trace_log_dir: str | Path | None = None,
    user_role: str = "readonly",
    user_sub: str = "",
    pending_approval_sessions: dict[str, PendingApprovalSession] | None = None,
) -> SQLAgentState:
    """Start a SQL copilot run and pause if a modification requires approval."""

    thread_id = uuid4().hex
    config = _build_thread_config(thread_id)
    graph = build_sql_agent_graph(
        sql_generator_model=sql_generator_model,
        agent_model=agent_model,
        analyst_model=analyst_model,
        selector_model=selector_model,
        intent_model=intent_model,
        selected_database=selected_database,
        validator=validator,
        execution_limit=execution_limit,
    )
    initial_state: SQLAgentState = {
        "question": question,
        "user_role": user_role,
        "retry_count": 0,
        "max_retries": 3,
        "last_execution_error": None,
    }
    if selected_database is not None:
        initial_state["selected_database"] = selected_database.name if hasattr(selected_database, 'name') else "default"

    if include_trace:
        trace = list(
            stream_sql_agent_execution(
                graph,
                initial_state,
                config=config,
            )
        )
        snapshot = graph.get_state(config)
        result_state = (
            cast(SQLAgentState, trace[-1]["state"])
            if trace
            else _snapshot_to_state(snapshot, initial_state)
        )
        interrupt_payload = _snapshot_interrupt_payload(snapshot)
        if interrupt_payload is not None:
            result_state = _build_pending_approval_state(
                _snapshot_to_state(snapshot, result_state),
                thread_id=thread_id,
                interrupt_payload=interrupt_payload,
            )
            if pending_approval_sessions is not None:
                pending_approval_sessions[thread_id] = PendingApprovalSession(
                    graph=graph,
                    question=question,
                    user_sub=user_sub,
                    trace_log_path=None,
                )
        result_state, log_path = _attach_trace_metadata(
            result_state,
            question=question,
            trace=trace,
            trace_log_dir=trace_log_dir,
        )
        if interrupt_payload is not None and pending_approval_sessions is not None:
            pending_approval_sessions[thread_id].trace_log_path = log_path
        return result_state

    result = graph.invoke(initial_state, config=config)
    interrupt_payload = _extract_interrupt_payload(result)
    if interrupt_payload is not None:
        snapshot = graph.get_state(config)
        result_state = _build_pending_approval_state(
            _snapshot_to_state(snapshot, initial_state),
            thread_id=thread_id,
            interrupt_payload=interrupt_payload,
        )
        if pending_approval_sessions is not None:
            pending_approval_sessions[thread_id] = PendingApprovalSession(
                graph=graph,
                question=question,
                user_sub=user_sub,
            )
        return result_state

    snapshot = graph.get_state(config)
    return _snapshot_to_state(snapshot, initial_state)


def resume_question(
    thread_id: str,
    decision: str,
    *,
    edited_sql: str | None = None,
    answers: list[dict[str, Any]] | None = None,
    pending_approval_sessions: dict[str, PendingApprovalSession],
    include_trace: bool = False,
    trace_log_dir: str | Path | None = None,
    user_sub: str = "",
) -> SQLAgentState:
    """Resume an interrupted SQL copilot run.

    Args:
        thread_id (str): Memory ID of the graph
        decision (str): Decision of the user: "approve" | "reject" | "answer" | "cancel"
        edited_sql (str | None): Optional SQL modified by the user, applied on "approve"
        answers (list[dict[str, Any]] | None): `{key, answer}` dicts, applied on "answer"
        pending_approval_sessions (dict[str, PendingApprovalSession]): Pending session
        include_trace (bool, optional): When true, attach a normalized per-step execution trace to `metadata["execution_trace"]`. Defaults to False.
        trace_log_dir (str | Path | None, optional): directory path where to save the log files. Defaults to None.
        user_sub (str, optional): ID of the authentificated user. Defaults to "".

    Raises:
        ValueError: No pending session found
        ValueError: If the user ID is different from the user ID of the session

    Returns:
        SQLAgentState: State of the agent after the modification of the database
    """

    session = pending_approval_sessions.get(thread_id)
    if session is None:
        raise ValueError("No pending approval session found for this thread.")
    if session.user_sub and session.user_sub != user_sub:
        raise ValueError("No pending approval found for this user.")

    config = _build_thread_config(thread_id)

    if decision in ("answer", "cancel"):
        resume_command: Command = Command(
            resume={"decision": decision, "answers": answers or []}
        )
    elif edited_sql and decision == "approve":
        resume_command = Command(resume=decision, update={"generated_sql": edited_sql})
    else:
        resume_command = Command(resume=decision)

    if include_trace:
        trace = list(
            stream_sql_agent_execution(
                session.graph,
                resume_command,
                config=config,
            )
        )
        snapshot = session.graph.get_state(config)
        result_state = (
            cast(SQLAgentState, trace[-1]["state"])
            if trace
            else _snapshot_to_state(snapshot, {"question": session.question})
        )
        interrupt_payload = _snapshot_interrupt_payload(snapshot)
        if interrupt_payload is not None:
            result_state = _build_pending_approval_state(
                _snapshot_to_state(snapshot, result_state),
                thread_id=thread_id,
                interrupt_payload=interrupt_payload,
            )
        else:
            pending_approval_sessions.pop(thread_id, None)
            result_state.pop("thread_id", None)
            result_state.pop("interrupt", None)
        result_state, _ = _attach_trace_metadata(
            result_state,
            question=session.question,
            trace=trace,
            trace_log_dir=trace_log_dir,
            existing_log_path=session.trace_log_path,
        )
        return result_state

    result = session.graph.invoke(resume_command, config=config)
    interrupt_payload = _extract_interrupt_payload(result)
    snapshot = session.graph.get_state(config)
    if interrupt_payload is not None:
        return _build_pending_approval_state(
            _snapshot_to_state(snapshot, {"question": session.question}),
            thread_id=thread_id,
            interrupt_payload=interrupt_payload,
        )

    pending_approval_sessions.pop(thread_id, None)
    result_state = _snapshot_to_state(snapshot, {"question": session.question})
    result_state.pop("thread_id", None)
    result_state.pop("interrupt", None)
    return result_state


def answer_question(
    question: str,
    sql_generator_model: LLM,
    agent_model: ToolCallingChatModel,
    analyst_model: LLM | None = None,
    selector_model: LLM | None = None,
    intent_model: LLM | None = None,
    selected_database: SQLiteDatabase | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
    include_trace: bool = False,
    trace_log_dir: str | Path | None = None,
    user_role: str = "readonly",
) -> SQLAgentState:
    """Run the SQL copilot graph.

    Args:
        question: The natural language question to answer.
        sql_generator_model: Default text model, used as the intent classifier's fallback.
        agent_model: bind_tools-capable model driving the SQL generation agent loop.
        analyst_model: The language model to use for result analysis (optional).
        selector_model: The language model to use for database selection (optional).
        intent_model: The language model to use for intent classification (optional).
        selected_database: The SQLiteDatabase instance to query (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).
        include_trace: When true, attach a normalized per-step execution trace to `metadata["execution_trace"]`.
        trace_log_dir: Optional directory where per-run trace logs are persisted. Defaults to `logs/`.
        user_role: The role of the authenticated user (default: readonly).

    Returns:
        An instance of SQLAgentState representing the state of the SQL copilot graph after processing the question.
    """
    return start_question(
        question=question,
        sql_generator_model=sql_generator_model,
        agent_model=agent_model,
        analyst_model=analyst_model,
        selector_model=selector_model,
        intent_model=intent_model,
        selected_database=selected_database,
        validator=validator,
        execution_limit=execution_limit,
        include_trace=include_trace,
        trace_log_dir=trace_log_dir,
        user_role=user_role,
        pending_approval_sessions=None,
    )


def stream_question(
    question: str,
    sql_generator_model: LLM,
    agent_model: ToolCallingChatModel,
    analyst_model: LLM | None = None,
    selector_model: LLM | None = None,
    intent_model: LLM | None = None,
    selected_database: SQLiteDatabase | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
    user_role: str = "readonly",
) -> Iterator[SQLAgentTraceStep]:
    """Stream normalized execution steps for a single question.

    This is a graph-specific debugging helper over LangGraph's raw `stream()`
    output. It exposes node names, state deltas, and a compact outcome label.

    Args:
        question: The natural language question to answer.
        sql_generator_model: Default text model, used as the intent classifier's fallback.
        agent_model: bind_tools-capable model driving the SQL generation agent loop.
        analyst_model: The language model to use for result analysis (optional).
        selector_model: The language model to use for database selection (optional).
        intent_model: The language model to use for intent classification (optional).
        selected_database: The SQLiteDatabase instance to query (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).
        user_role: The role of the authenticated user (default: readonly).

    Returns:
        An iterator of SQLAgentTraceStep dictionaries representing the execution trace of the graph.
    """
    graph = build_sql_agent_graph(
        sql_generator_model=sql_generator_model,
        agent_model=agent_model,
        analyst_model=analyst_model,
        selector_model=selector_model,
        intent_model=intent_model,
        selected_database=selected_database,
        validator=validator,
        execution_limit=execution_limit,
    )
    initial_state: SQLAgentState = {
        "question": question,
        "user_role": user_role,
    }
    if selected_database is not None:
        initial_state["selected_database"] = "default"
    return stream_sql_agent_execution(graph, initial_state)

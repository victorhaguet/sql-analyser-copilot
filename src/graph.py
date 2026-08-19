"""LangGraph orchestration for the SQL copilot agent."""

from __future__ import annotations

from typing import Any, Iterator, Literal, Protocol, TypedDict, cast
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt

from nodes.database_checker import DatabaseCheckerNode
from nodes.intent_classifier import IntentClassifierNode
from nodes.result_analyst import ResultAnalystNode
from nodes.role_authorizer import RoleAuthorizerNode
from nodes.sql_agent import (
    SQLAgentClarifyNode,
    SQLAgentFinalizeNode,
    SQLAgentLLMNode,
    SQLAgentToolsNode,
)
from nodes.sql_executor import SQLExecutorNode
from nodes.sql_validator import SQLValidatorNode
from nodes.sql_modification_validator import SQLModificationValidatorNode
from state import SQLAgentState
from tools.agent_tools import AgentToolLimits, build_agent_tools
from tools.database import SQLiteDatabase
from tools.sql_safety import SQLSafetyValidator
from utils.nodes import LLM, ToolCallingChatModel

SQLAgentStateUpdate = SQLAgentState

TraceStreamMode = Literal["updates", "values"]

NODE_DATABASE_CHECKER = "database_checker"
NODE_INTENT_CLASSIFIER = "intent_classifier"
NODE_ROLE_AUTHORIZER = "role_authorizer"
NODE_SQL_AGENT_LLM = "sql_agent_llm"
NODE_SQL_AGENT_TOOLS = "sql_agent_tools"
NODE_SQL_AGENT_CLARIFY = "sql_agent_clarify"
NODE_SQL_AGENT_FINALIZE = "sql_agent_finalize"
NODE_SQL_VALIDATOR = "sql_validator"
NODE_SQL_EXECUTOR = "sql_executor"
NODE_RESULT_ANALYST = "result_analyst"
NODE_MODIFICATION_VALIDATOR = "modification_validator"
ROUTE_ABORT = "abort"
INTERRUPT_EVENT = "__interrupt__"

# Intent-scoped iteration budgets (D5). Placeholders; Step 10 of the agentic
# SQL generation plan wires these to env vars (SQL_AGENT_MAX_ITERATIONS_QUERY /
# SQL_AGENT_MAX_ITERATIONS_MODIFICATION).
DEFAULT_MAX_AGENT_ITERATIONS_QUERY = 4
DEFAULT_MAX_AGENT_ITERATIONS_MODIFICATION = 8


class CompiledSQLAgentGraph(Protocol):
    """Protocol for the compiled graph methods used by the trace helpers."""

    def invoke(
        self,
        input_: SQLAgentState | Command[Any] | None,
        config: Any | None = None,
    ) -> dict[str, Any] | Any:
        """Run the graph and return the final state."""

    def stream(
        self,
        state: SQLAgentState | Command[Any],
        config: Any | None = None,
        *,
        stream_mode: TraceStreamMode = "updates",
    ) -> Iterator[object]:
        """Yield raw LangGraph stream events."""

    def get_state(self, config: Any) -> Any:
        """Return the current graph state snapshot for a thread."""


class SQLAgentTraceStep(TypedDict):
    """Normalized view of a single graph step."""

    node: str
    update: SQLAgentStateUpdate
    state: SQLAgentState
    outcome: str


def _route_after_database_check(state: SQLAgentState) -> str:
    return ROUTE_ABORT if state.get("execution_error") else NODE_INTENT_CLASSIFIER


def _route_after_authorization(state: SQLAgentState) -> str:
    return ROUTE_ABORT if not state.get("is_authorized") else NODE_SQL_AGENT_LLM


def _default_max_agent_iterations(state: SQLAgentState) -> int:
    """Intent-scoped default iteration budget (D5): reads must stay fast."""
    return (
        DEFAULT_MAX_AGENT_ITERATIONS_MODIFICATION
        if state.get("intent") == "modification"
        else DEFAULT_MAX_AGENT_ITERATIONS_QUERY
    )


def _resolve_max_agent_iterations(state: SQLAgentState) -> int:
    """The state's explicit budget if set, else the intent-scoped default."""
    explicit = state.get("max_agent_iterations")
    return explicit if explicit is not None else _default_max_agent_iterations(state)


def _route_after_agent_llm(state: SQLAgentState) -> str:
    """should_continue, extended with budget enforcement and the clarify split (D3)."""
    if state.get("agent_iterations", 0) >= _resolve_max_agent_iterations(state):
        return ROUTE_ABORT

    messages = state.get("messages") or []
    last_message = messages[-1] if messages else None
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return NODE_SQL_AGENT_FINALIZE
    if len(tool_calls) == 1 and tool_calls[0]["name"] == "ask_user":
        return NODE_SQL_AGENT_CLARIFY
    return NODE_SQL_AGENT_TOOLS


def _route_after_clarify(state: SQLAgentState) -> str:
    return ROUTE_ABORT if state.get("agent_status") == "cancelled" else NODE_SQL_AGENT_LLM


def _route_after_finalize(state: SQLAgentState) -> str:
    """Route the finished SQL to the matching safety gate, or abort on failure."""
    if state.get("agent_status") == "failed":
        return ROUTE_ABORT
    return NODE_SQL_VALIDATOR if state["intent"] == "query" else NODE_MODIFICATION_VALIDATOR


def _route_after_executor(state: SQLAgentState) -> str:
    if state.get("execution_error"):
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        if retry_count < max_retries:
            return NODE_SQL_AGENT_LLM
        return ROUTE_ABORT
    return NODE_RESULT_ANALYST


def _route_from_intent_classifier(state: SQLAgentState) -> str:
    return NODE_ROLE_AUTHORIZER if state["intent"] == "modification" else NODE_SQL_AGENT_LLM


def _route_from_sql_validator(state: SQLAgentState) -> str:
    if state.get("sql_validation_error"):
        return ROUTE_ABORT
    return NODE_SQL_EXECUTOR


def _route_from_modification_validator(state: SQLAgentState) -> str:
    if state.get("execution_confirmed"):
        return NODE_SQL_EXECUTOR
    return ROUTE_ABORT


def _summarize_step_outcome(
    update: SQLAgentState, state: SQLAgentState | None = None
) -> str:
    """Summarize the current state of the graph

    Args:
        update (SQLAgentStateUpdate): Graph state
        state (SQLAgentState | None): The full state after this update was merged
            in, used only to evaluate budget thresholds (agent_budget_exhausted).
            Optional so existing single-argument callers keep working.

    Returns:
        str: Current state of the graph
    """
    metadata = update.get("metadata") or {}
    if update.get("authorization_error"):
        return "authorization_failed"
    if update.get("interrupt"):
        interrupt_payload = update["interrupt"]
        if isinstance(interrupt_payload, dict) and interrupt_payload.get("kind") == "clarification":
            return "awaiting_clarification"
        return "execution_pending_approval"
    if update.get("agent_status") == "repairing":
        return "agent_repairing"
    if update.get("regeneration_error"):
        return "regeneration_failed"
    if update.get("sql_validation_error"):
        return "validation_failed"
    if update.get("execution_error"):
        if metadata.get("database_selection_ambiguous"):
            return "database_selection_ambiguous"
        if metadata.get("database_selection_failed"):
            return "database_selection_failed"
        if metadata.get("intent_failed"):
            return "intent_failed"
        return "execution_failed"
    if "query_result" in update:
        return "query_executed"
    if "analysis" in update:
        return "analysis_ready"
    if "generated_sql" in update:
        return "sql_generated"
    if "probe_count" in update:
        return "agent_tool_call"
    if "agent_iterations" in update:
        if state is not None and state.get("agent_iterations", 0) >= _resolve_max_agent_iterations(state):
            return "agent_budget_exhausted"
        return "agent_thinking"
    if "is_authorized" in update:
        return "User role authorisation"
    if "intent" in update:
        return "Intention classified"
    if "selected_database" in update:
        return "database_checked"
    return "updated"


def _merge_state_update(
    current_state: SQLAgentState, update: dict[str, Any]
) -> dict[str, Any]:
    """Merge a node update into state, concatenating reducer fields (messages)
    instead of overwriting them.

    A naive `{**current_state, **update}` would replace `messages` on every
    step, so the trace's view of state would show only the newest message(s).
    `messages` is `SQLAgentState`'s only reducer field (Step 4); this special-cases
    it to append, mirroring `add_messages`'s own behaviour.

    Args:
        current_state: State before this step's update.
        update: The raw update returned by this step's node.

    Returns:
        dict[str, Any]: The merged state.
    """
    merged: dict[str, Any] = {**current_state, **update}
    if "messages" in update:
        existing_messages = current_state.get("messages") or []
        merged["messages"] = [*existing_messages, *(update.get("messages") or [])]
    return merged


def _normalize_stream_event(
    event: object,
    current_state: SQLAgentState,
    stream_mode: TraceStreamMode,
) -> tuple[str, SQLAgentStateUpdate, SQLAgentState]:
    """Converts raw LangGraph stream events into a normalized format

    Args:
        event (object): Raw event from LangGraph's stream() output.
        current_state (SQLAgentState): Current state of the graph
        stream_mode (TraceStreamMode): Either updates (incremental deltas) or "values" (full state)

    Raises:
        TypeError: If the event isn't a filled dict
        TypeError: If the node name isn't a string and the update isn't a dict
        TypeError: If the event usn't a dict (for stream_mode = values)

    Returns:
        tuple[str, SQLAgentStateUpdate, SQLAgentState]: Normalize state of the graph
    """
    if isinstance(event, dict) and INTERRUPT_EVENT in event:
        interrupts = event[INTERRUPT_EVENT]
        if not isinstance(interrupts, (list, tuple)) or not interrupts:
            raise TypeError(f"Unexpected interrupt event: {event!r}")
        first_interrupt = interrupts[0]
        if not isinstance(first_interrupt, Interrupt):
            raise TypeError(f"Unexpected interrupt event: {event!r}")
        payload = first_interrupt.value
        serialized_payload = (
            cast(dict[str, Any], payload)
            if isinstance(payload, dict)
            else {"value": payload}
        )
        update: SQLAgentState = {
            "interrupt": serialized_payload,
            "execution_requested": True,
            "execution_confirmed": False,
        }
        merged_state = _merge_state_update(current_state, update)
        return INTERRUPT_EVENT, update, cast(SQLAgentState, merged_state)

    if stream_mode == "updates":
        if not isinstance(event, dict) or len(event) != 1:
            raise TypeError(f"Unexpected graph update event: {event!r}")
        node_name, update = next(iter(event.items()))
        if not isinstance(node_name, str) or not isinstance(update, dict):
            raise TypeError(f"Unexpected graph update event: {event!r}")
        merged_state = _merge_state_update(current_state, update)
        return node_name, cast(SQLAgentState, update), cast(SQLAgentState, merged_state)

    if not isinstance(event, dict):
        raise TypeError(f"Unexpected graph values event: {event!r}")
    return "state", cast(SQLAgentState, event), cast(SQLAgentState, event)


def stream_sql_agent_execution(
    graph: CompiledSQLAgentGraph,
    initial_state: SQLAgentState | Command[Any],
    *,
    config: Any | None = None,
    stream_mode: TraceStreamMode = "updates",
) -> Iterator[SQLAgentTraceStep]:
    """Stream the SQL agent execution

    Args:
        graph (CompiledSQLAgentGraph): Compiled graph
        initial_state (SQLAgentState): Initial state of the graph
        stream_mode (TraceStreamMode, optional): Current state of the graph. Defaults to "updates".

    Yields:
        SQLAgentTraceStep: Current state of the graph
    """
    resolved_config = config or {"configurable": {"thread_id": uuid4().hex}}

    if isinstance(initial_state, dict):
        state = cast(SQLAgentState, initial_state)
    elif resolved_config is not None:
        snapshot = graph.get_state(resolved_config)
        values = getattr(snapshot, "values", {})
        state = cast(SQLAgentState, values if isinstance(values, dict) else {})
    else:
        state = cast(SQLAgentState, {})
    for event in graph.stream(
        initial_state,
        config=resolved_config,
        stream_mode=stream_mode,
    ):
        node_name, update, state = _normalize_stream_event(event, state, stream_mode)
        yield {
            "node": node_name,
            "update": update,
            "state": state,
            "outcome": _summarize_step_outcome(update, state),
        }


def build_sql_agent_graph(
    sql_generator_model: LLM,
    agent_model: ToolCallingChatModel,
    analyst_model: LLM | None = None,
    selector_model: LLM | None = None,
    intent_model: LLM | None = None,
    selected_database: SQLiteDatabase | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
    agent_tool_limits: AgentToolLimits | None = None,
    checkpointer: InMemorySaver | None = None,
) -> CompiledSQLAgentGraph:
    """Build the SQL agent graph

    Args:
        sql_generator_model (LLM): Default text model, used as the intent classifier's fallback.
        agent_model (ToolCallingChatModel): bind_tools-capable model driving the SQL generation agent loop.
        analyst_model (LLM | None, optional): SQL analyser model. Defaults to None.
        selector_model (LLM | None, optional): Database selector model. Defaults to None.
        intent_model (LLM | None, optional): Intent classifier model. Defaults to sql_generator_model.
        selected_database (SQLiteDatabase): The selected database to query.
        validator (SQLSafetyValidator | None, optional): SQL safety validator. Defaults to None.
        execution_limit (int, optional): Maximum rows for query execution. Defaults to 200.
        agent_tool_limits (AgentToolLimits | None, optional): Row/timeout limits for the agent's tools.

    Returns:
        CompiledStateGraph: The compiled graph
    """
    if selected_database is None:
        raise ValueError("selected_database is required")

    agent_validator = validator or SQLSafetyValidator()
    agent_tools = build_agent_tools(selected_database, agent_validator, agent_tool_limits)

    graph = StateGraph(SQLAgentState)
    graph.add_node(NODE_DATABASE_CHECKER, DatabaseCheckerNode(database=selected_database, model=selector_model))
    graph.add_node(NODE_INTENT_CLASSIFIER, IntentClassifierNode(intent_model or sql_generator_model))
    graph.add_node(NODE_ROLE_AUTHORIZER, RoleAuthorizerNode())
    graph.add_node(NODE_SQL_AGENT_LLM, SQLAgentLLMNode(agent_model, agent_tools, selected_database))
    graph.add_node(NODE_SQL_AGENT_TOOLS, SQLAgentToolsNode(agent_tools))
    graph.add_node(NODE_SQL_AGENT_CLARIFY, SQLAgentClarifyNode())
    graph.add_node(NODE_SQL_AGENT_FINALIZE, SQLAgentFinalizeNode())
    graph.add_node(NODE_SQL_VALIDATOR, SQLValidatorNode(agent_validator))
    graph.add_node(NODE_MODIFICATION_VALIDATOR, SQLModificationValidatorNode())
    graph.add_node(NODE_SQL_EXECUTOR, SQLExecutorNode(selected_database, limit=execution_limit))
    graph.add_node(NODE_RESULT_ANALYST, ResultAnalystNode(analyst_model))

    graph.add_edge(START, NODE_DATABASE_CHECKER)
    graph.add_conditional_edges(
        NODE_DATABASE_CHECKER,
        _route_after_database_check,
        {
            NODE_INTENT_CLASSIFIER: NODE_INTENT_CLASSIFIER,
            ROUTE_ABORT: END,
        },
    )
    graph.add_conditional_edges(
        NODE_INTENT_CLASSIFIER,
        _route_from_intent_classifier,
        {
            NODE_ROLE_AUTHORIZER: NODE_ROLE_AUTHORIZER,
            NODE_SQL_AGENT_LLM: NODE_SQL_AGENT_LLM,
        }
    )
    graph.add_conditional_edges(
        NODE_ROLE_AUTHORIZER,
        _route_after_authorization,
        {
            NODE_SQL_AGENT_LLM: NODE_SQL_AGENT_LLM,
            ROUTE_ABORT: END,
        },
    )
    graph.add_conditional_edges(
        NODE_SQL_AGENT_LLM,
        _route_after_agent_llm,
        {
            NODE_SQL_AGENT_FINALIZE: NODE_SQL_AGENT_FINALIZE,
            NODE_SQL_AGENT_CLARIFY: NODE_SQL_AGENT_CLARIFY,
            NODE_SQL_AGENT_TOOLS: NODE_SQL_AGENT_TOOLS,
            ROUTE_ABORT: END,
        },
    )
    graph.add_edge(NODE_SQL_AGENT_TOOLS, NODE_SQL_AGENT_LLM)
    graph.add_conditional_edges(
        NODE_SQL_AGENT_CLARIFY,
        _route_after_clarify,
        {
            NODE_SQL_AGENT_LLM: NODE_SQL_AGENT_LLM,
            ROUTE_ABORT: END,
        },
    )
    graph.add_conditional_edges(
        NODE_SQL_AGENT_FINALIZE,
        _route_after_finalize,
        {
            NODE_SQL_VALIDATOR: NODE_SQL_VALIDATOR,
            NODE_MODIFICATION_VALIDATOR: NODE_MODIFICATION_VALIDATOR,
            ROUTE_ABORT: END,
        },
    )
    graph.add_conditional_edges(
        NODE_SQL_VALIDATOR,
        _route_from_sql_validator,
        {
            NODE_SQL_EXECUTOR: NODE_SQL_EXECUTOR,
            ROUTE_ABORT: END,
        },
    )
    graph.add_conditional_edges(
        NODE_MODIFICATION_VALIDATOR,
        _route_from_modification_validator,
        {
            NODE_SQL_EXECUTOR: NODE_SQL_EXECUTOR,
            ROUTE_ABORT: END,
        },
    )
    graph.add_conditional_edges(
        NODE_SQL_EXECUTOR,
        _route_after_executor,
        {
            NODE_RESULT_ANALYST: NODE_RESULT_ANALYST,
            NODE_SQL_AGENT_LLM: NODE_SQL_AGENT_LLM,
            ROUTE_ABORT: END,
        },
    )
    graph.add_edge(NODE_RESULT_ANALYST, END)
    return cast(
        CompiledSQLAgentGraph,
        graph.compile(checkpointer=checkpointer or InMemorySaver()),
    )

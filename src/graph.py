"""LangGraph orchestration for the SQL copilot agent."""

from __future__ import annotations

from typing import Any, Iterator, Literal, Protocol, TypedDict, cast
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt

from nodes.database_selector import DatabaseSelectorNode
from nodes.intent_classifier import IntentClassifierNode
from nodes.result_analyst import ResultAnalystNode
from nodes.role_authorizer import RoleAuthorizerNode
from nodes.sql_executor import SQLExecutorNode
from nodes.sql_generator import LLM, SQLGeneratorNode
from nodes.sql_validator import SQLValidatorNode
from nodes.sql_modification_validator import SQLModificationValidatorNode
from state import SQLAgentState
from tools.database import RegisteredDatabase, get_default_database_catalog
from tools.sql_safety import SQLSafetyValidator

SQLAgentStateUpdate = SQLAgentState

TraceStreamMode = Literal["updates", "values"]

NODE_DATABASE_SELECTOR = "database_selector"
NODE_INTENT_CLASSIFIER = "intent_classifier"
NODE_ROLE_AUTHORIZER = "role_authorizer"
NODE_SQL_GENERATOR = "sql_generator"
NODE_SQL_VALIDATOR = "sql_validator"
NODE_SQL_EXECUTOR = "sql_executor"
NODE_RESULT_ANALYST = "result_analyst"
NODE_MODIFICATION_VALIDATOR = "modification_validator"
ROUTE_ABORT = "abort"
INTERRUPT_EVENT = "__interrupt__"


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


def _route_after_database_selection(state: SQLAgentState) -> str:
    return ROUTE_ABORT if state.get("execution_error") else NODE_INTENT_CLASSIFIER


def _route_after_authorization(state: SQLAgentState) -> str:
    return ROUTE_ABORT if not state.get("is_authorized") else NODE_SQL_GENERATOR


def _route_after_executor(state: SQLAgentState) -> str:
    if state.get("execution_error"):
        return ROUTE_ABORT
    return NODE_RESULT_ANALYST


def _route_from_database_selector(state: SQLAgentState) -> str:
    return _route_after_database_selection(state)


def _route_from_intent_classifier(state: SQLAgentState) -> str:
    return NODE_ROLE_AUTHORIZER if state["intent"] == "modification" else NODE_SQL_GENERATOR

def _route_after_generation(state: SQLAgentState) -> str:
    return (
        NODE_SQL_VALIDATOR
        if state["intent"] == "query"
        else NODE_MODIFICATION_VALIDATOR
    )

def _route_from_sql_validator(state: SQLAgentState) -> str:
    if state.get("sql_validation_error"):
        return ROUTE_ABORT
    return NODE_SQL_EXECUTOR


def _route_from_sql_executor(state: SQLAgentState) -> str:
    return _route_after_executor(state)


def _route_from_modification_validator(state: SQLAgentState) -> str:
    if state.get("execution_confirmed"):
        return NODE_SQL_EXECUTOR
    return ROUTE_ABORT


def _summarize_step_outcome(update: SQLAgentState) -> str:
    """Summarize the current state of the graph

    Args:
        update (SQLAgentStateUpdate): Graph state

    Returns:
        str: Current state of the graph
    """    
    metadata = update.get("metadata") or {}
    if update.get("authorization_error"):
        return "authorization_failed"
    if update.get("interrupt"):
        return "execution_pending_approval"
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
    if "is_authorized" in update:
        return "User role authorisation"
    if "intent" in update:
        return "Intention classified"
    if "selected_database" in update:
        return "database_selected"
    return "updated"


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
        merged_state = {**current_state, **update}
        return INTERRUPT_EVENT, update, cast(SQLAgentState, merged_state)

    if stream_mode == "updates":
        if not isinstance(event, dict) or len(event) != 1:
            raise TypeError(f"Unexpected graph update event: {event!r}")
        node_name, update = next(iter(event.items()))
        if not isinstance(node_name, str) or not isinstance(update, dict):
            raise TypeError(f"Unexpected graph update event: {event!r}")
        merged_state = {**current_state, **update}
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
            "outcome": _summarize_step_outcome(update),
        }


def build_sql_agent_graph(
    sql_generator_model: LLM,
    analyst_model: LLM | None = None,
    selector_model: LLM | None = None,
    intent_model: LLM | None = None,
    databases: list[RegisteredDatabase] | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
    checkpointer: InMemorySaver | None = None,
) -> CompiledSQLAgentGraph:
    """Build the SQL agent graph

    Args:
        sql_generator_model (LLM): SQL generator model
        analyst_model (LLM | None, optional): SQL analyser model. Defaults to None.
        selector_model (LLM | None, optional): Database selector model. Defaults to None.
        intent_model (LLM | None, optional): Intent classifier model. Defaults to sql_generator_model.
        databases (list[RegisteredDatabase] | None, optional): Database catalog. Defaults to None.
        validator (SQLSafetyValidator | None, optional): SQL safety validator. Defaults to None.
        execution_limit (int, optional): Maximum rows for query execution. Defaults to 200.

    Returns:
        CompiledStateGraph: _description_
    """
    database_catalog = databases or get_default_database_catalog()
    default_database = database_catalog[0].database
    graph = StateGraph(SQLAgentState)
    graph.add_node(NODE_DATABASE_SELECTOR, DatabaseSelectorNode(database_catalog, model=selector_model))
    graph.add_node(NODE_INTENT_CLASSIFIER, IntentClassifierNode(intent_model or sql_generator_model))
    graph.add_node(NODE_ROLE_AUTHORIZER, RoleAuthorizerNode())
    graph.add_node(NODE_SQL_GENERATOR, SQLGeneratorNode(sql_generator_model, default_database, database_catalog))
    graph.add_node(NODE_SQL_VALIDATOR, SQLValidatorNode(validator, database_catalog))
    graph.add_node(NODE_MODIFICATION_VALIDATOR, SQLModificationValidatorNode())
    graph.add_node(NODE_SQL_EXECUTOR, SQLExecutorNode(default_database, database_catalog, limit=execution_limit))
    graph.add_node(NODE_RESULT_ANALYST, ResultAnalystNode(analyst_model))

    graph.add_edge(START, NODE_DATABASE_SELECTOR)
    graph.add_conditional_edges(
        NODE_DATABASE_SELECTOR,
        _route_from_database_selector,
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
            NODE_SQL_GENERATOR: NODE_SQL_GENERATOR,
        }
    )
    graph.add_conditional_edges(
        NODE_ROLE_AUTHORIZER,
        _route_after_authorization,
        {
            NODE_SQL_GENERATOR: NODE_SQL_GENERATOR,
            ROUTE_ABORT: END,
        },
    )
    graph.add_conditional_edges(
        NODE_SQL_GENERATOR,
        _route_after_generation,
        {
            NODE_SQL_VALIDATOR: NODE_SQL_VALIDATOR,
            NODE_MODIFICATION_VALIDATOR: NODE_MODIFICATION_VALIDATOR,
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
            ROUTE_ABORT: END,
        },
    )
    graph.add_edge(NODE_RESULT_ANALYST, END)
    return cast(
        CompiledSQLAgentGraph,
        graph.compile(checkpointer=checkpointer or InMemorySaver()),
    )

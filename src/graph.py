"""LangGraph orchestration for the SQL copilot agent."""

from __future__ import annotations

from typing import Iterator, Literal, Protocol, TypedDict, Any

from langgraph.graph import END, START, StateGraph  
from langgraph.graph.state import CompiledStateGraph 

from nodes.database_selector import DatabaseSelectorNode
from nodes.result_analyst import ResultAnalystNode
from nodes.sql_executor import SQLExecutorNode
from nodes.sql_generator import LLM, SQLGeneratorNode
from nodes.sql_validator import SQLValidatorNode
from state import SQLAgentState
from tools.database import RegisteredDatabase, get_default_database_catalog
from tools.sql_safety import SQLSafetyValidator

SQLAgentStateUpdate = SQLAgentState

TraceStreamMode = Literal["updates", "values"]

NODE_DATABASE_SELECTOR = "database_selector"
NODE_SQL_GENERATOR = "sql_generator"
NODE_SQL_VALIDATOR = "sql_validator"
NODE_SQL_EXECUTOR = "sql_executor"
NODE_RESULT_ANALYST = "result_analyst"
ROUTE_ABORT = "abort"


class CompiledSQLAgentGraph(Protocol):
    """Protocol for the compiled graph methods used by the trace helpers."""

    def stream(
        self,
        state: SQLAgentState,
        *,
        stream_mode: TraceStreamMode = "updates",
    ) -> Iterator[object]:
        """Yield raw LangGraph stream events."""


class SQLAgentTraceStep(TypedDict):
    """Normalized view of a single graph step."""

    node: str
    update: SQLAgentStateUpdate
    state: SQLAgentState
    outcome: str


def _route_after_database_selection(state: SQLAgentState) -> str:
    return ROUTE_ABORT if state.get("execution_error") else NODE_SQL_GENERATOR


def _route_after_validation(state: SQLAgentState) -> str:
    return ROUTE_ABORT if state.get("sql_validation_error") else NODE_SQL_EXECUTOR


def _route_after_execution(state: SQLAgentState) -> str:
    return ROUTE_ABORT if state.get("execution_error") else NODE_RESULT_ANALYST


def _route_from_database_selector(state: SQLAgentState) -> str:
    return _route_after_database_selection(state)


def _route_from_sql_validator(state: SQLAgentState) -> str:
    return _route_after_validation(state)


def _route_from_sql_executor(state: SQLAgentState) -> str:
    return _route_after_execution(state)


def _summarize_step_outcome(update: dict[str, Any]) -> str:
    """Summarize the current state of the graph

    Args:
        update (SQLAgentStateUpdate): Graph state

    Returns:
        str: Current state of the graph
    """    
    metadata = update.get("metadata") or {}
    if update.get("sql_validation_error"):
        return "validation_failed"
    if update.get("execution_error"):
        if metadata.get("database_selection_ambiguous"):
            return "database_selection_ambiguous"
        if metadata.get("database_selection_failed"):
            return "database_selection_failed"
        return "execution_failed"
    if "query_result" in update:
        return "query_executed"
    if "analysis" in update:
        return "analysis_ready" 
    if "generated_sql" in update:
        return "sql_generated"
    if "selected_database" in update:
        return "database_selected"
    return "updated"


def _normalize_stream_event(
    event: object,
    current_state: dict[str, Any],
    stream_mode: TraceStreamMode,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
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
        tuple[str, dict[str, Any], dict[str, Any]]: Normalize state of the graph
    """    
    if stream_mode == "updates":
        if not isinstance(event, dict) or len(event) != 1:
            raise TypeError(f"Unexpected graph update event: {event!r}")
        node_name, update = next(iter(event.items()))
        if not isinstance(node_name, str) or not isinstance(update, dict):
            raise TypeError(f"Unexpected graph update event: {event!r}")
        merged_state = dict(current_state)
        merged_state.update(update)
        return node_name, update, merged_state

    if not isinstance(event, dict):
        raise TypeError(f"Unexpected graph values event: {event!r}")
    return "state", event, event


def stream_sql_agent_execution(
    graph: CompiledSQLAgentGraph,
    initial_state: SQLAgentState,
    *,
    stream_mode: TraceStreamMode = "updates",
) -> Iterator[dict[str, Any]]:
    """Stream the SQL agent execution

    Args:
        graph (CompiledSQLAgentGraph): Compiled graph
        initial_state (SQLAgentState): Initial state of the graph
        stream_mode (TraceStreamMode, optional): Current state of the graph. Defaults to "updates".

    Yields:
        Iterator[dict[str, Any]]: Current state of the graph
    """
    state = dict(initial_state)
    for event in graph.stream(initial_state, stream_mode=stream_mode):
        node_name, update, state = _normalize_stream_event(event, state, stream_mode)
        yield {
            "node": node_name,
            "update": update,
            "state": dict(state),
            "outcome": _summarize_step_outcome(update),
        }


def build_sql_agent_graph(
    sql_generator_model: LLM,
    analyst_model: LLM | None = None,
    selector_model: LLM | None = None,
    databases: list[RegisteredDatabase] | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
) -> CompiledStateGraph:
    """Build the SQL agent graph

    Args:
        sql_generator_model (LLM): SQL generator model
        analyst_model (LLM | None, optional): SQL analyser model. Defaults to None.
        selector_model (LLM | None, optional): Database selector model. Defaults to None.
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
    graph.add_node(NODE_SQL_GENERATOR, SQLGeneratorNode(sql_generator_model, default_database))
    graph.add_node(NODE_SQL_VALIDATOR, SQLValidatorNode(validator))
    graph.add_node(NODE_SQL_EXECUTOR, SQLExecutorNode(default_database, limit=execution_limit))
    graph.add_node(NODE_RESULT_ANALYST, ResultAnalystNode(analyst_model))

    graph.add_edge(START, NODE_DATABASE_SELECTOR)
    graph.add_conditional_edges(
        NODE_DATABASE_SELECTOR,
        _route_from_database_selector,
        {
            NODE_SQL_GENERATOR: NODE_SQL_GENERATOR,
            ROUTE_ABORT: END,
        },
    )
    graph.add_edge(NODE_SQL_GENERATOR, NODE_SQL_VALIDATOR)

    graph.add_conditional_edges(
        NODE_SQL_VALIDATOR,
        _route_from_sql_validator,
        {
            NODE_SQL_EXECUTOR: NODE_SQL_EXECUTOR,
            ROUTE_ABORT: END,
        },
    )
    graph.add_conditional_edges(
        NODE_SQL_EXECUTOR,
        _route_from_sql_executor,
        {
            NODE_RESULT_ANALYST: NODE_RESULT_ANALYST,
            ROUTE_ABORT: END,
        },
    )
    graph.add_edge(NODE_RESULT_ANALYST, END)
    return graph.compile()
